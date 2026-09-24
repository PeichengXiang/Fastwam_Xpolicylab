#!/usr/bin/env python3
"""Fail-fast metadata and FastWAM dataloader smoke test for EgoVLA."""

from __future__ import annotations

import argparse
import json
import subprocess
from pathlib import Path

import numpy as np
import torch
import torchvision.transforms as T
from omegaconf import OmegaConf

from fastwam.datasets.lerobot.robot_video_dataset import RobotVideoDataset
from fastwam.datasets.lerobot.processors.fastwam_processor import FastWAMProcessor
from fastwam.datasets.lerobot.transforms.action_state_merger import ConcatLeftAlign
from fastwam.datasets.lerobot.utils.normalizer import load_dataset_stats_from_json


def first_pts(path: Path) -> float:
    out = subprocess.check_output(
        [
            "ffprobe", "-v", "error", "-select_streams", "v:0",
            "-show_entries", "frame=pts_time", "-read_intervals", "%+#1",
            "-of", "csv=p=0", str(path),
        ],
        text=True,
    )
    token = out.strip().splitlines()[0].strip().rstrip(",")
    return float(token)


def make_dataset(dataset: Path, stats: Path, cache: Path, training: bool) -> RobotVideoDataset:
    shape_meta = {
        "images": [
            {"key": "cam_high", "raw_shape": [3, 384, 384], "shape": [3, 240, 320]},
            {"key": "cam_left_wrist", "raw_shape": [3, 384, 384], "shape": [3, 240, 320]},
            {"key": "cam_right_wrist", "raw_shape": [3, 384, 384], "shape": [3, 240, 320]},
        ],
        "action": [{"key": "default", "raw_shape": 38, "shape": 38}],
        "state": [{"key": "default", "raw_shape": 38, "shape": 38}],
    }
    transforms = [
        T.ConvertImageDtype(torch.float32),
        T.Resize([240, 320], antialias=True),
    ]
    shape_meta_cfg = OmegaConf.create(shape_meta)
    processor = FastWAMProcessor(
        shape_meta=shape_meta_cfg,
        num_obs_steps=33,
        num_output_cameras=3,
        action_output_dim=38,
        proprio_output_dim=38,
        action_state_transforms=None,
        use_stepwise_action_norm=False,
        norm_default_mode="z-score",
        norm_exception_mode=None,
        action_state_merger=ConcatLeftAlign(),
        train_transforms=transforms,
        val_transforms=transforms,
    )
    return RobotVideoDataset(
        dataset_dirs=[str(dataset)],
        shape_meta=shape_meta_cfg,
        num_frames=33,
        video_size=[384, 320],
        processor=processor,
        text_embedding_cache_dir=str(cache),
        pretrained_norm_stats=str(stats),
        val_set_proportion=0.01,
        is_training_set=training,
        action_video_freq_ratio=4,
        concat_multi_camera="robotwin",
    )


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", type=Path, required=True)
    ap.add_argument("--stats", type=Path, required=True)
    ap.add_argument("--cache", type=Path, required=True)
    args = ap.parse_args()
    dataset = args.dataset.resolve(strict=True)
    stats_path = args.stats.resolve(strict=True)
    cache = args.cache.resolve(strict=True)
    info = json.loads((dataset / "meta" / "info.json").read_text(encoding="utf-8"))
    manifest = json.loads((dataset / "meta" / "fastwam_conversion_manifest.json").read_text(encoding="utf-8"))
    stats = load_dataset_stats_from_json(str(stats_path))
    assert info["total_episodes"] == 1903 and info["total_frames"] == 510546
    assert info["fps"] == 30 and info["features"]["action"]["shape"] == [38]
    assert manifest["include_deprecated"] is False and manifest["deprecated_policy"]["excluded_episode_count"] == 100
    assert stats["action"]["default"]["global_mean"].shape[-1] == 38
    assert stats["state"]["default"]["global_mean"].shape[-1] == 38

    parquet_count = len(list((dataset / "data").rglob("episode_*.parquet")))
    video_count = len(list((dataset / "videos").rglob("episode_*.mp4")))
    assert parquet_count == 1903, parquet_count
    assert video_count == 5709, video_count

    # Check one normalized head stream. Missing wrists stay independent black videos.
    sample_video = dataset / "videos" / "chunk-000" / "observation.images.cam_high" / "episode_000000.mp4"
    pts = first_pts(sample_video)
    assert abs(pts) <= 1e-4, pts

    train_ds = make_dataset(dataset, stats_path, cache, True)
    val_ds = make_dataset(dataset, stats_path, cache, False)
    indices = sorted({0, len(train_ds) // 2, len(train_ds) - 1})
    summaries = []
    for idx in indices:
        sample = train_ds._get(idx)
        assert tuple(sample["video"].shape) == (3, 9, 384, 320), sample["video"].shape
        assert tuple(sample["action"].shape) == (32, 38), sample["action"].shape
        assert tuple(sample["proprio"].shape) == (32, 38), sample["proprio"].shape
        assert torch.isfinite(sample["video"]).all()
        assert torch.isfinite(sample["action"]).all()
        assert torch.isfinite(sample["proprio"]).all()
        assert isinstance(sample["prompt"], str) and sample["prompt"]
        summaries.append({"idx": idx, "prompt": sample["prompt"]})
    val_sample = val_ds._get(0)
    assert tuple(val_sample["video"].shape) == (3, 9, 384, 320)
    print(json.dumps({
        "status": "ok",
        "episodes": info["total_episodes"],
        "frames": info["total_frames"],
        "deprecated_excluded": 100,
        "action_dim": 38,
        "train_samples": len(train_ds),
        "val_samples": len(val_ds),
        "video_shape": [3, 9, 384, 320],
        "action_shape": [32, 38],
        "first_pts": pts,
        "sample_prompts": summaries,
    }, ensure_ascii=False, indent=2), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
