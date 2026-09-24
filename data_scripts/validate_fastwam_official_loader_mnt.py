#!/usr/bin/env python3
"""Build the official FastWAM train/val datasets and decode all 12 shards."""

from __future__ import annotations

from pathlib import Path

import torch
from hydra import compose, initialize_config_dir
from hydra.utils import instantiate

from fastwam.utils import misc
from fastwam.utils.config_resolvers import register_default_resolvers


TASKS = (
    "collect_objects",
    "dual_bottles_pick",
    "hammer_beat",
    "insert_block",
    "retrieve_gap",
    "stack_bowls",
)
SHARDS = ("ep000_049", "ep050_099")


def assert_sample(sample: dict, label: str) -> None:
    assert tuple(sample["video"].shape) == (3, 9, 384, 320), (
        label,
        sample["video"].shape,
    )
    assert tuple(sample["action"].shape) == (32, 54), (label, sample["action"].shape)
    assert tuple(sample["proprio"].shape) == (32, 54), (
        label,
        sample["proprio"].shape,
    )
    assert sample["context"].shape[-1] > 0
    for key in ("video", "action", "proprio", "context"):
        assert torch.isfinite(sample[key]).all(), (label, key)
    assert isinstance(sample["prompt"], str) and sample["prompt"]


def main() -> None:
    workspace = Path("/personal/xiangpc/0812_Xpolicylab_bench/FastWAM")
    fw = workspace / "Xpolicylab/policy/FastWAM/FastWAM"
    dataset_dirs = [
        str(workspace / "data" / f"spark0_mnt20260812_fastwam_{task}_{shard}_v21_joint54")
        for task in TASKS
        for shard in SHARDS
    ]
    stats = workspace / "data/spark0_mnt20260812_fastwam_6tasks_v21_joint54/dataset_stats.json"
    cache = fw / "data/text_embeds_cache/xpolicylab/spark0_mnt20260812_fastwam_6tasks_v21_joint54"
    assert len(dataset_dirs) == 12 and all(Path(path).is_dir() for path in dataset_dirs)
    assert stats.is_file()
    assert len(list(cache.glob("*.pt"))) == 6

    register_default_resolvers()
    with initialize_config_dir(version_base="1.3", config_dir=str(fw / "configs")):
        cfg = compose(
            config_name="train",
            overrides=[
                "task=robotwin_uncond_3cam_384_1e-4",
                f"data.train.dataset_dirs={dataset_dirs}",
                f"data.val.dataset_dirs={dataset_dirs}",
                f"data.train.text_embedding_cache_dir={cache}",
                f"data.val.text_embedding_cache_dir={cache}",
                f"data.train.pretrained_norm_stats={stats}",
                f"data.val.pretrained_norm_stats={stats}",
                "data.train.shape_meta.action.0.raw_shape=54",
                "data.train.shape_meta.action.0.shape=54",
                "data.train.shape_meta.state.0.raw_shape=54",
                "data.train.shape_meta.state.0.shape=54",
                "data.val.shape_meta.action.0.raw_shape=54",
                "data.val.shape_meta.action.0.shape=54",
                "data.val.shape_meta.state.0.raw_shape=54",
                "data.val.shape_meta.state.0.shape=54",
                "data.train.processor.action_output_dim=54",
                "data.train.processor.proprio_output_dim=54",
                "data.val.processor.action_output_dim=54",
                "data.val.processor.proprio_output_dim=54",
            ],
        )

    smoke_dir = workspace / "data/spark0_mnt20260812_fastwam_6tasks_v21_joint54/loader_smoke"
    smoke_dir.mkdir(parents=True, exist_ok=True)
    misc.register_work_dir(str(smoke_dir))
    train = instantiate(cfg.data.train)
    val = instantiate(cfg.data.val)

    train_parts = train.lerobot_dataset.multi_dataset._datasets
    val_parts = val.lerobot_dataset.multi_dataset._datasets
    assert len(train_parts) == len(val_parts) == 12
    train_offset = 0
    val_offset = 0
    for index, (train_part, val_part) in enumerate(zip(train_parts, val_parts, strict=True)):
        assert_sample(train[train_offset], f"train_shard_{index}")
        assert_sample(val[val_offset], f"val_shard_{index}")
        train_offset += len(train_part)
        val_offset += len(val_part)
        print(
            f"SHARD_OK {index + 1}/12 train_frames={len(train_part)} "
            f"val_frames={len(val_part)}",
            flush=True,
        )
    assert train_offset == len(train) and val_offset == len(val)
    print(
        f"FASTWAM_OFFICIAL_LOADER_OK datasets=12 source_episodes=600 "
        f"train_frames={len(train)} val_frames={len(val)} action_dim=54 state_dim=54",
        flush=True,
    )


if __name__ == "__main__":
    main()
