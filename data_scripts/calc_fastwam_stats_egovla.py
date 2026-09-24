#!/usr/bin/env python3
"""Generate FastWAM's nested normalization stats for EgoVLA joint-38 data."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from fastwam.datasets.lerobot.base_lerobot_dataset import BaseLerobotDataset
from fastwam.datasets.lerobot.processors.fastwam_processor import FastWAMProcessor
from fastwam.datasets.lerobot.transforms.action_state_merger import ConcatLeftAlign
from fastwam.datasets.lerobot.utils.normalizer import save_dataset_stats_to_json


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--val-proportion", type=float, default=0.01)
    args = parser.parse_args()
    dataset = args.dataset.resolve(strict=True)
    output = args.output.resolve()
    if output.exists():
        raise FileExistsError(f"refusing to overwrite {output}")

    shape_meta = {
        "images": [
            {"key": "cam_high", "raw_shape": [3, 384, 384], "shape": [3, 240, 320]},
            {"key": "cam_left_wrist", "raw_shape": [3, 384, 384], "shape": [3, 240, 320]},
            {"key": "cam_right_wrist", "raw_shape": [3, 384, 384], "shape": [3, 240, 320]},
        ],
        "action": [{"key": "default", "raw_shape": 38, "shape": 38}],
        "state": [{"key": "default", "raw_shape": 38, "shape": 38}],
    }
    raw_dataset = BaseLerobotDataset(
        dataset_dirs=[str(dataset)],
        shape_meta=shape_meta,
        obs_size=33,
        action_size=32,
        val_set_proportion=args.val_proportion,
        is_training_set=True,
        global_sample_stride=1,
    )
    processor = FastWAMProcessor(
        shape_meta=shape_meta,
        num_obs_steps=33,
        num_output_cameras=3,
        action_output_dim=38,
        proprio_output_dim=38,
        action_state_transforms=None,
        use_stepwise_action_norm=False,
        norm_default_mode="z-score",
        norm_exception_mode=None,
        action_state_merger=ConcatLeftAlign(),
        train_transforms=None,
        val_transforms=None,
    )
    stats = raw_dataset.get_dataset_stats(processor)
    if stats["state"]["default"]["global_mean"].shape[-1] != 38:
        raise ValueError("generated state stats are not 38-dimensional")
    if stats["action"]["default"]["global_mean"].shape[-1] != 38:
        raise ValueError("generated action stats are not 38-dimensional")
    output.parent.mkdir(parents=True, exist_ok=True)
    save_dataset_stats_to_json(stats, str(output))
    # A small sidecar makes the split and source semantics auditable without
    # changing the format consumed by FastWAM.
    sidecar = output.with_name(output.stem + ".manifest.json")
    sidecar.write_text(
        json.dumps(
            {
                "dataset": str(dataset),
                "val_set_proportion": args.val_proportion,
                "stats_action_source": "action column (raw commanded action)",
                "action_dim": 38,
                "num_episodes": int(stats["num_episodes"]),
                "num_transition": int(stats["num_transition"]),
            },
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )
    print(
        f"FASTWAM_STATS_OK output={output} episodes={stats['num_episodes']} "
        f"transitions={stats['num_transition']}",
        flush=True,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
