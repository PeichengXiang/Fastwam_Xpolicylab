#!/usr/bin/env python3
"""Calculate FastWAM normalization stats with the official dataset classes."""

from __future__ import annotations

import argparse
from pathlib import Path

from fastwam.datasets.lerobot.base_lerobot_dataset import BaseLerobotDataset
from fastwam.datasets.lerobot.processors.fastwam_processor import FastWAMProcessor
from fastwam.datasets.lerobot.transforms.action_state_merger import ConcatLeftAlign
from fastwam.datasets.lerobot.utils.normalizer import save_dataset_stats_to_json


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    dataset = args.dataset.resolve(strict=True)
    if args.output.exists():
        raise FileExistsError(f"refusing to overwrite {args.output}")

    shape_meta = {
        "images": [
            {"key": "cam_high", "raw_shape": [3, 480, 640], "shape": [3, 240, 320]},
            {"key": "cam_left_wrist", "raw_shape": [3, 480, 640], "shape": [3, 240, 320]},
            {"key": "cam_right_wrist", "raw_shape": [3, 480, 640], "shape": [3, 240, 320]},
        ],
        "action": [{"key": "default", "raw_shape": 54, "shape": 54}],
        "state": [{"key": "default", "raw_shape": 54, "shape": 54}],
    }
    raw_dataset = BaseLerobotDataset(
        dataset_dirs=[str(dataset)],
        shape_meta=shape_meta,
        obs_size=33,
        action_size=32,
        val_set_proportion=0.01,
        is_training_set=True,
        global_sample_stride=1,
    )
    processor = FastWAMProcessor(
        shape_meta=shape_meta,
        num_obs_steps=33,
        num_output_cameras=3,
        action_output_dim=54,
        proprio_output_dim=54,
        action_state_transforms=None,
        use_stepwise_action_norm=False,
        norm_default_mode="z-score",
        norm_exception_mode=None,
        action_state_merger=ConcatLeftAlign(),
        train_transforms=None,
        val_transforms=None,
    )
    stats = raw_dataset.get_dataset_stats(processor)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    save_dataset_stats_to_json(stats, str(args.output))
    print(
        f"FASTWAM_STATS_OK output={args.output} episodes={stats['num_episodes']} "
        f"transitions={stats['num_transition']}"
    )


if __name__ == "__main__":
    main()
