#!/usr/bin/env python3
"""Verify FastWAM LeRobot v2.1 joint54 rows against all source HDF5 episodes."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import h5py
import numpy as np
import pyarrow.parquet as pq

TASKS = (
    "collect_objects",
    "dual_bottles_pick",
    "hammer_beat",
    "insert_block",
    "retrieve_gap",
    "stack_bowls",
)
FIELDS = (
    "left_arm_joint_states",
    "left_ee_joint_states",
    "right_arm_joint_states",
    "right_ee_joint_states",
)


def episode_files(root: Path, task: str, limit: int) -> list[Path]:
    data_dir = root / task / "tianji_marvin_wuji" / "data"
    files = sorted(data_dir.glob("episode_*.hdf5")) + sorted(data_dir.glob("episode_*.h5"))
    files = list(dict.fromkeys(path.resolve() for path in files))
    if len(files) != 100:
        raise AssertionError(f"{task}: expected 100 HDF5 episodes, got {len(files)}")
    return files[:limit]


def list_column_to_numpy(column) -> np.ndarray:
    combined = column.combine_chunks()
    return np.asarray(combined.values).reshape(len(combined), -1).astype(np.float32, copy=False)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source-root", type=Path, required=True)
    parser.add_argument("--dataset", type=Path, required=True)
    parser.add_argument("--episodes-per-task", type=int, default=100)
    args = parser.parse_args()
    if not 1 <= args.episodes_per_task <= 100:
        raise ValueError("--episodes-per-task must be in [1, 100]")
    source_root = args.source_root.resolve(strict=True)
    dataset = args.dataset.resolve(strict=True)

    expected_episodes = len(TASKS) * args.episodes_per_task
    expected_frames = 0
    for task in TASKS:
        for source in episode_files(source_root, task, args.episodes_per_task):
            with h5py.File(source, "r") as handle:
                expected_frames += int(handle["state/left_arm_joint_states"].shape[0])

    info = json.loads((dataset / "meta/info.json").read_text(encoding="utf-8"))
    if info["codebase_version"] != "v2.1" or info["robot_type"] != "tianji_marvin_wuji":
        raise AssertionError("unexpected LeRobot version or robot_type")
    if (info["total_episodes"], info["total_frames"], info["total_tasks"]) != (
        expected_episodes,
        expected_frames,
        6,
    ):
        raise AssertionError("unexpected dataset totals")
    for key in ("observation.state", "action"):
        feature = info["features"][key]
        if feature["dtype"] != "float32" or feature["shape"] != [54]:
            raise AssertionError(f"{key}: expected float32[54], got {feature}")
    expected_cameras = {
        "observation.images.cam_high",
        "observation.images.cam_left_wrist",
        "observation.images.cam_right_wrist",
    }
    actual_cameras = {key for key in info["features"] if key.startswith("observation.images.")}
    if actual_cameras != expected_cameras:
        raise AssertionError(f"official FastWAM camera keys mismatch: {actual_cameras}")

    video_files = list((dataset / "videos").rglob("*.mp4"))
    if len(video_files) != expected_episodes * len(expected_cameras):
        raise AssertionError(
            f"expected {expected_episodes * len(expected_cameras)} videos, got {len(video_files)}"
        )

    episodes_meta = [json.loads(line) for line in (dataset / "meta/episodes.jsonl").read_text().splitlines()]
    if len(episodes_meta) != expected_episodes:
        raise AssertionError(
            f"expected {expected_episodes} episode metadata rows, got {len(episodes_meta)}"
        )

    global_index = 0
    episode_index = 0
    for task_index, task in enumerate(TASKS):
        for source in episode_files(source_root, task, args.episodes_per_task):
            parquet = dataset / f"data/chunk-{episode_index // 1000:03d}/episode_{episode_index:06d}.parquet"
            table = pq.read_table(
                parquet,
                columns=["observation.state", "action", "timestamp", "frame_index", "episode_index", "index", "task_index"],
            )
            observation = list_column_to_numpy(table["observation.state"])
            action = list_column_to_numpy(table["action"])
            with h5py.File(source, "r") as handle:
                source_observation = np.concatenate(
                    [np.asarray(handle[f"state/{field}"][:], dtype=np.float32) for field in FIELDS], axis=1
                )
                source_action = np.concatenate(
                    [np.asarray(handle[f"action/{field}"][:], dtype=np.float32) for field in FIELDS], axis=1
                )
                frequency = int(handle["additional_info/frequency"][()])
            length = len(source_observation)
            if frequency != 25 or observation.shape != (length, 54) or action.shape != (length, 54):
                raise AssertionError(f"episode {episode_index}: frequency or shape mismatch")
            if not np.array_equal(observation, source_observation):
                raise AssertionError(f"episode {episode_index}: observation.state differs from HDF5")
            if not np.array_equal(action, source_action):
                raise AssertionError(f"episode {episode_index}: action differs from HDF5")
            if episodes_meta[episode_index]["length"] != length:
                raise AssertionError(f"episode {episode_index}: metadata length mismatch")
            expected_frame = np.arange(length, dtype=np.int64)
            expected_index = np.arange(global_index, global_index + length, dtype=np.int64)
            if not np.array_equal(table["frame_index"].to_numpy(), expected_frame):
                raise AssertionError(f"episode {episode_index}: frame_index mismatch")
            if not np.all(table["episode_index"].to_numpy() == episode_index):
                raise AssertionError(f"episode {episode_index}: episode_index column mismatch")
            if not np.all(table["task_index"].to_numpy() == task_index):
                raise AssertionError(f"episode {episode_index}: task_index column mismatch")
            if not np.array_equal(table["index"].to_numpy(), expected_index):
                raise AssertionError(f"episode {episode_index}: global index mismatch")
            expected_timestamp = expected_frame.astype(np.float32) / 25.0
            if not np.allclose(table["timestamp"].to_numpy(), expected_timestamp, rtol=0, atol=1e-6):
                raise AssertionError(f"episode {episode_index}: timestamp mismatch")
            global_index += length
            episode_index += 1
            if episode_index % 25 == 0:
                print(f"VALIDATED {episode_index}/600", flush=True)

    if global_index != expected_frames:
        raise AssertionError(f"expected {expected_frames} total frames, got {global_index}")
    print(
        f"FASTWAM_LEROBOT_VALIDATION_OK episodes={expected_episodes} "
        f"frames={expected_frames} tasks=6 mode=joint action_dim=54 state_dim=54 fps=25",
        flush=True,
    )


if __name__ == "__main__":
    main()
