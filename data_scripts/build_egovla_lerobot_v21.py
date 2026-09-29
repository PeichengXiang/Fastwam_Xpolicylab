#!/usr/bin/env python3
"""Build the FastWAM LeRobot v2.1 dataset directly from EgoVLA HDF5.

The FastWAM checkout contains a v2.1 reader, while the newer EgoVLA converter
produces v3 metadata.  This small builder uses the vendored v2.1 writer for
the tabular metadata and encodes each RGB stream with ffmpeg.  Labels always
come from the raw HDF5 ``/action`` dataset; qpos is stored only as state.
"""

from __future__ import annotations

import argparse
import json
import os
import re
import shutil
import subprocess
import tempfile
from pathlib import Path

import h5py
import numpy as np

from fastwam.datasets.lerobot.lerobot.lerobot_dataset import LeRobotDataset


FPS = 30
IMAGE_SIZE = (384, 384)
ACTION_DIM = 38
POLICY_JOINT_INDICES = (
    4, 8, 12, 16, 20, 22, 24,
    26, 36, 27, 37, 28, 38, 29, 39, 30, 40, 46, 48,
    5, 9, 13, 17, 21, 23, 25,
    31, 41, 32, 42, 33, 43, 34, 44, 35, 45, 47, 49,
)
TASK_INSTRUCTIONS = {
    "Close-Drawer": "Close the opened drawer",
    "Flip-Mug": "Flip the mug",
    "Insert-And-Unload-Cans": (
        "Insert the left can into the slot and insert the right can into the slot, "
        "unload the left cans and then unload the right cans"
    ),
    "Insert-Cans": "Insert cans into the boxes",
    "Open-Drawer": "Open the closed drawer",
    "Open-Laptop": "open the laptop",
    "Pour-Balls": "pour balls in cup into bowl",
    "Push-Box": "push box to the marker",
    "Sort-Cans": "Put sprite cans to the left box, and orange cans to the right box",
    "Stack-Can": "put can on the saucer",
    "Stack-Can-Into-Drawer": "Open the drawer, and Put can on the saucer",
    "Unload-Cans": "unload the right cans and then unload the left cans",
}


def natural_key(path: Path) -> tuple[int, str]:
    match = re.search(r"episode_(\d+)$", path.stem)
    return (int(match.group(1)) if match else 10**18, path.as_posix())


def discover(root: Path) -> list[tuple[str, Path, int, bool]]:
    plans: list[tuple[str, Path, int, bool]] = []
    for task in sorted(TASK_INSTRUCTIONS):
        task_root = root / task
        if not task_root.is_dir():
            raise FileNotFoundError(task_root)
        files = sorted(task_root.rglob("episode_*.hdf5"), key=natural_key)
        for path in files:
            # Only path components below the task directory can designate a
            # deprecated trajectory.  The raw root itself contains the word
            # "deprecated" by design.
            rel_parts = path.relative_to(task_root).parts
            if any("deprecated" in part.casefold() for part in rel_parts):
                continue
            with h5py.File(path, "r") as h5:
                action = h5["action"]
                qpos = h5["observations/qpos"]
                main = h5["observations/images/main"]
                if action.shape != qpos.shape or action.ndim != 2 or action.shape[1] != 50:
                    raise ValueError(f"{path}: action/qpos shape mismatch {action.shape}/{qpos.shape}")
                if main.shape != (action.shape[0], *IMAGE_SIZE, 3) or main.dtype != np.uint8:
                    raise ValueError(f"{path}: main image shape/dtype {main.shape}/{main.dtype}")
                has_wrist = (
                    "observations/images/left_hand" in h5
                    and "observations/images/right_hand" in h5
                )
                if has_wrist:
                    for key in ("left_hand", "right_hand"):
                        ds = h5[f"observations/images/{key}"]
                        if ds.shape != main.shape or ds.dtype != np.uint8:
                            raise ValueError(f"{path}: invalid {key} image dataset")
                plans.append((task, path, int(action.shape[0]), has_wrist))
    return plans


def start_encoder(path: Path) -> subprocess.Popen:
    path.parent.mkdir(parents=True, exist_ok=True)
    return subprocess.Popen(
        [
            "ffmpeg", "-hide_banner", "-loglevel", "error", "-y",
            "-f", "rawvideo", "-pix_fmt", "rgb24",
            "-s:v", f"{IMAGE_SIZE[1]}x{IMAGE_SIZE[0]}", "-r", str(FPS), "-i", "-",
            "-an", "-c:v", "libx264", "-preset", "veryfast", "-crf", "18",
            "-pix_fmt", "yuv420p", "-movflags", "+faststart", str(path),
        ],
        stdin=subprocess.PIPE,
    )


def finish_encoder(proc: subprocess.Popen, path: Path) -> None:
    assert proc.stdin is not None
    proc.stdin.close()
    rc = proc.wait()
    if rc != 0:
        raise RuntimeError(f"ffmpeg failed for {path} with exit code {rc}")
    if not path.is_file() or path.stat().st_size == 0:
        raise RuntimeError(f"ffmpeg produced no video at {path}")


def features() -> dict[str, dict[str, object]]:
    names = [f"joint_{i}" for i in range(ACTION_DIM)]
    out: dict[str, dict[str, object]] = {
        "observation.state": {"dtype": "float32", "shape": (ACTION_DIM,), "names": names},
        "action": {"dtype": "float32", "shape": (ACTION_DIM,), "names": names},
        "provenance.raw_commanded_action": {
            "dtype": "float32", "shape": (ACTION_DIM,), "names": names,
        },
        "observation.camera_mask": {
            "dtype": "float32", "shape": (3,),
            "names": ["cam_high", "cam_left_wrist", "cam_right_wrist"],
        },
    }
    for key in (
        "observation.images.cam_high",
        "observation.images.cam_left_wrist",
        "observation.images.cam_right_wrist",
    ):
        out[key] = {
            "dtype": "video", "shape": (3, IMAGE_SIZE[0], IMAGE_SIZE[1]),
            "names": ["channel", "height", "width"],
        }
    return out


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--raw-root", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--max-episodes", type=int, default=None)
    args = parser.parse_args()
    raw_root = args.raw_root.resolve(strict=True)
    output_root = args.output_root.resolve()
    if output_root.exists() and any(output_root.iterdir()):
        raise FileExistsError(f"refusing to overwrite non-empty output: {output_root}")
    output_root.parent.mkdir(parents=True, exist_ok=True)
    staging = output_root.parent / f".{output_root.name}.source-staging"
    if staging.exists():
        raise FileExistsError(f"staging path exists: {staging}")

    plans = discover(raw_root)
    if args.max_episodes is not None:
        plans = plans[: args.max_episodes]
    if not plans:
        raise ValueError("no active episodes found")
    print(f"[plan] episodes={len(plans)} frames={sum(p[2] for p in plans)}", flush=True)

    dataset = None
    try:
        dataset = LeRobotDataset.create(
            repo_id="EgoVLA_benchmark_fastwam_v21_joint38_cmd_source",
            fps=FPS,
            robot_type="ego_h1_inspire",
            features=features(),
            root=staging,
            use_videos=True,
            video_codec="h264",
            is_compute_episode_stats_image=False,
        )
        black = np.zeros((*IMAGE_SIZE, 3), dtype=np.uint8)
        for ep_index, (task, source, frames, has_wrist) in enumerate(plans):
            print(f"[episode] {ep_index + 1}/{len(plans)} {task} {source.name} frames={frames} wrist={has_wrist}", flush=True)
            chunk = ep_index // 1000
            video_paths = {
                "observation.images.cam_high": staging / f"videos/chunk-{chunk:03d}/observation.images.cam_high/episode_{ep_index:06d}.mp4",
                "observation.images.cam_left_wrist": staging / f"videos/chunk-{chunk:03d}/observation.images.cam_left_wrist/episode_{ep_index:06d}.mp4",
                "observation.images.cam_right_wrist": staging / f"videos/chunk-{chunk:03d}/observation.images.cam_right_wrist/episode_{ep_index:06d}.mp4",
            }
            encoders = {key: start_encoder(path) for key, path in video_paths.items()}
            try:
                with h5py.File(source, "r") as h5:
                    qpos = h5["observations/qpos"]
                    action = h5["action"]
                    main = h5["observations/images/main"]
                    left = h5.get("observations/images/left_hand") if has_wrist else None
                    right = h5.get("observations/images/right_hand") if has_wrist else None
                    camera_mask = np.asarray([1.0, float(has_wrist), float(has_wrist)], dtype=np.float32)
                    for index in range(frames):
                        state_vec = np.asarray(qpos[index], dtype=np.float32)[list(POLICY_JOINT_INDICES)]
                        action_vec = np.asarray(action[index], dtype=np.float32)[list(POLICY_JOINT_INDICES)]
                        if state_vec.shape != (ACTION_DIM,) or action_vec.shape != (ACTION_DIM,):
                            raise ValueError(f"{source}: bad projected vector at frame {index}")
                        head = np.asarray(main[index], dtype=np.uint8)
                        l_frame = np.asarray(left[index], dtype=np.uint8) if left is not None else black
                        r_frame = np.asarray(right[index], dtype=np.uint8) if right is not None else black
                        encoders["observation.images.cam_high"].stdin.write(head.tobytes())
                        encoders["observation.images.cam_left_wrist"].stdin.write(l_frame.tobytes())
                        encoders["observation.images.cam_right_wrist"].stdin.write(r_frame.tobytes())
                        dataset.add_frame(
                            {
                                "observation.state": state_vec,
                                "action": action_vec,
                                "provenance.raw_commanded_action": action_vec.copy(),
                                "observation.camera_mask": camera_mask,
                                "observation.images.cam_high": head,
                                "observation.images.cam_left_wrist": l_frame,
                                "observation.images.cam_right_wrist": r_frame,
                            },
                            task=[TASK_INSTRUCTIONS[task]] * 4,
                        )
                for key, proc in encoders.items():
                    finish_encoder(proc, video_paths[key])
            except Exception:
                for proc in encoders.values():
                    if proc.poll() is None:
                        proc.kill()
                raise

            # We encoded videos directly; skip the vendored writer's image-dir
            # encoding step and return the paths that already exist.
            dataset.encode_episode_videos = lambda _ep, paths={k: str(v) for k, v in video_paths.items()}: paths
            dataset.save_episode()

    finally:
        if dataset is not None:
            dataset.stop_image_writer()

    task_episode_counts = {}
    for task, _source, _frames, _has_wrist in plans:
        task_episode_counts[task] = task_episode_counts.get(task, 0) + 1
    manifest = {
        "converter": "build_egovla_lerobot_v21.py",
        "converter_version": "1.0.0",
        "source_root": str(raw_root),
        "include_deprecated": False,
        "total_episodes": len(plans),
        "total_frames": sum(p[2] for p in plans),
        "task_episode_counts": task_episode_counts,
        "fps": FPS,
        "action_dim": ACTION_DIM,
        "action_source": "raw commanded action projected by POLICY_JOINT_INDICES",
        "policy_joint_indices_in_raw_50d": list(POLICY_JOINT_INDICES),
        "episodes": [
            {
                "dataset_episode_index": index,
                "source_relative_path": source.relative_to(raw_root).as_posix(),
                "source_episode_index": int(re.search(r"episode_(\d+)", source.stem).group(1)),
                "task": task,
                "frames": frames,
                "has_real_wrist_cameras": has_wrist,
            }
            for index, (task, source, frames, has_wrist) in enumerate(plans)
        ],
    }
    (staging / "meta" / "xpolicylab_source_conversion.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    staging.rename(output_root)
    print(f"[done] output={output_root}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
