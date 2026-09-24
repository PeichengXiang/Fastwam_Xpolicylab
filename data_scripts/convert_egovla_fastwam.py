#!/usr/bin/env python3
"""Materialize the active EgoVLA episodes as a FastWAM-ready LeRobot v2.1 set.

The raw EgoVLA recordings are HDF5 files, while the FastWAM dataloader consumes
LeRobot v2.1.  A separately audited v2.1 conversion is available on the same
JuiceFS volume.  This utility makes a hard-link based, controlled copy of that
conversion under FastWAM/data, verifies it against the raw HDF5 manifest, and
switches the training action from the Pi05 ``next state`` convention back to
the commanded action used by FastWAM.  EgoVLA tasks without a wrist camera
keep the source black wrist placeholders.  Do not copy the head view into
those slots.

The operation is intentionally fail-closed: an existing output is never
modified, deprecated paths are rejected, and all metadata edits are atomic.
"""

from __future__ import annotations

import argparse
import copy
import datetime as dt
import concurrent.futures
import json
import os
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path

import h5py
import numpy as np
import pyarrow.parquet as pq


POLICY_JOINT_INDICES = (
    4, 8, 12, 16, 20, 22, 24,
    26, 36, 27, 37, 28, 38, 29, 39, 30, 40, 46, 48,
    5, 9, 13, 17, 21, 23, 25,
    31, 41, 32, 42, 33, 43, 34, 44, 35, 45, 47, 49,
)

EXPECTED_COUNTS = {
    "Close-Drawer": 50,
    "Flip-Mug": 100,
    "Insert-And-Unload-Cans": 900,
    "Insert-Cans": 100,
    "Open-Drawer": 100,
    "Open-Laptop": 100,
    "Pour-Balls": 102,
    "Push-Box": 100,
    "Sort-Cans": 101,
    "Stack-Can": 100,
    "Stack-Can-Into-Drawer": 50,
    "Unload-Cans": 100,
}
EXPECTED_EPISODES = 1903
EXPECTED_FRAMES = 510546


def natural_episode_key(path: Path) -> tuple[int, str]:
    stem = path.stem
    try:
        return int(stem.rsplit("_", 1)[1]), str(path)
    except (IndexError, ValueError):
        return (10**18, str(path))


def is_deprecated(path: Path) -> bool:
    return any("deprecated" in part.lower() for part in path.parts)


def discover_raw(raw_root: Path) -> tuple[list[dict], dict[str, int], int]:
    episodes: list[dict] = []
    counts: dict[str, int] = {}
    total_frames = 0
    for task_dir in sorted(p for p in raw_root.iterdir() if p.is_dir() and p.name != ".cache"):
        all_files = sorted(task_dir.rglob("episode_*.hdf5"), key=natural_episode_key)
        active = [p for p in all_files if not is_deprecated(p)]
        deprecated = [p for p in all_files if is_deprecated(p)]
        if deprecated:
            print(f"[info] excluding {len(deprecated)} deprecated files under {task_dir.name}")
        if not active:
            continue
        counts[task_dir.name] = len(active)
        for p in active:
            rel = p.relative_to(raw_root).as_posix()
            with h5py.File(p, "r") as h5:
                if "action" not in h5 or "observations/qpos" not in h5:
                    raise ValueError(f"missing action/qpos in {p}")
                action_shape = tuple(h5["action"].shape)
                qpos_shape = tuple(h5["observations/qpos"].shape)
                if len(action_shape) != 2 or action_shape[1] != 50:
                    raise ValueError(f"unexpected action shape {action_shape} in {p}")
                if qpos_shape != action_shape:
                    raise ValueError(f"qpos/action shape mismatch {qpos_shape} vs {action_shape} in {p}")
                has_wrist = (
                    "observations/images/left_hand" in h5
                    and "observations/images/right_hand" in h5
                )
                frames = int(action_shape[0])
            total_frames += frames
            ep_num = natural_episode_key(p)[0]
            episodes.append({
                "source_relative_path": rel,
                "source_episode_index": ep_num,
                "task": task_dir.name,
                "frames": frames,
                "has_real_wrist_cameras": has_wrist,
            })
    return episodes, counts, total_frames


def read_json(path: Path):
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def write_json_atomic(path: Path, obj) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp_name = tempfile.mkstemp(prefix=path.name + ".", suffix=".tmp", dir=str(path.parent))
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as f:
            json.dump(obj, f, ensure_ascii=False, indent=2)
            f.write("\n")
            f.flush()
            os.fsync(f.fileno())
        os.replace(tmp_name, path)
    finally:
        if os.path.exists(tmp_name):
            os.unlink(tmp_name)


def write_jsonl_atomic(path: Path, rows: list[dict]) -> None:
    fd, tmp_name = tempfile.mkstemp(prefix=path.name + ".", suffix=".tmp", dir=str(path.parent))
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as f:
            for row in rows:
                f.write(json.dumps(row, ensure_ascii=False, separators=(",", ":")) + "\n")
            f.flush()
            os.fsync(f.fileno())
        os.replace(tmp_name, path)
    finally:
        if os.path.exists(tmp_name):
            os.unlink(tmp_name)


def copy_tree_hardlink(src: Path, dst: Path) -> None:
    """Copy a tree with hardlinks, falling back to regular copies on EXDEV."""
    dst.mkdir(parents=True, exist_ok=False)
    fallback = False
    for root, dirs, files in os.walk(src):
        rel = Path(root).relative_to(src)
        out_dir = dst / rel
        out_dir.mkdir(parents=True, exist_ok=True)
        for d in dirs:
            (out_dir / d).mkdir(parents=True, exist_ok=True)
        for name in files:
            s, d = Path(root) / name, out_dir / name
            try:
                os.link(s, d)
            except OSError as exc:
                if exc.errno != 18:  # EXDEV
                    raise
                fallback = True
                shutil.copy2(s, d)
    if fallback:
        print("[warning] source/output are on different filesystems; regular copies were used")


def load_manifest(source_root: Path) -> dict:
    p = source_root / "meta" / "xpolicylab_source_conversion.json"
    if not p.is_file():
        raise FileNotFoundError(f"missing audited source manifest: {p}")
    return read_json(p)


def validate_manifest(raw_eps: list[dict], raw_counts: dict[str, int], raw_frames: int, source_manifest: dict) -> None:
    if raw_counts != EXPECTED_COUNTS:
        raise ValueError(f"active task counts differ: {raw_counts!r} != {EXPECTED_COUNTS!r}")
    if len(raw_eps) != EXPECTED_EPISODES or raw_frames != EXPECTED_FRAMES:
        raise ValueError(f"raw active set is {len(raw_eps)} episodes/{raw_frames} frames; expected {EXPECTED_EPISODES}/{EXPECTED_FRAMES}")
    if source_manifest.get("include_deprecated") is not False:
        raise ValueError("source conversion does not explicitly exclude deprecated episodes")
    if source_manifest.get("total_episodes") != EXPECTED_EPISODES or source_manifest.get("total_frames") != EXPECTED_FRAMES:
        raise ValueError("audited source manifest totals do not match the active raw set")
    src_eps = source_manifest.get("episodes")
    if not isinstance(src_eps, list) or len(src_eps) != EXPECTED_EPISODES:
        raise ValueError("audited source manifest has an unexpected episode list")
    raw_by_rel = {e["source_relative_path"]: e for e in raw_eps}
    for e in src_eps:
        rel = e.get("source_relative_path", "")
        if "deprecated" in rel.lower() or rel not in raw_by_rel:
            raise ValueError(f"source manifest contains a non-active/missing path: {rel}")
        r = raw_by_rel[rel]
        if int(e.get("frames", -1)) != r["frames"]:
            raise ValueError(f"frame mismatch for {rel}")
    if {e["source_relative_path"] for e in src_eps} != set(raw_by_rel):
        raise ValueError("source manifest and active raw path sets differ")


def rewrite_actions_and_validate(output_root: Path, raw_root: Path, source_manifest: dict) -> tuple[float, int]:
    """Replace action with commanded action and compare every row to raw HDF5."""
    data_root = output_root / "data"
    entries = {int(e["dataset_episode_index"]): e for e in source_manifest["episodes"]}
    max_diff = 0.0
    rows = 0
    parquet_files = sorted(data_root.rglob("episode_*.parquet"))
    if len(parquet_files) != EXPECTED_EPISODES:
        raise ValueError(f"expected {EXPECTED_EPISODES} parquet files, found {len(parquet_files)}")
    for parquet in parquet_files:
        ep_idx = int(parquet.stem.rsplit("_", 1)[1])
        if ep_idx not in entries:
            raise ValueError(f"unexpected parquet episode {ep_idx}")
        entry = entries[ep_idx]
        raw_path = raw_root / entry["source_relative_path"]
        with h5py.File(raw_path, "r") as h5:
            # h5py requires increasing fancy-index lists; the policy layout
            # intentionally interleaves the two finger-joint groups.  Read
            # the compact 50-D arrays, then project with NumPy.
            raw_action = np.asarray(h5["action"], dtype=np.float32)[:, POLICY_JOINT_INDICES]
            raw_state = np.asarray(h5["observations/qpos"], dtype=np.float32)[:, POLICY_JOINT_INDICES]
        table = pq.read_table(parquet)
        if "action" not in table.column_names or "provenance.raw_commanded_action" not in table.column_names:
            raise ValueError(f"required action provenance columns missing in {parquet}")
        action = np.asarray(table["action"].to_pylist(), dtype=np.float32)
        state = np.asarray(table["observation.state"].to_pylist(), dtype=np.float32)
        commanded = np.asarray(table["provenance.raw_commanded_action"].to_pylist(), dtype=np.float32)
        if action.shape != raw_action.shape or state.shape != raw_state.shape or commanded.shape != raw_action.shape:
            raise ValueError(f"shape mismatch in {parquet}: action={action.shape}, state={state.shape}, raw={raw_action.shape}")
        diff = float(np.max(np.abs(commanded - raw_action)))
        max_diff = max(max_diff, diff)
        if diff > 2e-5:
            raise ValueError(f"raw commanded action mismatch {diff} in {parquet}")
        state_diff = float(np.max(np.abs(state - raw_state)))
        if state_diff > 2e-5:
            raise ValueError(f"raw state mismatch {state_diff} in {parquet}")
        # Set the canonical action column to the commanded action.  Write via a
        # sibling temporary file because source files may be hard-linked.
        action_col = table["provenance.raw_commanded_action"]
        table = table.set_column(table.schema.get_field_index("action"), "action", action_col)
        fd, tmp_name = tempfile.mkstemp(prefix=parquet.name + ".", suffix=".tmp", dir=str(parquet.parent))
        os.close(fd)
        try:
            pq.write_table(table, tmp_name, compression="zstd")
            os.replace(tmp_name, parquet)
        finally:
            if os.path.exists(tmp_name):
                os.unlink(tmp_name)
        rows += int(raw_action.shape[0])
    return max_diff, rows


def assert_missing_wrists_stay_black(output_root: Path, source_manifest: dict) -> int:
    """Refuse to train on head frames that were copied into absent wrist slots."""
    info = read_json(output_root / "meta" / "info.json")
    chunks_size = int(info.get("chunks_size", 1000))
    checked = 0
    for entry in source_manifest["episodes"]:
        if entry.get("has_real_wrist_cameras", False):
            continue
        ep = int(entry["dataset_episode_index"])
        chunk = ep // chunks_size
        stem = f"episode_{ep:06d}.mp4"
        base = output_root / "videos" / f"chunk-{chunk:03d}"
        high = base / "observation.images.cam_high" / stem
        if not high.is_file():
            raise FileNotFoundError(high)
        for key in ("observation.images.cam_left_wrist", "observation.images.cam_right_wrist"):
            target = base / key / stem
            if not target.is_file():
                raise FileNotFoundError(target)
            if os.path.samefile(high, target):
                raise ValueError(
                    f"{target} is the head video; EgoVLA missing wrists must stay black frames"
                )
            checked += 1
    return checked


def repair_video_timestamps(output_root: Path, source_manifest: dict, workers: int) -> int:
    """Stream-copy videos with a zero-based PTS without re-encoding pixels.

    The audited source was produced with ``-avoid_negative_ts 1``.  That
    leaves the first decoded frame at about 0.033 s, which is outside the
    vendored FastWAM reader's 1e-4 s timestamp tolerance.  A plain stream
    copy with ``-avoid_negative_ts disabled`` restores the original 0,
    1/fps, ... timeline and is lossless for the H.264 packets.
    """
    jobs: list[Path] = []
    info = read_json(output_root / "meta" / "info.json")
    chunks_size = int(info.get("chunks_size", 1000))
    for entry in source_manifest["episodes"]:
        ep = int(entry["dataset_episode_index"])
        # Repair every camera, including black wrist placeholders.  Those
        # placeholders are independent files and must not be replaced by cam_high.
        keys = [
            "observation.images.cam_high",
            "observation.images.cam_left_wrist",
            "observation.images.cam_right_wrist",
        ]
        chunk = ep // chunks_size
        stem = f"episode_{ep:06d}.mp4"
        for key in keys:
            p = output_root / "videos" / f"chunk-{chunk:03d}" / key / stem
            if not p.is_file():
                raise FileNotFoundError(p)
            jobs.append(p)

    def one(path: Path) -> None:
        fd, tmp_name = tempfile.mkstemp(prefix=path.name + ".pts.", suffix=".mp4", dir=str(path.parent))
        os.close(fd)
        try:
            # Do not use -fflags +genpts: it can retain the one-frame offset
            # in some ffmpeg builds.  The muxer option below shifts timestamps
            # while copying the compressed packets verbatim.
            cmd = [
                "ffmpeg", "-hide_banner", "-loglevel", "error", "-y",
                "-i", str(path), "-map", "0", "-c", "copy",
                "-avoid_negative_ts", "disabled", tmp_name,
            ]
            subprocess.run(cmd, check=True, stdout=subprocess.DEVNULL, stderr=subprocess.PIPE)
            os.replace(tmp_name, path)
        except Exception:
            if os.path.exists(tmp_name):
                os.unlink(tmp_name)
            raise

    workers = max(1, int(workers))
    with concurrent.futures.ThreadPoolExecutor(max_workers=workers) as pool:
        futures = [pool.submit(one, p) for p in jobs]
        for i, fut in enumerate(concurrent.futures.as_completed(futures), 1):
            fut.result()
            if i % 250 == 0 or i == len(jobs):
                print(f"[video] repaired {i}/{len(jobs)} streams", flush=True)
    return len(jobs)


def rewrite_episode_stats(output_root: Path, source_manifest: dict) -> None:
    """Keep LeRobot metadata stats consistent with the commanded action/videos."""
    path = output_root / "meta" / "episodes_stats.jsonl"
    rows = []
    by_ep = {int(e["dataset_episode_index"]): e for e in source_manifest["episodes"]}
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            row = json.loads(line)
            ep = int(row["episode_index"])
            stats = row.get("stats", {})
            raw_stats = stats.get("provenance.raw_commanded_action")
            if raw_stats is not None:
                stats["action"] = copy.deepcopy(raw_stats)
            # Absent wrists stay the source black-frame stats.  Copying cam_high
            # stats here was the bug that treated the head view as a wrist view.
            if ep not in by_ep:
                raise ValueError(f"episodes_stats row has no source manifest entry: {ep}")
            row["stats"] = stats
            rows.append(row)
    if len(rows) != EXPECTED_EPISODES:
        raise ValueError("unexpected episodes_stats row count")
    write_jsonl_atomic(path, rows)


def aggregate_standard_stats(output_root: Path) -> None:
    """Regenerate the flat LeRobot stats.json from episode stats (best effort)."""
    path = output_root / "meta" / "episodes_stats.jsonl"
    by_key: dict[str, list[dict]] = {}
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            stats = json.loads(line)["stats"]
            for key, value in stats.items():
                if not isinstance(value, dict) or "count" not in value:
                    continue
                by_key.setdefault(key, []).append(value)
    out = {}
    for key, vals in by_key.items():
        counts = np.asarray([v["count"][0] for v in vals], dtype=np.float64)
        total = float(counts.sum())
        means = np.asarray([v["mean"] for v in vals], dtype=np.float64)
        stds = np.asarray([v["std"] for v in vals], dtype=np.float64)
        shape = (len(vals),) + (1,) * (means.ndim - 1)
        weights = counts.reshape(shape)
        mean = (means * weights).sum(axis=0) / total
        var = (((stds**2) + (means - mean) ** 2) * weights).sum(axis=0) / total
        out[key] = {
            "min": np.min(np.asarray([v["min"] for v in vals]), axis=0).tolist(),
            "max": np.max(np.asarray([v["max"] for v in vals]), axis=0).tolist(),
            "mean": mean.tolist(),
            "std": np.sqrt(np.maximum(var, 0)).tolist(),
            "count": [int(total)],
        }
    write_json_atomic(output_root / "meta" / "stats.json", out)


def write_provenance(output_root: Path, raw_root: Path, source_root: Path, source_manifest: dict, max_diff: float, rows: int, repaired_videos: int) -> None:
    payload = {
        "converter": "convert_egovla_fastwam.py",
        "converter_version": "1.0.0",
        "converted_at_utc": dt.datetime.now(dt.timezone.utc).isoformat(),
        "source_raw_root": str(raw_root),
        "source_lerobot_root": str(source_root),
        "include_deprecated": False,
        "deprecated_policy": {
            "excluded_paths": [
                "Insert-Cans/Left_Right_Deprecated",
                "Sort-Cans/Single_hand_first_deprecated",
            ],
            "excluded_episode_count": 100,
        },
        "total_episodes": EXPECTED_EPISODES,
        "total_frames": EXPECTED_FRAMES,
        "action_dim": 38,
        "fps": 30,
        "action_source": "raw commanded action projected by POLICY_JOINT_INDICES",
        "policy_joint_indices_in_raw_50d": list(POLICY_JOINT_INDICES),
        "wrist_policy": "missing wrist videos stay black placeholders; camera_mask remains false",
        "video_timestamp_policy": "lossless ffmpeg stream-copy with -avoid_negative_ts disabled; zero-based PTS",
        "video_streams_repaired": repaired_videos,
        "validated_commanded_action_max_abs_diff": max_diff,
        "validated_rows": rows,
        "source_manifest_sha256_note": "validated against source xpolicylab_source_conversion.json",
        "source_task_counts": source_manifest.get("task_episode_counts", {}),
    }
    write_json_atomic(output_root / "meta" / "fastwam_conversion_manifest.json", payload)


def make_lerobot_compat_view(output_root: Path) -> None:
    """Expose the root data through the conventional ``<id>/lerobot`` path."""
    view = output_root / "lerobot"
    view.mkdir()
    for name in ("meta", "data", "videos"):
        os.symlink("../" + name, view / name)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--raw-root", type=Path, required=True)
    ap.add_argument("--source-lerobot", type=Path, required=True)
    ap.add_argument("--output-root", type=Path, required=True)
    ap.add_argument("--video-workers", type=int, default=16)
    ap.add_argument("--force", action="store_true", help="remove a prior incomplete output only when its marker is present")
    args = ap.parse_args()
    raw_root = args.raw_root.resolve()
    source_root = args.source_lerobot.resolve()
    output_root = args.output_root.resolve()
    if not raw_root.is_dir() or not source_root.is_dir():
        raise FileNotFoundError("raw/source root does not exist")
    source_manifest = load_manifest(source_root)
    raw_eps, raw_counts, raw_frames = discover_raw(raw_root)
    validate_manifest(raw_eps, raw_counts, raw_frames, source_manifest)
    print(f"[ok] active raw set: {len(raw_eps)} episodes, {raw_frames} frames; deprecated excluded=100")
    if output_root.exists():
        marker = output_root / "meta" / "fastwam_conversion_manifest.json"
        if not args.force or not marker.is_file():
            raise FileExistsError(f"refusing to overwrite existing output: {output_root}")
        print(f"[force] removing prior marked output {output_root}")
        shutil.rmtree(output_root)
    output_root.parent.mkdir(parents=True, exist_ok=True)
    staging = output_root.parent / ("." + output_root.name + ".staging")
    if staging.exists():
        raise FileExistsError(f"staging path already exists: {staging}")
    try:
        print(f"[copy] hard-linking audited v2.1 source to {staging}")
        copy_tree_hardlink(source_root, staging)
        info = read_json(staging / "meta" / "info.json")
        if info.get("total_episodes") != EXPECTED_EPISODES or info.get("total_frames") != EXPECTED_FRAMES:
            raise ValueError("source LeRobot info totals do not match active set")
        if info.get("fps") != 30 or info.get("features", {}).get("action", {}).get("shape") != [38]:
            raise ValueError("source LeRobot info is not the expected 38D/30Hz format")
        max_diff, rows = rewrite_actions_and_validate(staging, raw_root, source_manifest)
        repaired_videos = repair_video_timestamps(staging, source_manifest, args.video_workers)
        black_wrists = assert_missing_wrists_stay_black(staging, source_manifest)
        rewrite_episode_stats(staging, source_manifest)
        aggregate_standard_stats(staging)
        write_provenance(staging, raw_root, source_root, source_manifest, max_diff, rows, repaired_videos)
        os.replace(staging, output_root)
        make_lerobot_compat_view(output_root)
        print(f"[done] output={output_root}")
        print(f"[done] validated action rows={rows}, max_abs_diff={max_diff:.3g}, black wrist videos kept={black_wrists}")
    except Exception:
        if staging.exists():
            print(f"[error] leaving staging directory for inspection: {staging}", file=sys.stderr)
        raise
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
