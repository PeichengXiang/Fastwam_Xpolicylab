#!/usr/bin/env python3
"""Validate EgoVLA FastWAM video RGB order, alignment, PTS, and black wrists."""
from __future__ import annotations
import argparse, json
from pathlib import Path
import av
import h5py
import numpy as np

CAMERAS = {
    "cam_high": "main",
    "cam_left_wrist": "left_hand",
    "cam_right_wrist": "right_hand",
}
IMAGE_SHAPE = (384, 384, 3)

def decode_samples(path: Path, indices: list[int]):
    wanted = set(indices)
    samples = {}
    count = 0
    first_pts = None
    with av.open(str(path)) as container:
        stream = container.streams.video[0]
        for i, frame in enumerate(container.decode(video=0)):
            if i == 0:
                first_pts = float(frame.pts * stream.time_base) if frame.pts is not None else None
            if i in wanted:
                samples[i] = frame.to_ndarray(format="rgb24")
            count = i + 1
    if set(samples) != wanted:
        raise AssertionError(f"{path}: missing samples {sorted(wanted - set(samples))}")
    return [samples[i] for i in indices], count, first_pts

def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", type=Path, required=True)
    ap.add_argument("--max-episodes", type=int, default=None)
    args = ap.parse_args()
    if args.max_episodes is not None and args.max_episodes <= 0:
        raise ValueError("--max-episodes must be positive")
    dataset = args.dataset.resolve(strict=True)
    if (dataset / "lerobot" / "meta").is_dir() and not (dataset / "meta").is_dir():
        dataset = dataset / "lerobot"
    meta = dataset / "meta"
    conversion = json.loads((meta / "fastwam_conversion_manifest.json").read_text())
    source_manifest = json.loads((meta / "xpolicylab_source_conversion.json").read_text())
    raw_root = Path(source_manifest["source_root"]).resolve(strict=True)
    entries = sorted(source_manifest["episodes"], key=lambda e: int(e["dataset_episode_index"]))
    if args.max_episodes is not None:
        entries = entries[: args.max_episodes]
    if not entries:
        raise ValueError("no episodes selected")
    rgb_sse = swapped_sse = pixels = 0
    wrist_max = wrist_mean = 0.0
    checked_videos = 0
    for ordinal, entry in enumerate(entries, 1):
        ep = int(entry["dataset_episode_index"])
        length = int(entry["frames"])
        raw_path = raw_root / entry["source_relative_path"]
        if not raw_path.is_file():
            raise FileNotFoundError(raw_path)
        sample_indices = sorted({0, length // 2, length - 1})
        with h5py.File(raw_path, "r") as h5:
            for cam, raw_name in CAMERAS.items():
                video = dataset / "videos" / f"chunk-{ep // 1000:03d}" / f"observation.images.{cam}" / f"episode_{ep:06d}.mp4"
                rendered, frame_count, first_pts = decode_samples(video, sample_indices)
                if frame_count != length:
                    raise AssertionError(f"{video}: expected {length} frames, got {frame_count}")
                if first_pts is None or abs(first_pts) > 1e-4:
                    raise AssertionError(f"{video}: first PTS={first_pts}, expected 0")
                has_raw = f"observations/images/{raw_name}" in h5
                expected_real = bool(entry.get("has_real_wrist_cameras", False))
                if cam != "cam_high" and has_raw != expected_real:
                    raise AssertionError(f"{raw_path}: manifest wrist={expected_real}, raw {raw_name} present={has_raw}")
                if has_raw:
                    raw = np.asarray(h5[f"observations/images/{raw_name}"][sample_indices], dtype=np.uint8)
                else:
                    raw = np.zeros((len(sample_indices), *IMAGE_SHAPE), dtype=np.uint8)
                rendered = np.stack(rendered)
                if rendered.shape != raw.shape or rendered.dtype != np.uint8:
                    raise AssertionError(f"{video}: decoded {rendered.shape}/{rendered.dtype}, raw {raw.shape}/{raw.dtype}")
                if not has_raw:
                    # yuv420p black remains black after decoding; allow a tiny codec rounding margin.
                    max_abs = float(np.max(rendered.astype(np.int16)))
                    mean_abs = float(np.mean(rendered.astype(np.float32)))
                    wrist_max = max(wrist_max, max_abs)
                    wrist_mean = max(wrist_mean, mean_abs)
                    if max_abs > 3.0 or mean_abs > 0.5:
                        raise AssertionError(f"{video}: missing wrist is not black (max={max_abs}, mean={mean_abs})")
                else:
                    rf = rendered.astype(np.float32)
                    xf = raw.astype(np.float32)
                    rgb_sse += float(np.square(rf - xf).sum())
                    swapped_sse += float(np.square(rf - xf[..., ::-1]).sum())
                    pixels += int(raw.size)
                checked_videos += 1
        if ordinal % 25 == 0 or ordinal == len(entries):
            print(f"VIDEO_RGB_VALIDATED {ordinal}/{len(entries)}", flush=True)
    rgb_mse = rgb_sse / pixels if pixels else 0.0
    swapped_mse = swapped_sse / pixels if pixels else 0.0
    if pixels and not rgb_mse < swapped_mse:
        raise AssertionError(f"RGB order failed: rgb_mse={rgb_mse}, swapped_mse={swapped_mse}")
    # Missing wrists must be independent streams, never hard links to the head.
    checked_missing = 0
    for entry in entries:
        if entry.get("has_real_wrist_cameras", False):
            continue
        ep = int(entry["dataset_episode_index"])
        high = dataset / "videos" / f"chunk-{ep // 1000:03d}" / "observation.images.cam_high" / f"episode_{ep:06d}.mp4"
        for cam in ("cam_left_wrist", "cam_right_wrist"):
            wrist = dataset / "videos" / f"chunk-{ep // 1000:03d}" / f"observation.images.{cam}" / f"episode_{ep:06d}.mp4"
            if high.samefile(wrist):
                raise AssertionError(f"{wrist}: wrist stream aliases head stream")
            checked_missing += 1
        left = dataset / "videos" / f"chunk-{ep // 1000:03d}" / "observation.images.cam_left_wrist" / f"episode_{ep:06d}.mp4"
        right = dataset / "videos" / f"chunk-{ep // 1000:03d}" / "observation.images.cam_right_wrist" / f"episode_{ep:06d}.mp4"
        if left.samefile(right):
            raise AssertionError(f"{left}: left/right wrist streams alias each other")
    print(json.dumps({
        "status": "ok", "episodes": len(entries), "videos": checked_videos,
        "rgb_mse": rgb_mse, "channel_swapped_mse": swapped_mse,
        "missing_wrist_streams": checked_missing,
        "missing_wrist_max_decoded_value": wrist_max,
        "missing_wrist_mean_decoded_value": wrist_mean,
        "action_source": conversion.get("action_source"),
        "video_timestamp_policy": conversion.get("video_timestamp_policy"),
    }, ensure_ascii=False, indent=2), flush=True)
    return 0

if __name__ == "__main__":
    raise SystemExit(main())
