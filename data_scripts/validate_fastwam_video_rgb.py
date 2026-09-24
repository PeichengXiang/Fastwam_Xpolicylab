#!/usr/bin/env python3
"""Validate FastWAM H.264 video shape, length, and RGB channel ordering."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import av
import h5py
import numpy as np

from XPolicyLab.utils.process_data import decode_image_bit


CAMERAS = {
    "cam_high": "cam_head",
    "cam_left_wrist": "cam_left_wrist",
    "cam_right_wrist": "cam_right_wrist",
}


def decode_video_samples(path: Path, indices: list[int]) -> tuple[list[np.ndarray], int]:
    wanted = set(indices)
    samples: dict[int, np.ndarray] = {}
    count = 0
    with av.open(str(path)) as container:
        for index, frame in enumerate(container.decode(video=0)):
            if index in wanted:
                samples[index] = frame.to_ndarray(format="rgb24")
            count = index + 1
    if set(samples) != wanted:
        raise AssertionError(f"{path}: missing sampled frames {wanted - set(samples)}")
    return [samples[index] for index in indices], count


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", type=Path, required=True)
    parser.add_argument("--max-episodes", type=int, default=None)
    args = parser.parse_args()
    dataset = args.dataset.resolve(strict=True)
    manifest = json.loads((dataset / "meta/spark0_source_manifest.json").read_text())
    if args.max_episodes is not None:
        manifest = manifest[: args.max_episodes]

    rgb_squared_error = 0.0
    swapped_squared_error = 0.0
    value_count = 0
    checked_videos = 0
    for episode_index, episode in enumerate(manifest):
        length = int(episode["frames"])
        sample_indices = sorted({0, length // 2, length - 1})
        with h5py.File(episode["source"], "r") as source:
            for video_key, source_key in CAMERAS.items():
                video_path = (
                    dataset
                    / "videos"
                    / f"chunk-{episode_index // 1000:03d}"
                    / f"observation.images.{video_key}"
                    / f"episode_{episode_index:06d}.mp4"
                )
                decoded, frame_count = decode_video_samples(video_path, sample_indices)
                if frame_count != length:
                    raise AssertionError(
                        f"{video_path}: expected {length} frames, decoded {frame_count}"
                    )
                raw = np.asarray(
                    decode_image_bit(source[f"vision/{source_key}/colors"][sample_indices])
                )
                rendered = np.stack(decoded)
                if rendered.shape != raw.shape or rendered.dtype != np.uint8:
                    raise AssertionError(
                        f"{video_path}: decoded contract {rendered.shape}/{rendered.dtype}, "
                        f"source {raw.shape}/{raw.dtype}"
                    )
                rendered_f = rendered.astype(np.float32)
                raw_f = raw.astype(np.float32)
                rgb_squared_error += float(np.square(rendered_f - raw_f).sum())
                swapped_squared_error += float(np.square(rendered_f - raw_f[..., ::-1]).sum())
                value_count += int(raw.size)
                checked_videos += 1
        if (episode_index + 1) % 25 == 0:
            print(f"VIDEO_VALIDATED {episode_index + 1}/{len(manifest)}", flush=True)

    rgb_mse = rgb_squared_error / value_count
    swapped_mse = swapped_squared_error / value_count
    if not rgb_mse < swapped_mse:
        raise AssertionError(
            f"RGB ordering check failed: rgb_mse={rgb_mse}, swapped_mse={swapped_mse}"
        )
    print(
        f"FASTWAM_VIDEO_RGB_VALIDATION_OK episodes={len(manifest)} videos={checked_videos} "
        f"rgb_mse={rgb_mse:.6f} swapped_mse={swapped_mse:.6f}",
        flush=True,
    )


if __name__ == "__main__":
    main()
