#!/usr/bin/env python3
"""Download exactly the official FastWAM Wan components into pretrain_model."""

from fastwam.models.wan22.helpers.loader import _resolve_configs


def main() -> None:
    configs = _resolve_configs(
        model_id="Wan-AI/Wan2.2-TI2V-5B",
        tokenizer_model_id="Wan-AI/Wan2.1-T2V-1.3B",
        redirect_common_files=True,
    )
    labels = ("video_dit", "text_encoder", "vae", "tokenizer")
    for label, config in zip(labels, configs):
        print(f"DOWNLOAD_START {label} {config.model_id} {config.origin_file_pattern}", flush=True)
        config.download_if_necessary()
        print(f"DOWNLOAD_OK {label} {config.path}", flush=True)


if __name__ == "__main__":
    main()
