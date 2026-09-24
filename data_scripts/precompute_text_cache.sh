#!/usr/bin/env bash
set -euo pipefail

WORKSPACE=${WORKSPACE:-/personal/xiangpc/0812_Xpolicylab_bench/FastWAM}
FW="$WORKSPACE/Xpolicylab/policy/FastWAM/FastWAM"
FASTWAM_ENV=${FASTWAM_ENV:-/personal/miniconda3/envs/fastwam}
DATASET_ID=Spark0_bench-cotrain-tianji_marvin_wuji-joint
DATASET="$WORKSPACE/data/$DATASET_ID/lerobot"
CACHE="$FW/data/text_embeds_cache/xpolicylab/$DATASET_ID"

cd "$FW"
export CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-0}
export PATH="$FASTWAM_ENV/bin:$PATH"
export DIFFSYNTH_MODEL_BASE_PATH="$WORKSPACE/pretrain_model"
export DIFFSYNTH_SKIP_DOWNLOAD=true
export PYTHONPATH="$FW:$FW/src:$WORKSPACE:${PYTHONPATH:-}"

"$FASTWAM_ENV/bin/python" scripts/precompute_text_embeds.py \
    task=robotwin_uncond_3cam_384_1e-4 \
    "data.train.dataset_dirs=[$DATASET]" \
    "data.val.dataset_dirs=[$DATASET]" \
    "data.train.text_embedding_cache_dir=$CACHE" \
    "data.val.text_embedding_cache_dir=$CACHE" \
    +overwrite=false
