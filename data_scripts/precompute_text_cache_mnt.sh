#!/usr/bin/env bash
set -euo pipefail

WORKSPACE=${WORKSPACE:-/personal/xiangpc/0812_Xpolicylab_bench/FastWAM}
FW="$WORKSPACE/Xpolicylab/policy/FastWAM/FastWAM"
FASTWAM_ENV=${FASTWAM_ENV:-/personal/miniconda3/envs/fastwam}
DATASET_ID=spark0_mnt20260812_fastwam_6tasks_v21_joint54
TASKS=(collect_objects dual_bottles_pick hammer_beat insert_block retrieve_gap stack_bowls)
DATASETS=()
for task_name in "${TASKS[@]}"; do
    for shard in ep000_049 ep050_099; do
        DATASETS+=("$WORKSPACE/data/spark0_mnt20260812_fastwam_${task_name}_${shard}_v21_joint54")
    done
done
DATASET_LIST="[$(IFS=,; echo "${DATASETS[*]}")]"
CACHE="$FW/data/text_embeds_cache/xpolicylab/$DATASET_ID"

for dataset in "${DATASETS[@]}"; do
    test -d "$dataset/meta"
done
cd "$FW"
export CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-0}
export PATH="$FASTWAM_ENV/bin:$PATH"
export DIFFSYNTH_MODEL_BASE_PATH="$WORKSPACE/pretrain_model"
export DIFFSYNTH_SKIP_DOWNLOAD=true
export PYTHONPATH="$FW:$FW/src:$WORKSPACE:${PYTHONPATH:-}"

"$FASTWAM_ENV/bin/python" scripts/precompute_text_embeds.py \
    task=robotwin_uncond_3cam_384_1e-4 \
    "data.train.dataset_dirs=$DATASET_LIST" \
    "data.val.dataset_dirs=$DATASET_LIST" \
    "data.train.text_embedding_cache_dir=$CACHE" \
    "data.val.text_embedding_cache_dir=$CACHE" \
    +overwrite=false
