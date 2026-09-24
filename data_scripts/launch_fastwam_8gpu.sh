#!/usr/bin/env bash
set -euo pipefail

# Official FastWAM joint-mode training with config-only benchmark overrides.
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
STATS="$WORKSPACE/data/$DATASET_ID/dataset_stats.json"
CACHE="$FW/data/text_embeds_cache/xpolicylab/$DATASET_ID"
OUTPUT="$WORKSPACE/chpt/20260812_mnt_joint_bs64_s42_80k"
ACTION_DIT="$WORKSPACE/pretrain_model/ActionDiT_linear_interp_Wan22_alphascale_1024hdim.pt"

for dataset in "${DATASETS[@]}"; do
    test -d "$dataset/meta"
done
test -f "$STATS"
test -f "$ACTION_DIT"
test -n "$(find "$CACHE" -name '*.pt' -print -quit 2>/dev/null)"
mkdir -p "$OUTPUT" "$WORKSPACE/logs"

RESUME_STATE=${FASTWAM_RESUME_STATE:-}
if [[ -z "$RESUME_STATE" && -d "$OUTPUT/checkpoints/state" ]]; then
    RESUME_STATE=$(find "$OUTPUT/checkpoints/state" -mindepth 1 -maxdepth 1 -type d -name 'step_*' | sort -V | tail -n 1)
fi
RESUME_OVERRIDE=()
if [[ -n "$RESUME_STATE" ]]; then
    RESUME_OVERRIDE=("resume=$RESUME_STATE")
fi

cd "$FW"
export CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7
export PATH="$FASTWAM_ENV/bin:$PATH"
export DIFFSYNTH_MODEL_BASE_PATH="$WORKSPACE/pretrain_model"
export DIFFSYNTH_SKIP_DOWNLOAD=true
export PYTHONPATH="$FW:$FW/src:$WORKSPACE:${PYTHONPATH:-}"
export PYTORCH_CUDA_ALLOC_CONF=${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}
export WANDB_MODE=online
export RUN_ID=20260812_mnt_joint_bs64_s42_80k
export MASTER_PORT=${MASTER_PORT:-42489}

bash scripts/train_zero1.sh 8 \
    task=robotwin_uncond_3cam_384_1e-4 \
    seed=42 \
    batch_size=8 \
    gradient_accumulation_steps=1 \
    num_workers=8 \
    num_epochs=100000 \
    max_steps=80000 \
    save_every=10000 \
    wandb.enabled=true \
    wandb.mode=online \
    wandb.workspace=peichengxiang773-hkust \
    wandb.project=fast-wam \
    wandb.name=20260812_mnt_joint_bs64_s42_80k_resume25k \
    "${RESUME_OVERRIDE[@]}" \
    "data.train.dataset_dirs=$DATASET_LIST" \
    "data.val.dataset_dirs=$DATASET_LIST" \
    "data.train.text_embedding_cache_dir=$CACHE" \
    "data.val.text_embedding_cache_dir=$CACHE" \
    "data.train.pretrained_norm_stats=$STATS" \
    "data.val.pretrained_norm_stats=$STATS" \
    data.train.shape_meta.action.0.raw_shape=54 \
    data.train.shape_meta.action.0.shape=54 \
    data.train.shape_meta.state.0.raw_shape=54 \
    data.train.shape_meta.state.0.shape=54 \
    data.val.shape_meta.action.0.raw_shape=54 \
    data.val.shape_meta.action.0.shape=54 \
    data.val.shape_meta.state.0.raw_shape=54 \
    data.val.shape_meta.state.0.shape=54 \
    data.train.processor.action_output_dim=54 \
    data.train.processor.proprio_output_dim=54 \
    data.val.processor.action_output_dim=54 \
    data.val.processor.proprio_output_dim=54 \
    "output_dir=$OUTPUT"
