#!/usr/bin/env bash
set -euo pipefail

# Fresh 8-GPU FastWAM run for the active EgoVLA benchmark.  The W&B token is
# deliberately not stored in this file; export WANDB_API_KEY in the launch
# environment (or pass it through a protected scheduler secret).
WORKSPACE=${WORKSPACE:-/personal/xiangpc/0812_Xpolicylab_bench/FastWAM}
FW="$WORKSPACE/Xpolicylab/policy/FastWAM/FastWAM"
FASTWAM_ENV=${FASTWAM_ENV:-/personal/miniconda3/envs/fastwam}
DATASET_ID=${DATASET_ID:-EgoVLA_benchmark_fastwam_v21_joint38_cmd}
DATASET_ROOT="$WORKSPACE/data/$DATASET_ID"
DATASET="$DATASET_ROOT/lerobot"
STATS="$DATASET_ROOT/dataset_stats.json"
CACHE="$FW/data/text_embeds_cache/xpolicylab/$DATASET_ID"
RUN_ID=${RUN_ID:-egovla_joint38_taskuniform_bs64_s42_80k}
OUTPUT=${OUTPUT:-$WORKSPACE/chpt/$RUN_ID}
NUM_GPUS=${NUM_GPUS:-8}
SAMPLING_MODE=${SAMPLING_MODE:-task_uniform}
case "$SAMPLING_MODE" in
  frame_uniform|task_uniform) ;;
  *) echo "[error] SAMPLING_MODE must be frame_uniform or task_uniform" >&2; exit 2 ;;
esac
NUM_WORKERS=${FASTWAM_NUM_WORKERS:-8}
MASTER_PORT=${MASTER_PORT:-42539}
WANDB_WORKSPACE=${WANDB_WORKSPACE:-peichengxiang773-hkust}
WANDB_PROJECT=${WANDB_PROJECT:-fast-wam}

test -d "$DATASET/meta"
test -f "$STATS"
test -f "$WORKSPACE/pretrain_model/ActionDiT_linear_interp_Wan22_alphascale_1024hdim.pt"
test -n "$(find "$CACHE" -maxdepth 1 -name '*.pt' -print -quit 2>/dev/null)"
if [[ -e "$OUTPUT" ]]; then
  echo "[error] output already exists; refusing accidental resume: $OUTPUT" >&2
  echo "        choose another RUN_ID/OUTPUT or explicitly resume from a state directory." >&2
  exit 2
fi
if [[ -z "${WANDB_API_KEY:-}" ]]; then
  echo "[error] WANDB_API_KEY is required for online W&B logging; it is not embedded in this script." >&2
  exit 3
fi

mkdir -p "$WORKSPACE/logs"
cd "$FW"
export CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-0,1,2,3,4,5,6,7}
export PATH="$FASTWAM_ENV/bin:$PATH"
export DIFFSYNTH_MODEL_BASE_PATH="$WORKSPACE/pretrain_model"
export DIFFSYNTH_SKIP_DOWNLOAD=true
export PYTHONPATH="$FW:$FW/src:$WORKSPACE:${PYTHONPATH:-}"
export PYTORCH_CUDA_ALLOC_CONF=${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}
export RUN_ID
export MASTER_PORT

# Per-GPU batch 8 × 8 GPUs = global batch 64; fresh ActionDiT backbone with
# 38-D input/output heads; no 54-D checkpoint is reused.
bash scripts/train_zero1.sh "$NUM_GPUS" \
  task=egovla_joint38_3cam_384_1e-4 \
  seed=42 \
  "train_sampling=$SAMPLING_MODE" \
  batch_size=8 \
  gradient_accumulation_steps=1 \
  num_workers="$NUM_WORKERS" \
  num_epochs=100000 \
  max_steps=80000 \
  save_every=10000 \
  eval_every=10000 \
  mixed_precision=bf16 \
  wandb.enabled=true \
  wandb.mode=online \
  "wandb.workspace=$WANDB_WORKSPACE" \
  "wandb.project=$WANDB_PROJECT" \
  "wandb.name=$RUN_ID" \
  "data.train.dataset_dirs=[$DATASET]" \
  "data.val.dataset_dirs=[$DATASET]" \
  "data.train.pretrained_norm_stats=$STATS" \
  "data.val.pretrained_norm_stats=$STATS" \
  "data.train.text_embedding_cache_dir=$CACHE" \
  "data.val.text_embedding_cache_dir=$CACHE" \
  "model.action_dit_pretrained_path=$WORKSPACE/pretrain_model/ActionDiT_linear_interp_Wan22_alphascale_1024hdim.pt" \
  "output_dir=$OUTPUT"
