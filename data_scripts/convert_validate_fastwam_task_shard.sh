#!/usr/bin/env bash
set -euo pipefail

if [[ $# -ne 2 ]]; then
  echo "Usage: $0 <task> <episode-start: 0 or 50>" >&2
  exit 2
fi
TASK=$1
EPISODE_START=$2
case "$TASK" in
  collect_objects|dual_bottles_pick|hammer_beat|insert_block|retrieve_gap|stack_bowls) ;;
  *) echo "Unsupported task: $TASK" >&2; exit 2 ;;
esac
case "$EPISODE_START" in
  0) EPISODE_STOP=49 ;;
  50) EPISODE_STOP=99 ;;
  *) echo "Episode start must be 0 or 50" >&2; exit 2 ;;
esac

WORKSPACE=/personal/xiangpc/0812_Xpolicylab_bench/FastWAM
SHARD=$(printf 'ep%03d_%03d' "$EPISODE_START" "$EPISODE_STOP")
DATASET_ID="spark0_mnt20260812_fastwam_${TASK}_${SHARD}_v21_joint54"
DATASET="$WORKSPACE/data/$DATASET_ID"
CONVERSION_LOG="$WORKSPACE/logs/convert_mnt_fastwam_${TASK}_${SHARD}_full_20260812.log"
NUMERIC_LOG="$WORKSPACE/logs/validate_mnt_fastwam_${TASK}_${SHARD}_full_20260812.log"
VIDEO_LOG="$WORKSPACE/logs/validate_mnt_fastwam_${TASK}_${SHARD}_video_full_20260812.log"

test ! -e "$DATASET"
export HF_LEROBOT_HOME="$WORKSPACE/data"
export PYTHONPATH="$WORKSPACE:$WORKSPACE/Xpolicylab:${PYTHONPATH:-}"

/personal/miniconda3/envs/dexora_1b/bin/python \
  "$WORKSPACE/data_scripts/hdf5_to_lerobot_v21_joint_shards.py" \
  --source-root /mnt/xspark-data/tjy/spark0_bench \
  --output "$DATASET" \
  --repo-root "$WORKSPACE/Xpolicylab" \
  --episodes-per-task 50 \
  --episode-start "$EPISODE_START" \
  --image-writer-threads 6 \
  --video-batch-size 10 \
  --video-encoding-workers 12 \
  --tasks "$TASK" >"$CONVERSION_LOG" 2>&1

/personal/miniconda3/envs/dexora_1b/bin/python \
  "$WORKSPACE/data_scripts/validate_lerobot_joint54_shards.py" \
  --source-root /mnt/xspark-data/tjy/spark0_bench \
  --dataset "$DATASET" \
  --episodes-per-task 50 \
  --episode-start "$EPISODE_START" \
  --tasks "$TASK" >"$NUMERIC_LOG" 2>&1

/personal/miniconda3/envs/dexora_1b/bin/python \
  "$WORKSPACE/data_scripts/validate_fastwam_video_rgb.py" \
  --dataset "$DATASET" >"$VIDEO_LOG" 2>&1

echo "FASTWAM_TASK_CONVERSION_VALIDATION_OK task=$TASK shard=$SHARD dataset=$DATASET"
