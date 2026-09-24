#!/usr/bin/env bash
set -euo pipefail

if [[ $# -ne 1 ]]; then
  echo "Usage: $0 <task>" >&2
  exit 2
fi
TASK=$1
case "$TASK" in
  collect_objects|dual_bottles_pick|hammer_beat|insert_block|retrieve_gap|stack_bowls) ;;
  *) echo "Unsupported task: $TASK" >&2; exit 2 ;;
esac

WORKSPACE=/personal/xiangpc/0812_Xpolicylab_bench/FastWAM
DATASET_ID="spark0_mnt20260812_fastwam_${TASK}_v21_joint54"
DATASET="$WORKSPACE/data/$DATASET_ID"
CONVERSION_LOG="$WORKSPACE/logs/convert_mnt_fastwam_${TASK}_full_20260812.log"
NUMERIC_LOG="$WORKSPACE/logs/validate_mnt_fastwam_${TASK}_full_20260812.log"
VIDEO_LOG="$WORKSPACE/logs/validate_mnt_fastwam_${TASK}_video_full_20260812.log"

test ! -e "$DATASET"
export HF_LEROBOT_HOME="$WORKSPACE/data"
export PYTHONPATH="$WORKSPACE:$WORKSPACE/Xpolicylab:${PYTHONPATH:-}"

/personal/miniconda3/envs/dexora_1b/bin/python \
  "$WORKSPACE/data_scripts/hdf5_to_lerobot_v21_joint_parallel_tasks.py" \
  --source-root /mnt/xspark-data/tjy/spark0_bench \
  --output "$DATASET" \
  --repo-root "$WORKSPACE/Xpolicylab" \
  --episodes-per-task 100 \
  --image-writer-threads 8 \
  --video-batch-size 10 \
  --video-encoding-workers 16 \
  --tasks "$TASK" >"$CONVERSION_LOG" 2>&1

/personal/miniconda3/envs/dexora_1b/bin/python \
  "$WORKSPACE/data_scripts/validate_lerobot_joint54_tasks.py" \
  --source-root /mnt/xspark-data/tjy/spark0_bench \
  --dataset "$DATASET" \
  --episodes-per-task 100 \
  --tasks "$TASK" >"$NUMERIC_LOG" 2>&1

/personal/miniconda3/envs/dexora_1b/bin/python \
  "$WORKSPACE/data_scripts/validate_fastwam_video_rgb.py" \
  --dataset "$DATASET" >"$VIDEO_LOG" 2>&1

echo "FASTWAM_TASK_CONVERSION_VALIDATION_OK task=$TASK dataset=$DATASET"
