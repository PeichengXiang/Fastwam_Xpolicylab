#!/usr/bin/env bash
set -euo pipefail

WORKSPACE=/personal/xiangpc/0812_Xpolicylab_bench/FastWAM
FW="$WORKSPACE/Xpolicylab/policy/FastWAM/FastWAM"
TASKS=(collect_objects dual_bottles_pick hammer_beat insert_block retrieve_gap stack_bowls)
SHARDS=(ep000_049 ep050_099)
DATASETS=()

while true; do
  complete=0
  for task in "${TASKS[@]}"; do
    for shard in "${SHARDS[@]}"; do
      start=${shard:2:3}
      pipeline_log="$WORKSPACE/logs/pipeline_mnt_fastwam_${task}_ep${start}_full_20260812.log"
      if grep -q '^FASTWAM_TASK_CONVERSION_VALIDATION_OK ' "$pipeline_log" 2>/dev/null; then
        complete=$((complete + 1))
      fi
    done
  done
  echo "FASTWAM_SHARDS_VALIDATED $complete/12 $(date --iso-8601=seconds)"
  [[ $complete -eq 12 ]] && break
  sleep 30
done

for task in "${TASKS[@]}"; do
  for shard in "${SHARDS[@]}"; do
    DATASETS+=("$WORKSPACE/data/spark0_mnt20260812_fastwam_${task}_${shard}_v21_joint54")
  done
done

STATS_DIR="$WORKSPACE/data/spark0_mnt20260812_fastwam_6tasks_v21_joint54"
STATS="$STATS_DIR/dataset_stats.json"
mkdir -p "$STATS_DIR"
test ! -e "$STATS"

args=()
for dataset in "${DATASETS[@]}"; do
  args+=(--dataset "$dataset")
done
export PYTHONPATH="$FW:$FW/src:$WORKSPACE:${PYTHONPATH:-}"
/personal/miniconda3/envs/fastwam/bin/python \
  "$WORKSPACE/data_scripts/calc_fastwam_stats_multi.py" \
  "${args[@]}" --output "$STATS"

echo "FASTWAM_SHARDED_DATA_READY datasets=12 episodes=600 stats=$STATS"
