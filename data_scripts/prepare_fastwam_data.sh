#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
POLICY_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
SOURCE_ROOT="${SPARK0_HDF5_ROOT:-/personal/tjy/spark0_bench}"
DATASET_ID="${FASTWAM_DATASET_ID:-Spark0_bench-cotrain-tianji_marvin_wuji-joint}"
KNOWN_DATASET="${FASTWAM_KNOWN_LEROBOT_V21:-/personal/xiangpc/0811_Xpolicylab_bench/GrootN17/data/Spark0_bench-groot_n17_lerobotV21_joint54_h264-tianji_marvin_wuji-joint}"
DEST_ROOT="${POLICY_ROOT}/data/${DATASET_ID}"

bash "${SCRIPT_DIR}/link_raw_hdf5.sh"
[[ -f "${KNOWN_DATASET}/meta/info.json" ]] || { echo "Missing known LeRobot dataset: ${KNOWN_DATASET}" >&2; exit 1; }
python3 - "${KNOWN_DATASET}/meta/info.json" <<'PY'
import json, sys
info = json.load(open(sys.argv[1], encoding="utf-8"))
assert info["codebase_version"] == "v2.1"
assert info["total_episodes"] == 600
assert info["total_tasks"] == 6
assert info["fps"] == 25
assert info["features"]["observation.state"]["shape"] == [54]
assert info["features"]["action"]["shape"] == [54]
expected = {
    "observation.images.cam_high",
    "observation.images.cam_left_wrist",
    "observation.images.cam_right_wrist",
}
actual = {k for k, v in info["features"].items() if v.get("dtype") == "video"}
assert actual == expected, actual
print("validated LeRobot v2.1: 600 episodes, 6 tasks, joint54, 3 RGB cameras")
PY

mkdir -p "${DEST_ROOT}"
ln -sfn "$(realpath -e "${KNOWN_DATASET}")" "${DEST_ROOT}/lerobot"
echo "lerobot=${DEST_ROOT}/lerobot -> $(readlink -f "${DEST_ROOT}/lerobot")"
echo "The FastWAM-specific dataset_stats.json must be generated with its processor before training."
