#!/usr/bin/env bash
set -euo pipefail

SOURCE_ROOT="${SPARK0_HDF5_ROOT:-/personal/tjy/spark0_bench}"
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
POLICY_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
TARGET="${POLICY_ROOT}/data/raw_hdf5"
TASKS=(collect_objects dual_bottles_pick hammer_beat insert_block retrieve_gap stack_bowls)

SOURCE_ROOT="$(realpath -e "${SOURCE_ROOT}")"
for task in "${TASKS[@]}"; do
  data_dir="${SOURCE_ROOT}/${task}/tianji_marvin_wuji/data"
  [[ -d "${data_dir}" ]] || { echo "Missing ${data_dir}" >&2; exit 1; }
  count="$(find "${data_dir}" -maxdepth 1 -type f -name 'episode_*.hdf5' | wc -l)"
  [[ "${count}" == 100 ]] || { echo "${task}: expected 100 HDF5 episodes, got ${count}" >&2; exit 1; }
done

mkdir -p "${POLICY_ROOT}/data"
ln -sfn "${SOURCE_ROOT}" "${TARGET}"
echo "raw_hdf5=${TARGET} -> $(readlink -f "${TARGET}")"
echo "validated_tasks=6 validated_episodes=600"
