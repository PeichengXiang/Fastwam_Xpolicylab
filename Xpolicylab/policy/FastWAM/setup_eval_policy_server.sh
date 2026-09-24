#!/bin/bash
set -euo pipefail

bench_name=$1
task_name=$2
ckpt_name=$3
env_cfg_type=$4
action_type=$5
seed=$6
policy_gpu_id=$7
policy_conda_env=$8
policy_server_port=$9
policy_server_host=${10:-localhost}

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
XPL_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"
BENCH_ROOT="$(cd "${XPL_ROOT}/.." && pwd)"
UTILS_DIR="${XPL_ROOT}/utils"

policy_name="$(basename "${SCRIPT_DIR}")"
POLICY_DIR="${XPL_ROOT}/policy/${policy_name}"
FASTWAM_DIR="${POLICY_DIR}/FastWAM"
yaml_file="${POLICY_DIR}/deploy.yml"

if [[ "${action_type,,}" != "joint" ]]; then
    echo "[ERROR] FastWAM supports action_type=joint only (got ${action_type})" >&2
    exit 2
fi

allow_dummy_policy="${FASTWAM_ALLOW_DUMMY_POLICY:-false}"
# The policy server can be launched by a minimal web-runner environment where
# neither `conda` nor its shell function is on PATH.  Resolve the executable
# explicitly, then source that installation's shell hook before activation.
conda_exe="${CONDA_EXE:-}"
if [[ ! -x "${conda_exe}" ]]; then
    conda_exe="$(type -P conda 2>/dev/null || true)"
fi
if [[ ! -x "${conda_exe}" ]]; then
    for candidate in /personal/miniconda3/bin/conda /data/miniconda3/bin/conda; do
        if [[ -x "${candidate}" ]]; then
            conda_exe="${candidate}"
            break
        fi
    done
fi

if [[ -x "${conda_exe}" ]]; then
    conda_root="$(cd "$(dirname "${conda_exe}")/.." && pwd)"
    if [[ -f "${conda_root}/etc/profile.d/conda.sh" ]]; then
        # shellcheck disable=SC1091
        source "${conda_root}/etc/profile.d/conda.sh"
    fi
    conda activate "${policy_conda_env}"
elif [[ -x "${policy_conda_env%/}/bin/python" ]]; then
    # A prefix environment is usable without Conda's shell integration.
    export PATH="${policy_conda_env%/}/bin:${PATH:-}"
else
    echo "[ERROR] Cannot locate Conda or policy environment Python: ${policy_conda_env}" >&2
    exit 1
fi

action_dim=$(bash "${UTILS_DIR}/get_action_dim.sh" "${BENCH_ROOT}" "${env_cfg_type}")
echo -e "\033[33m[SERVER] action_dim=${action_dim}\033[0m"

# Resolve the run directory through XPolicyLab's shared resolver.  FastWAM's
# training output is outside policy/FastWAM/checkpoints (under the model-root
# `chpt/` tree), so search both roots while retaining the standard precedence
# for explicit paths and the concatenated run name.
ckpt_setting="${FASTWAM_CKPT_SETTING:-${ckpt_name}}"
explicit_checkpoint="${FASTWAM_CHECKPOINT_PATH:-}"
resolve_with_shared_resolver() {
    local checkpoints_root="$1"
    BENCH_NAME="${bench_name}" \
    CKPT_NAME="${ckpt_setting}" \
    ENV_CFG_TYPE="${env_cfg_type}" \
    ACTION_TYPE="${action_type}" \
    SEED="${seed}" \
    POLICY_DIR="${POLICY_DIR}" \
    EXPLICIT_CHECKPOINT="${explicit_checkpoint}" \
    PYTHONPATH="${BENCH_ROOT}:${FASTWAM_DIR}:${FASTWAM_DIR}/src:${PYTHONPATH:-}" \
    python - "${checkpoints_root}" <<'PY'
import os
import sys

from XPolicyLab.utils.checkpoint_resolver import resolve_checkpoint_root

cfg = {
    "bench_name": os.environ["BENCH_NAME"],
    "ckpt_name": os.environ["CKPT_NAME"],
    "env_cfg_type": os.environ["ENV_CFG_TYPE"],
    "action_type": os.environ["ACTION_TYPE"],
    "seed": os.environ["SEED"],
}
explicit = os.environ.get("EXPLICIT_CHECKPOINT", "").strip()
if explicit:
    cfg["checkpoint_path"] = explicit
try:
    result = resolve_checkpoint_root(
        cfg,
        sys.argv[1],
        policy_dir=os.environ["POLICY_DIR"],
        must_exist=True,
    )
except FileNotFoundError:
    raise SystemExit(1)
print(result)
PY
}

declare -a checkpoint_roots=()
if [[ -n "${FASTWAM_CKPT_ROOT:-}" ]]; then
    checkpoint_roots+=("${FASTWAM_CKPT_ROOT}")
fi
checkpoint_roots+=("${BENCH_ROOT}/chpt" "${POLICY_DIR}/checkpoints")
ckpt_dir=""
for checkpoints_root in "${checkpoint_roots[@]}"; do
    resolved_ckpt="$(resolve_with_shared_resolver "${checkpoints_root}" 2>/dev/null || true)"
    if [[ -n "${resolved_ckpt}" && -e "${resolved_ckpt}" ]]; then
        ckpt_dir="${resolved_ckpt}"
        break
    fi
done

if [[ -z "${ckpt_dir}" ]]; then
    if [[ "${allow_dummy_policy}" == "true" ]]; then
        ckpt_dir="${BENCH_ROOT}/chpt/${ckpt_setting}"
    else
        echo "[ERROR] FastWAM checkpoint could not be resolved for ckpt_name=${ckpt_name}" >&2
        echo "        pass FASTWAM_CHECKPOINT_PATH or a valid run directory" >&2
        exit 2
    fi
fi

checkpoint_path="${explicit_checkpoint}"
weights_dir=""
if [[ -z "${checkpoint_path}" ]]; then
    if [[ -f "${ckpt_dir}" ]]; then
        checkpoint_path="${ckpt_dir}"
    elif [[ -d "${ckpt_dir}/checkpoints/weights" ]]; then
        weights_dir="${ckpt_dir}/checkpoints/weights"
    elif [[ -d "${ckpt_dir}/weights" ]]; then
        weights_dir="${ckpt_dir}/weights"
    elif [[ -d "${ckpt_dir}" ]]; then
        weights_dir="${ckpt_dir}"
    fi
    if [[ -n "${weights_dir}" ]]; then
        checkpoint_path="$(find "${weights_dir}" -maxdepth 1 -type f -name 'step_*.pt' | sort -V | tail -n 1)"
    fi
fi
if [[ -n "${checkpoint_path}" && -d "${checkpoint_path}" ]]; then
    indexed_checkpoint="$(find "${checkpoint_path}" -maxdepth 1 -type f -name 'step_*.pt' | sort -V | tail -n 1)"
    if [[ -n "${indexed_checkpoint}" ]]; then
        checkpoint_path="${indexed_checkpoint}"
    fi
fi
if [[ "${allow_dummy_policy}" != "true" && ( -z "${checkpoint_path}" || ! -f "${checkpoint_path}" ) ]]; then
    echo "[ERROR] No FastWAM weights found below ${ckpt_dir}" >&2
    exit 2
fi

# The FastWAM stats file is produced beside the converted EgoVLA dataset, not
# inside the DeepSpeed run directory.  An explicit path wins; the standard
# dataset location is a safe fallback for this campaign.
dataset_stats_path="${FASTWAM_DATASET_STATS_PATH:-${EGOVLA_DATASET_STATS_PATH:-}}"
if [[ -z "${dataset_stats_path}" && -f "${ckpt_dir}/dataset_stats.json" ]]; then
    dataset_stats_path="${ckpt_dir}/dataset_stats.json"
fi
dataset_id="${FASTWAM_DATASET_ID:-EgoVLA_benchmark_fastwam_v21_joint38_cmd}"
if [[ -z "${dataset_stats_path}" ]]; then
    for candidate in \
        "${BENCH_ROOT}/data/${dataset_id}/dataset_stats.json" \
        "${BENCH_ROOT}/data/EgoVLA_benchmark_fastwam_v21_joint38_cmd/dataset_stats.json"; do
        if [[ -f "${candidate}" ]]; then
            dataset_stats_path="${candidate}"
            break
        fi
    done
fi
if [[ -z "${dataset_stats_path}" ]]; then
    dataset_stats_path="${ckpt_dir}/dataset_stats.json"
fi
seven_task_stats="${BENCH_ROOT}/data/spark0_bench_7tasks_0908/dataset_stats_0908_7task.json"
if [[ "${ckpt_dir}" == *0908_7task* || "${dataset_stats_path}" == *0908_7task* ]]; then
    if [[ -f "${seven_task_stats}" ]]; then
        dataset_stats_path="${seven_task_stats}"
    fi
fi
if [[ "${allow_dummy_policy}" != "true" && ! -f "${dataset_stats_path}" ]]; then
    echo "[ERROR] FastWAM dataset stats not found: ${dataset_stats_path}" >&2
    echo "        set FASTWAM_DATASET_STATS_PATH to the training dataset_stats.json" >&2
    exit 2
fi

model_base_path="${FASTWAM_MODEL_BASE_PATH:-${model_base_path:-}}"
if [[ -z "${model_base_path}" ]]; then
    for candidate in "${BENCH_ROOT}/pretrain_model" "${FASTWAM_DIR}/checkpoints"; do
        if [[ -d "${candidate}" ]]; then
            model_base_path="${candidate}"
            break
        fi
    done
fi

echo -e "\033[33m[SERVER] policy=${policy_name}, task=${task_name}, ckpt=${ckpt_name}\033[0m"
echo -e "\033[33m[SERVER] checkpoint_path: ${checkpoint_path:-<dummy-policy>}\033[0m"
echo -e "\033[33m[SERVER] dataset_stats_path: ${dataset_stats_path}\033[0m"
echo -e "\033[33m[SERVER] model_base_path: ${model_base_path:-<unset>}\033[0m"
echo -e "\033[33m[SERVER] policy_server_host=${policy_server_host} policy_server_port=${policy_server_port}\033[0m"

# Scope env vars to this server process only; never export them in the
# orchestrator, otherwise they leak into the env client.
exec env \
    PYTHONWARNINGS=ignore::UserWarning \
    PYTHONUNBUFFERED=1 \
    CUDA_VISIBLE_DEVICES="${policy_gpu_id}" \
    PYTHONPATH="${BENCH_ROOT}:${FASTWAM_DIR}:${FASTWAM_DIR}/src:${PYTHONPATH:-}" \
    DIFFSYNTH_MODEL_BASE_PATH="${model_base_path:-${FASTWAM_DIR}/checkpoints}" \
    DIFFSYNTH_SKIP_DOWNLOAD="${DIFFSYNTH_SKIP_DOWNLOAD:-true}" \
    python -u "${XPL_ROOT}/setup_policy_server.py" \
        --config_path "${yaml_file}" \
        --overrides \
            port="${policy_server_port}" \
            host="${policy_server_host}" \
            bench_name="${bench_name}" \
            task_name="${task_name}" \
            ckpt_name="${ckpt_name}" \
            env_cfg_type="${env_cfg_type}" \
            seed="${seed}" \
            policy_name="${policy_name}" \
            action_type="${action_type}" \
            action_dim="${action_dim}" \
            checkpoint_path="${checkpoint_path}" \
            dataset_stats_path="${dataset_stats_path}" \
            checkpoint_root="${FASTWAM_CKPT_ROOT:-}" \
            checkpoint_num="${FASTWAM_CHECKPOINT_NUM:-}" \
            allow_dummy_policy="${allow_dummy_policy}" \
            model_base_path="${model_base_path}"
