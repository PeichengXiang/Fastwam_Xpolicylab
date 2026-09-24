#!/bin/bash
set -euo pipefail

bench_name=$1
task_name=$2
ckpt_name=$3
env_cfg_type=$4
action_type=$5
seed=$6
env_gpu_id=$7
eval_env_conda_env=$8
additional_info=$9
policy_server_port=${10}
policy_server_ip=${11:-localhost}

# Keep non-interactive Web runs diagnostically useful: preserve Python fatal
# traces and a per-run client phase/return-code file.  The trace is additive
# and does not alter the evaluator command or GPU selection.
export PYTHONFAULTHANDLER=1
export PYTHONUNBUFFERED=1
export FASTWAM_CLIENT_TRACE_LOG="${FASTWAM_CLIENT_TRACE_LOG:-/tmp/egovla-fastwam-client-${LUMINIS_RUN_ID:-manual}-${BASHPID}.trace}"

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
XPL_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"
BENCH_ROOT="${EVAL_MAIN_ROOT:-$(cd "${XPL_ROOT}/.." && pwd)}"
UTILS_DIR="${XPL_ROOT}/utils"

policy_name="$(basename "${SCRIPT_DIR}")"
yaml_file="${XPL_ROOT}/policy/${policy_name}/deploy.yml"

# Keep the env-client's workspace resolution pinned to this adapter checkout.
# Web normally supplies these markers, but direct/scheduler launches may omit
# them; falling back to the benchmark's generic XPolicyLab would silently load
# a different policy tree and can suppress the WS client phase.
export XPOLICYLAB_ROOT="${XPOLICYLAB_ROOT:-${XPL_ROOT}}"
export EGOVLA_POLICY_ROOT="${EGOVLA_POLICY_ROOT:-${XPL_ROOT}/policy/${policy_name}}"
export EGOVLA_POLICY_EVAL_SCRIPT="${EGOVLA_POLICY_EVAL_SCRIPT:-${SCRIPT_DIR}/eval.sh}"

echo -e "\033[34m[CLIENT] policy=${policy_name}, task=${task_name}, ckpt=${ckpt_name}\033[0m"
echo -e "\033[34m[CLIENT] server=${policy_server_ip}:${policy_server_port}\033[0m"

# EgoVLA's IsaacLab 1.2 runtime is the omni.isaac.lab extension.  A legacy
# top-level IsaacLab checkout under /personal/final_eval can be injected by a
# shared-worker startup; letting it win creates mixed-version ActionTerm
# imports.  Ask isaac_runtime to bind the top-level compatibility alias to the
# already-loaded benchmark omni package, and remove only that stale root from
# inherited PYTHONPATH.
export BEINGH_ISAACLAB_ALIAS="${BEINGH_ISAACLAB_ALIAS:-1}"
stale_isaaclab_root="${EVAL_STALE_ISAACLAB_ROOT:-/personal/final_eval/RoboDojo/third_party/IsaacLab}"
client_pythonpath="${PYTHONPATH:-}"
if [[ -n "${client_pythonpath}" ]]; then
    filtered_pythonpath=()
    IFS=: read -r -a pythonpath_parts <<< "${client_pythonpath}"
    for path_entry in "${pythonpath_parts[@]}"; do
        case "${path_entry}" in
            "${stale_isaaclab_root}/source"|"${stale_isaaclab_root}/source/"*)
                continue
                ;;
        esac
        [[ -n "${path_entry}" ]] && filtered_pythonpath+=("${path_entry}")
    done
    client_pythonpath="$(IFS=:; echo "${filtered_pythonpath[*]}")"
fi

# The evaluator may invoke this adapter from a minimal web-runner PATH.  Make
# the Conda executable discoverable by setup_env_client.sh (for named envs)
# and by run_*_env_client.sh (which sources conda.sh before activation).
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
client_path="${PATH:-}"
if [[ -x "${conda_exe}" ]]; then
    conda_root="$(cd "$(dirname "${conda_exe}")/.." && pwd)"
    client_path="${conda_root}/bin:${conda_root}/condabin${client_path:+:${client_path}}"
fi

# Do not set CUDA_VISIBLE_DEVICES here.  The evaluator passes the environment
# GPU id to the simulator, which must see the host device numbering unchanged;
# masking it here would renumber the visible devices and can select the wrong
# GPU (or make the requested --device_id unavailable).
exec env \
    PATH="${client_path}" \
    CONDA_EXE="${conda_exe}" \
    PYTHONPATH="${client_pythonpath}" \
    bash "${UTILS_DIR}/setup_env_client.sh" \
        "${UTILS_DIR}" \
        "${yaml_file}" \
        "${eval_env_conda_env}" \
        "${policy_server_port}" \
        "${bench_name}" \
        "${task_name}" \
        "${env_cfg_type}" \
        "${policy_name}" \
        "${additional_info}" \
        "${BENCH_ROOT}" \
        "${seed}" \
        "${env_gpu_id}" \
        "${policy_server_ip}"
