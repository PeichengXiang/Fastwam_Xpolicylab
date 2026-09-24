#!/bin/bash
set -e

UTILS_DIR="${1}"
yaml_file="${2}"
eval_env_conda_env="${3}"
policy_server_port="${4}"
bench_name="${5}"
task_name="${6}"
env_cfg_type="${7}"
policy_name="${8}"
additional_info="${9}"
ROOT_DIR="${10}"
seed="${11}"
env_gpu_id="${12}"
policy_server_ip="${13:-localhost}"
protocol_override="${14:-}"

# shellcheck source=resolve_eval_env_type.sh
source "${UTILS_DIR}/resolve_eval_env_type.sh"
eval_env_mode="$(resolve_eval_env_type)" || exit 1

# The web runner deliberately starts from a minimal environment.  In that
# environment, bare `python` can resolve to the base Conda interpreter even
# though the evaluator runtime was passed explicitly.  Read the deployment
# YAML with that evaluator interpreter so PyYAML and the subsequent simulator
# always use the same environment.
yaml_python=(python)
if [[ -x "${eval_env_conda_env%/}/bin/python" ]]; then
    # A prefix path can be used directly, even when this shell has not yet
    # sourced Conda's shell integration.
    yaml_python=("${eval_env_conda_env%/}/bin/python")
else
    # The web runner intentionally starts with a minimal PATH.  Resolve a
    # named environment without relying on `conda activate` (which happens
    # later in run_*_env_client.sh).
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
    if [[ -x "${conda_exe}" && "${eval_env_conda_env}" != */* ]]; then
        yaml_python=("${conda_exe}" run --no-capture-output -n "${eval_env_conda_env}" python)
    fi
fi

read eval_batch yaml_protocol < <("${yaml_python[@]}" - <<PY
import yaml
with open("${yaml_file}", "r") as f:
    data = yaml.safe_load(f)
print(
    str(data.get("eval_batch", False)).lower(),
    data.get("protocol", "ws"),
)
PY
)
protocol="${protocol_override:-${yaml_protocol}}"
# Allow an adapter/eval command to select the single-environment or batch RPC
# explicitly without editing the shared deploy.yml.  This is especially useful
# for debug smoke tests and for EgoVLA's one-environment simulator runs.
if [[ -n "${FASTWAM_EVAL_BATCH:-}" ]]; then
    eval_batch="${FASTWAM_EVAL_BATCH,,}"
fi

if [[ -z "${EVAL_ENV_TYPE:-}" ]]; then
    echo "[CLIENT] EVAL_ENV_TYPE=(default sim) -> ${eval_env_mode}"
else
    echo "[CLIENT] EVAL_ENV_TYPE=${EVAL_ENV_TYPE} -> ${eval_env_mode}"
fi

COMMON_ARGS=(
    "${eval_batch}"
    "${eval_env_conda_env}"
    "${policy_server_port}"
    "${bench_name}"
    "${task_name}"
    "${env_cfg_type}"
    "${policy_name}"
    "${additional_info}"
    "${ROOT_DIR}"
    "${seed}"
    "${env_gpu_id}"
    "${policy_server_ip}"
)

if [[ "${eval_env_mode}" == "debug" ]]; then
    bash "${UTILS_DIR}/run_debug_env_client.sh" "${COMMON_ARGS[@]}" "${protocol}"
elif [[ "${eval_env_mode}" == "sim" ]]; then
    bash "${UTILS_DIR}/run_sim_env_client.sh" "${COMMON_ARGS[@]}" "${protocol}"
elif [[ "${eval_env_mode}" == "real_world" ]]; then
    echo -e "\033[31m[WARN] EVAL_ENV_TYPE=real: real-world evaluation is not supported in the open-source release; continuing to real env client.\033[0m" >&2
    bash "${UTILS_DIR}/run_real_env_client.sh" "${COMMON_ARGS[@]}" "${protocol}"
else
    echo "[ERROR] Unknown eval env mode: ${eval_env_mode}" >&2
    exit 1
fi
