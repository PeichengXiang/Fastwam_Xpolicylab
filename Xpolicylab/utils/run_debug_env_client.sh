#!/bin/bash
set -e

eval_batch="${1}"
eval_env_conda_env="${2}"
free_port="${3}"
bench_name="${4}"
task_name="${5}"
env_cfg_type="${6}"
policy_name="${7}"
additional_info="${8}"
root_dir="${9}"
seed="${10}"
env_gpu_id="${11}"
policy_server_ip="${12:-localhost}"
protocol="${13:-ws}"

# Support both conda environment names and explicit prefixes when this script
# is called from a non-login/scheduler shell.
conda_base="${FASTWAM_CONDA_BASE:-${CONDA_BASE:-}}"
if [[ -n "${conda_base}" && -x "${conda_base}/bin/conda" ]]; then
    export PATH="${conda_base}/bin:${PATH}"
fi
if ! command -v conda >/dev/null 2>&1; then
    for candidate in /personal/miniconda3 /opt/conda; do
        if [[ -x "${candidate}/bin/conda" ]]; then
            export PATH="${candidate}/bin:${PATH}"
            break
        fi
    done
fi
if [[ -x "${eval_env_conda_env%/}/bin/python" ]]; then
    export PATH="${eval_env_conda_env%/}/bin:${PATH}"
elif command -v conda >/dev/null 2>&1; then
    # `conda activate` also handles named environments; source its shell hook.
    conda_base="$(conda info --base 2>/dev/null || true)"
    if [[ -f "${conda_base}/etc/profile.d/conda.sh" ]]; then
        # shellcheck disable=SC1090
        source "${conda_base}/etc/profile.d/conda.sh"
        conda deactivate || true
        conda activate "${eval_env_conda_env}"
    else
        echo "[CLIENT] conda shell hook not found; cannot activate ${eval_env_conda_env}" >&2
        exit 1
    fi
else
    echo "[CLIENT] cannot locate conda to activate ${eval_env_conda_env}" >&2
    exit 1
fi

echo -e "\033[34m[CLIENT] Activating Conda environment: ${eval_env_conda_env}\033[0m"
echo -e "\033[34m[CLIENT] Connecting to server ${policy_server_ip}:${free_port}...\033[0m"
echo -e "\033[34m[CLIENT] Watch for green [CONNECTED]; yellow [RECONNECT] means the client is retrying.\033[0m"

export PYTHONPATH="${root_dir}/XPolicyLab:${root_dir}${PYTHONPATH:+:${PYTHONPATH}}"

debug_args=(
    "${root_dir}/XPolicyLab/debug_env_client.py"
    --bench_name "${bench_name}"
    --task_name "${task_name}"
    --env_cfg_type "${env_cfg_type}"
    --policy_name "${policy_name}"
    --protocol "${protocol}"
    --host "${policy_server_ip}"
    --port "${free_port}"
    --eval_batch "${eval_batch}"
)
if [[ -n "${FASTWAM_DEBUG_EPISODES:-}" ]]; then
    debug_args+=(--eval_episode_num "${FASTWAM_DEBUG_EPISODES}")
fi
python \
    "${debug_args[@]}"
