#!/bin/bash
set -e

eval_batch="${1}"
eval_env_conda_env="${2}"
policy_server_port="${3}"
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

trace_log="${FASTWAM_CLIENT_TRACE_LOG:-/tmp/egovla-fastwam-client-${LUMINIS_RUN_ID:-manual}-${BASHPID}.trace}"
trace_dir="$(dirname "${trace_log}")"
mkdir -p "${trace_dir}"
trace() {
    printf "[FASTWAM-CLIENT-TRACE] %s\n" "$*" | tee -a "${trace_log}"
}
export BEINGH_ISAACLAB_ALIAS="${BEINGH_ISAACLAB_ALIAS:-1}"
export BEINGH_RUNTIME_TRACE="${BEINGH_RUNTIME_TRACE:-1}"
trace "phase=start pid=$$ eval_env=${eval_env_conda_env} server=${policy_server_ip}:${policy_server_port} gpu=${env_gpu_id} protocol=${protocol} isaaclab_alias=${BEINGH_ISAACLAB_ALIAS}"
trace "phase=paths root_dir=${root_dir} xpolicy_root=${XPOLICYLAB_ROOT:-<unset>} policy_root=${EGOVLA_POLICY_ROOT:-<unset>} eval_script=${EGOVLA_POLICY_EVAL_SCRIPT:-<unset>}"
trap 'rc=$?; trace "phase=exit pid=$$ rc=${rc}"; exit "${rc}"' EXIT

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
echo -e "\033[34m[CLIENT] Connecting to server ${policy_server_ip}:${policy_server_port}...\033[0m"
echo -e "\033[34m[CLIENT] Watch for green [CONNECTED]; yellow [RECONNECT] means the client is retrying.\033[0m"

# H1-Inspire is an official single-environment bridge.  The scheduler places
# the per-background contract in environment variables; relay it explicitly
# so the bridge cannot fall back to FastWAM's legacy deploy defaults or the
# 120-second WS call timeout during model warm-up.
egovla_cli_args=()
if [[ "${env_cfg_type}" == "ego_h1_inspire" ]]; then
    eval_batch="false"
    export EGOVLA_NUM_ENVS="${EGOVLA_NUM_ENVS:-1}"
    export EGOVLA_EPISODES="${EGOVLA_EPISODES:-${EVAL_NUM:-1}}"
    export EVAL_NUM="${EVAL_NUM:-${EGOVLA_EPISODES}}"
    egovla_cli_args+=(
        --episodes "${EGOVLA_EPISODES}"
        --num-envs "${EGOVLA_NUM_ENVS}"
        --benchmark-protocol "${EGOVLA_PROTOCOL:-official}"
        --official-split "${EGOVLA_OFFICIAL_SPLIT:-single}"
        --request-timeout-s "${EGOVLA_REQUEST_TIMEOUT_S:-600}"
    )
    # Official trial controls are invalid for a basic diagnostic run.
    # Preserve the existing official defaults and per-background overrides.
    if [[ "${EGOVLA_PROTOCOL:-official}" == "official" ]]; then
        egovla_cli_args+=(
            --official-episodes "${EGOVLA_OFFICIAL_EPISODES:-${EGOVLA_EPISODES}}"
            --official-trials "${EGOVLA_OFFICIAL_TRIALS:-1}"
        )
    fi
    if [[ -n "${EGOVLA_OUTPUT_DIR:-}" ]]; then
        egovla_cli_args+=(--output-dir "${EGOVLA_OUTPUT_DIR}")
    fi
    if [[ -n "${EGOVLA_CHECKPOINT:-}" ]]; then
        egovla_cli_args+=(--checkpoint "${EGOVLA_CHECKPOINT}")
    fi
    if [[ -n "${EGOVLA_ACTION_TYPE:-}" ]]; then
        egovla_cli_args+=(--action-type "${EGOVLA_ACTION_TYPE}")
    fi
    if [[ -n "${EGOVLA_ROOM_IDX:-}" ]]; then
        egovla_cli_args+=(--room-idx "${EGOVLA_ROOM_IDX}")
    fi
    if [[ -n "${EGOVLA_TABLE_IDX:-}" ]]; then
        egovla_cli_args+=(--table-idx "${EGOVLA_TABLE_IDX}")
    fi
    if [[ -n "${EGOVLA_RESULT_ROLE:-}" ]]; then
        egovla_cli_args+=(--result-role "${EGOVLA_RESULT_ROLE}")
    fi
    xpolicy_root_arg="${EGOVLA_XPOLICY_ROOT:-${XPOLICYLAB_ROOT:-}}"
    if [[ -n "${xpolicy_root_arg}" ]]; then
        egovla_cli_args+=(--xpolicy-root "${xpolicy_root_arg}")
    fi
    trace "phase=contract eval_batch=${eval_batch} episodes=${EGOVLA_EPISODES} split=${EGOVLA_OFFICIAL_SPLIT:-single} output=${EGOVLA_OUTPUT_DIR:-<default>} timeout=${EGOVLA_REQUEST_TIMEOUT_S:-600}"
fi

set +e
bash "${root_dir}/scripts/eval_policy.sh" \
    --bench_name "${bench_name}" \
    --task_name "${task_name}" \
    --env_cfg_type "${env_cfg_type}" \
    --policy_name "${policy_name}" \
    --host "${policy_server_ip}" \
    --port "${policy_server_port}" \
    --protocol "${protocol}" \
    --eval_batch "${eval_batch}" \
    --root_dir "${root_dir}" \
    --device_id "${env_gpu_id}" \
    --additional_info "${additional_info}" \
    --seed "${seed}" \
    "${egovla_cli_args[@]}" 2>&1 | tee -a "${trace_log}"
child_rc=${PIPESTATUS[0]}
set -e
trace "phase=eval_policy_return rc=${child_rc}"
exit "${child_rc}"
