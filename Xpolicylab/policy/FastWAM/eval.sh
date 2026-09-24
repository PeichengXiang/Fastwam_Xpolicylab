#!/bin/bash
set -euo pipefail

# Contract: 10 positional args. `task_name` is the simulator task; `ckpt_name`
# is the full run directory name under checkpoints/ used to resolve weights.
bench_name=$1
task_name=$2
ckpt_name=$3
env_cfg_type=$4
action_type=$5
seed=$6
policy_gpu_id=$7
env_gpu_id=$8
policy_conda_env=$9
eval_env_conda_env=${10}

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
XPL_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"
UTILS_DIR="${XPL_ROOT}/utils"

SERVER_SCRIPT="${SCRIPT_DIR}/setup_eval_policy_server.sh"
CLIENT_SCRIPT="${SCRIPT_DIR}/setup_eval_env_client.sh"

policy_server_port=$(bash "${UTILS_DIR}/get_free_port.sh")
policy_server_ip="localhost"

checkpoint_tag=$(basename "${FASTWAM_CHECKPOINT_PATH:-${ckpt_name}}")
additional_info="ckpt_name=${ckpt_name},checkpoint=${checkpoint_tag},action_type=${action_type}"
POLICY_SSH=()

cleanup() {
    if [[ -n "${REMOTE_PID_FILE:-}" && ${#POLICY_SSH[@]} -gt 0 ]]; then
        echo -e "\033[31m[CLEANUP] stop remote FastWAM server\033[0m"
        "${POLICY_SSH[@]}" "if test -f '${REMOTE_PID_FILE}'; then pid=\$(cat '${REMOTE_PID_FILE}'); kill -TERM \"\${pid}\" 2>/dev/null || true; rm -f '${REMOTE_PID_FILE}'; fi" || true
    fi
    if [[ -n "${SERVER_PID:-}" ]]; then
        echo -e "\033[31m[CLEANUP] kill server PID=${SERVER_PID}\033[0m"
        kill "${SERVER_PID}" 2>/dev/null || true
    fi
}
trap cleanup EXIT

if [[ "${EGOVLA_COMPONENT:-}" == "policy" ]]; then
    policy_server_port="${EGOVLA_POLICY_SERVER_PORT:?EGOVLA_POLICY_SERVER_PORT is required}"
    policy_server_ip="0.0.0.0"
    echo -e "\033[32m[MAIN] start FastWAM policy server, GPU=${policy_gpu_id}, bind=${policy_server_ip}:${policy_server_port}\033[0m"
    bash "${SERVER_SCRIPT}" \
        "${bench_name}" \
        "${task_name}" \
        "${ckpt_name}" \
        "${env_cfg_type}" \
        "${action_type}" \
        "${seed}" \
        "${policy_gpu_id}" \
        "${policy_conda_env}" \
        "${policy_server_port}" \
        "${policy_server_ip}" &
    SERVER_PID=$!
    bash "${UTILS_DIR}/wait_for_policy_server.sh" "127.0.0.1" "${policy_server_port}" "${SERVER_PID}" "Policy server" 1200
    wait "${SERVER_PID}"
    exit $?
fi

if [[ "${EGOVLA_COMPONENT:-}" == "environment" ]]; then
    policy_server_ip="${EGOVLA_POLICY_SERVER_HOST:?EGOVLA_POLICY_SERVER_HOST is required}"
    policy_server_port="${EGOVLA_POLICY_SERVER_PORT:?EGOVLA_POLICY_SERVER_PORT is required}"
    echo -e "\033[32m[MAIN] start FastWAM simulator client, GPU=${env_gpu_id}, server=${policy_server_ip}:${policy_server_port}\033[0m"
    ready=0
    for _ in $(seq 1 600); do
        if timeout 1 bash -c "</dev/tcp/${policy_server_ip}/${policy_server_port}" 2>/dev/null; then
            ready=1
            break
        fi
        sleep 2
    done
    if [[ "${ready}" != "1" ]]; then
        echo "[MAIN] FastWAM policy server did not listen within 1200 seconds" >&2
        exit 1
    fi
    bash "${CLIENT_SCRIPT}" \
        "${bench_name}" "${task_name}" "${ckpt_name}" "${env_cfg_type}" \
        "${action_type}" "${seed}" "${env_gpu_id}" "${eval_env_conda_env}" \
        "${additional_info}" "${policy_server_port}" "${policy_server_ip}"
    echo -e "\033[33m[MAIN] eval finished\033[0m"
    exit 0
fi

if [[ "${FASTWAM_SPLIT_REMOTE:-0}" == "1" ]]; then
    policy_server_ip="127.0.0.1"
    policy_server_port=$((24000 + policy_gpu_id))
    remote_server_port=$((24000 + policy_gpu_id))
    ssh_host="${FASTWAM_POLICY_SSH_HOST:-127.0.0.1}"
    ssh_port="${FASTWAM_POLICY_SSH_PORT:-31544}"
    ssh_user="${FASTWAM_POLICY_SSH_USER:-root}"
    ssh_key="${FASTWAM_POLICY_SSH_KEY:?FASTWAM_POLICY_SSH_KEY is required}"
    known_hosts="${FASTWAM_POLICY_KNOWN_HOSTS:?FASTWAM_POLICY_KNOWN_HOSTS is required}"
    jump_host="${FASTWAM_POLICY_JUMP_HOST:?FASTWAM_POLICY_JUMP_HOST is required}"
    jump_port="${FASTWAM_POLICY_JUMP_PORT:-22}"
    jump_user="${FASTWAM_POLICY_JUMP_USER:-root}"
    jump_key="${FASTWAM_POLICY_JUMP_KEY:?FASTWAM_POLICY_JUMP_KEY is required}"
    jump_known_hosts="${FASTWAM_POLICY_JUMP_KNOWN_HOSTS:?FASTWAM_POLICY_JUMP_KNOWN_HOSTS is required}"
    proxy_command=$(printf 'ssh -F /dev/null -p %q -i %q -o BatchMode=yes -o IdentitiesOnly=yes -o StrictHostKeyChecking=yes -o HostKeyAlias=%q -o UserKnownHostsFile=%q -o GlobalKnownHostsFile=/dev/null -W %%h:%%p %q' \
        "${jump_port}" "${jump_key}" "${jump_host}" "${jump_known_hosts}" "${jump_user}@${jump_host}")
    POLICY_SSH=(ssh -F /dev/null -p "${ssh_port}" -i "${ssh_key}"
        -o BatchMode=yes -o IdentitiesOnly=yes -o StrictHostKeyChecking=yes
        -o UserKnownHostsFile="${known_hosts}" -o GlobalKnownHostsFile=/dev/null
        -o "ProxyCommand=${proxy_command}"
        -o ConnectTimeout=20 "${ssh_user}@${ssh_host}")

    remote_run_id="${LUMINIS_RUN_ID:-manual}-${policy_gpu_id}"
    REMOTE_PID_FILE="/tmp/eval-web-fastwam-${remote_run_id}.pid"
    remote_command=$(printf 'set -euo pipefail; child=""; cleanup(){ if test -n "$child"; then kill -TERM "$child" 2>/dev/null || true; wait "$child" 2>/dev/null || true; fi; rm -f %q; }; trap cleanup EXIT HUP INT TERM; env PATH=%q HOME=/root PYTHONUNBUFFERED=1 FASTWAM_CHECKPOINT_PATH=%q FASTWAM_DATASET_STATS_PATH=%q EVAL_MAIN_ROOT=%q bash %q %q %q %q %q %q %q %q %q %q %q & child=$!; echo "$child" >%q; wait "$child"' \
        "${REMOTE_PID_FILE}" "${PATH}" "${FASTWAM_CHECKPOINT_PATH:-}" "${FASTWAM_DATASET_STATS_PATH:-}" "${EVAL_MAIN_ROOT:-}" \
        "${SERVER_SCRIPT}" "${bench_name}" "${task_name}" "${ckpt_name}" \
        "${env_cfg_type}" "${action_type}" "${seed}" "${policy_gpu_id}" \
        "${policy_conda_env}" "${remote_server_port}" "127.0.0.1" "${REMOTE_PID_FILE}")
    echo -e "\033[32m[MAIN] start remote FastWAM server, A800 GPU=${policy_gpu_id}, endpoint=${policy_server_ip}:${policy_server_port}\033[0m"
    "${POLICY_SSH[@]}" \
        -o ExitOnForwardFailure=yes \
        -L "127.0.0.1:${policy_server_port}:127.0.0.1:${remote_server_port}" \
        "${remote_command}" &
    SERVER_PID=$!
    for _ in $(seq 1 600); do
        if timeout 1 bash -c "</dev/tcp/${policy_server_ip}/${policy_server_port}" 2>/dev/null; then
            break
        fi
        if ! kill -0 "${SERVER_PID}" >/dev/null 2>&1; then
            echo "[MAIN] remote FastWAM server exited during startup" >&2
            exit 1
        fi
        sleep 2
    done
    if ! timeout 1 bash -c "</dev/tcp/${policy_server_ip}/${policy_server_port}" 2>/dev/null; then
        echo "[MAIN] remote FastWAM server did not listen within 1200 seconds" >&2
        exit 1
    fi

    echo -e "\033[32m[MAIN] start 4090 simulator client, GPU=${env_gpu_id}, server=${policy_server_ip}:${policy_server_port}\033[0m"
    bash "${CLIENT_SCRIPT}" \
        "${bench_name}" "${task_name}" "${ckpt_name}" "${env_cfg_type}" \
        "${action_type}" "${seed}" "${env_gpu_id}" "${eval_env_conda_env}" \
        "${additional_info}" "${policy_server_port}" "${policy_server_ip}"
    echo -e "\033[33m[MAIN] eval finished\033[0m"
    exit 0
fi

echo -e "\033[32m[MAIN] start server, policy_server_port=${policy_server_port}\033[0m"
bash "${SERVER_SCRIPT}" \
    "${bench_name}" \
    "${task_name}" \
    "${ckpt_name}" \
    "${env_cfg_type}" \
    "${action_type}" \
    "${seed}" \
    "${policy_gpu_id}" \
    "${policy_conda_env}" \
    "${policy_server_port}" \
    "${policy_server_ip}" &
SERVER_PID=$!
echo -e "\033[32m[MAIN] server PID=${SERVER_PID}\033[0m"

bash "${UTILS_DIR}/wait_for_policy_server.sh" "${policy_server_ip}" "${policy_server_port}" "${SERVER_PID}" "Policy server" 1200

echo -e "\033[32m[MAIN] start client, server=${policy_server_ip}:${policy_server_port}\033[0m"
bash "${CLIENT_SCRIPT}" \
    "${bench_name}" \
    "${task_name}" \
    "${ckpt_name}" \
    "${env_cfg_type}" \
    "${action_type}" \
    "${seed}" \
    "${env_gpu_id}" \
    "${eval_env_conda_env}" \
    "${additional_info}" \
    "${policy_server_port}" \
    "${policy_server_ip}"

echo -e "\033[33m[MAIN] eval finished\033[0m"
