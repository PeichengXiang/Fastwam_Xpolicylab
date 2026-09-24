#!/usr/bin/env bash
set -euo pipefail

# Download the exact files selected by FastWAM's official _resolve_configs.
# ModelScope's Python downloader was extremely slow on this host, while the
# official resolve endpoint supports reliable HTTP range resume.

WORKSPACE=${WORKSPACE:-/personal/xiangpc/0812_Xpolicylab_bench/FastWAM}
BASE="$WORKSPACE/pretrain_model"
LOG_PREFIX="FASTWAM_DIRECT"

# `proxy_on` is useful for many downloads on this cluster, but this particular
# ModelScope endpoint is substantially faster via the direct route.
unset http_proxy https_proxy HTTP_PROXY HTTPS_PROXY ALL_PROXY all_proxy

download_exact() {
    local url=$1
    local output=$2
    local expected_size=$3
    local partial="${output}.part"

    mkdir -p "$(dirname "$output")"
    if [[ -f "$output" ]] && [[ $(stat -c %s "$output") -eq $expected_size ]]; then
        echo "$LOG_PREFIX SKIP $(basename "$output") size=$expected_size"
        return
    fi

    curl --fail --location --continue-at - \
        --retry 20 --retry-delay 3 --retry-all-errors \
        --output "$partial" "$url"

    local actual_size
    actual_size=$(stat -c %s "$partial")
    if [[ $actual_size -ne $expected_size ]]; then
        echo "$LOG_PREFIX SIZE_MISMATCH file=$partial expected=$expected_size actual=$actual_size" >&2
        exit 1
    fi
    mv "$partial" "$output"
    echo "$LOG_PREFIX OK $(basename "$output") size=$actual_size"
}

converted="$BASE/DiffSynth-Studio/Wan-Series-Converted-Safetensors"
sdk_partial="$converted/._____temp/models_t5_umt5-xxl-enc-bf16.safetensors"
t5_output="$converted/models_t5_umt5-xxl-enc-bf16.safetensors"
if [[ ! -e "${t5_output}.part" && -f "$sdk_partial" ]]; then
    mkdir -p "$converted"
    mv "$sdk_partial" "${t5_output}.part"
    echo "$LOG_PREFIX REUSE_MODELSCOPE_PARTIAL size=$(stat -c %s "${t5_output}.part")"
fi

download_exact \
    "https://modelscope.cn/models/DiffSynth-Studio/Wan-Series-Converted-Safetensors/resolve/master/models_t5_umt5-xxl-enc-bf16.safetensors" \
    "$t5_output" 11361845432
download_exact \
    "https://modelscope.cn/models/DiffSynth-Studio/Wan-Series-Converted-Safetensors/resolve/master/Wan2.2_VAE.safetensors" \
    "$converted/Wan2.2_VAE.safetensors" 1409401152

tokenizer="$BASE/Wan-AI/Wan2.1-T2V-1.3B/google/umt5-xxl"
download_exact \
    "https://modelscope.cn/models/Wan-AI/Wan2.1-T2V-1.3B/resolve/master/google/umt5-xxl/special_tokens_map.json" \
    "$tokenizer/special_tokens_map.json" 6623
download_exact \
    "https://modelscope.cn/models/Wan-AI/Wan2.1-T2V-1.3B/resolve/master/google/umt5-xxl/spiece.model" \
    "$tokenizer/spiece.model" 4548313
download_exact \
    "https://modelscope.cn/models/Wan-AI/Wan2.1-T2V-1.3B/resolve/master/google/umt5-xxl/tokenizer.json" \
    "$tokenizer/tokenizer.json" 16837417
download_exact \
    "https://modelscope.cn/models/Wan-AI/Wan2.1-T2V-1.3B/resolve/master/google/umt5-xxl/tokenizer_config.json" \
    "$tokenizer/tokenizer_config.json" 61728

echo "$LOG_PREFIX COMPLETE"
