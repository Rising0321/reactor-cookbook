#!/usr/bin/env bash
set -euo pipefail
cd -- "$(dirname -- "${BASH_SOURCE[0]}")"
export DOCKER_HOST="${DOCKER_HOST:-unix:///run/reactor-worldmodels-docker/docker.sock}"
export HF_HOME=/opt/dlami/nvme/.cache_hf
export UV_CACHE_DIR=/opt/dlami/nvme/.cache_uv
IFS=',' read -ra devices <<< "${LIVEAVATAR_GPUS:?Set two available GPU indices, e.g. 0,1}"
if (( ${#devices[@]} != 2 )); then
  echo 'The container Turbo profile requires two GPUs.' >&2
  exit 1
fi
declare -A seen_devices=()
for gpu in "${devices[@]}"; do
  if [[ ! "$gpu" =~ ^[0-9]+$ || -n "${seen_devices[$gpu]:-}" ]]; then
    echo 'Provide two distinct numeric GPU indices.' >&2
    exit 1
  fi
  seen_devices[$gpu]=1
  free_mib=$(nvidia-smi -i "$gpu" --query-gpu=memory.free --format=csv,noheader,nounits)
  if (( free_mib < 110000 )); then
    echo "Refusing GPU $gpu: need 110000 MiB free; other jobs untouched." >&2
    exit 1
  fi
done
exec reactor run --gpus "\"device=${LIVEAVATAR_GPUS}\"" \
  --port "${LIVEAVATAR_PORT:-8793}" \
  --weights "${LIVEAVATAR_WEIGHTS:-/opt/dlami/nvme/.cache_hf/reactor_registry/liveavatar-deploy}"
