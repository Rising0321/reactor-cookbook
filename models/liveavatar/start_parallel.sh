#!/usr/bin/env bash
set -euo pipefail
cd -- "$(dirname -- "${BASH_SOURCE[0]}")"
export LIVEAVATAR_MODE=tpp
export CUDA_VISIBLE_DEVICES="${LIVEAVATAR_GPUS:?Set five available GPU indices, e.g. 0,1,2,3,4}"
IFS=',' read -ra devices <<< "$CUDA_VISIBLE_DEVICES"
if (( ${#devices[@]} != 5 )); then
  echo 'The released TPP path requires exactly five GPUs (four denoising stages and one VAE).'
  exit 1
fi
declare -A seen_devices=()
for gpu in "${devices[@]}"; do
  if [[ ! "$gpu" =~ ^[0-9]+$ || -n "${seen_devices[$gpu]:-}" ]]; then
    echo 'Provide five distinct numeric GPU indices.'
    exit 1
  fi
  seen_devices[$gpu]=1
  free_mib=$(nvidia-smi -i "$gpu" --query-gpu=memory.free --format=csv,noheader,nounits)
  if (( free_mib < 80000 )); then
    echo "Refusing GPU $gpu: need 80000 MiB free; other jobs untouched."
    exit 1
  fi
done
export UV_CACHE_DIR=/opt/dlami/nvme/.cache_uv
export HF_HOME=/opt/dlami/nvme/.cache_hf
export XDG_CACHE_HOME=/opt/dlami/nvme/.cache_hf/liveavatar-cache
export TMPDIR=/opt/dlami/nvme/.cache_hf/reactor_registry/liveavatar-stage1/tmp
export CUTE_DSL_CACHE_DIR=/opt/dlami/nvme/.cache_hf/reactor_registry/liveavatar-stage1/cute
export ENABLE_COMPILE=false
export OMP_NUM_THREADS=4
export HOST=127.0.0.1
export PORT="${LIVEAVATAR_PORT:-8791}"
exec /opt/dlami/nvme/.cache_uv/liveavatar-stage1/bin/python -m reactor_runtime.serve
