#!/usr/bin/env bash
set -euo pipefail
cd -- "$(dirname -- "${BASH_SOURCE[0]}")"
# Opt-in two-GPU turbo mode: 2+2 stage packing with a shared VAE, DiT
# torch.compile and three denoising steps (see liveavatar_turbo.py). ~31 FPS
# gap-free on two GPUs, quality-equivalent to the released 4-step/5-GPU path.
export LIVEAVATAR_MODE=tpp
export LIVEAVATAR_TURBO=1
export CUDA_VISIBLE_DEVICES="${LIVEAVATAR_GPUS:?Set two available GPU indices, e.g. 0,1}"
IFS=',' read -ra devices <<< "$CUDA_VISIBLE_DEVICES"
if (( ${#devices[@]} != 2 )); then
  echo 'Turbo requires exactly two GPUs (2+2 stage packing with a shared VAE).'
  exit 1
fi
declare -A seen_devices=()
for gpu in "${devices[@]}"; do
  if [[ ! "$gpu" =~ ^[0-9]+$ || -n "${seen_devices[$gpu]:-}" ]]; then
    echo 'Provide two distinct numeric GPU indices.'
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
# The worker enables DiT-only compilation in-process; the VAE stays eager.
export ENABLE_COMPILE=false
export OMP_NUM_THREADS=4
export HOST=127.0.0.1
export PORT="${LIVEAVATAR_PORT:-8791}"
exec /opt/dlami/nvme/.cache_uv/liveavatar-stage1/bin/python -m reactor_runtime.serve
