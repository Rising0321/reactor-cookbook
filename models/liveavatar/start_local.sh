#!/usr/bin/env bash
set -euo pipefail
cd -- "$(dirname -- "${BASH_SOURCE[0]}")"
export UV_CACHE_DIR=/opt/dlami/nvme/.cache_uv
export UV_PYTHON_INSTALL_DIR=/opt/dlami/nvme/.cache_uv/python
export HF_HOME=/opt/dlami/nvme/.cache_hf
export XDG_CACHE_HOME=/opt/dlami/nvme/.cache_hf/liveavatar-cache
export TORCH_HOME=/opt/dlami/nvme/.cache_hf/torch
export TMPDIR=/opt/dlami/nvme/.cache_hf/reactor_registry/liveavatar-stage1/tmp
mkdir -p "$TMPDIR"
export ENABLE_COMPILE=false
export CUDA_VISIBLE_DEVICES="${LIVEAVATAR_GPU:?Set LIVEAVATAR_GPU to one available GPU index}"
export HOST=127.0.0.1
export PORT="${LIVEAVATAR_PORT:-8791}"
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
# Keep a conservative full-default-cache budget and leave headroom for peers.
free_mib=$(nvidia-smi -i "$CUDA_VISIBLE_DEVICES" --query-gpu=memory.free --format=csv,noheader,nounits)
if (( free_mib < 100000 )); then
  echo "Refusing GPU $CUDA_VISIBLE_DEVICES: $free_mib MiB free; need at least 100000 MiB. Other jobs are untouched." >&2
  exit 1
fi
exec /opt/dlami/nvme/.cache_uv/liveavatar-stage1/bin/python -m reactor_runtime.serve
