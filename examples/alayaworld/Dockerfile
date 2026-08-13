# syntax=docker/dockerfile:1
# Reactor AlayaWorld image with the build tools required by torch.compile.

FROM python:3.12-slim

WORKDIR /app

# Git prepares the pinned public source checkouts at model load. TorchInductor
# compiles small host-side extensions for the selected CUDA device.
RUN apt-get update \
    && apt-get install --yes --no-install-recommends build-essential git \
    && rm -rf /var/lib/apt/lists/*

# Keep the expensive dependency layer reusable across adapter-only edits.
COPY requirements.txt /app/requirements.txt
RUN python -m pip install --no-cache-dir -r /app/requirements.txt

COPY . /app/

ENV PYTHONUNBUFFERED=1 \
    HOST=0.0.0.0 \
    PORT=8080 \
    PYTORCH_ALLOC_CONF=expandable_segments:True
EXPOSE 8080
ENTRYPOINT ["python", "-m", "reactor_runtime.serve"]
