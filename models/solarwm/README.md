# Play SolarWM 5B Stage2 through Reactor Runtime

Run the public [SolarWM](https://github.com/Junchao-cs/SolarWM) Wan2.2
TI2V-5B Stage2 checkpoint as an interactive Reactor backend. A client uploads
an anchor image and prompt, applies six-axis camera motion, and receives one
native autoregressive video chunk per inference turn.

This is the first-stage Reactor SDK adaptation intended for local debugging.
It deliberately does not include a Reactor image build configuration.

## Prerequisites

- Python 3.12, `uv`, CUDA 12.8, and one NVIDIA GPU with sufficient memory.
- Access approval for the gated
  [`junchaoh-cs/SolarWM`](https://huggingface.co/junchaoh-cs/SolarWM) repository.
- About 75 GB of persistent storage for the Wan2.2-5B base assets and Stage2
  checkpoint. The checked-in config places them under
  `/opt/dlami/nvme/.cache_hf/SolarWM`.

## Run locally

Create the development environment on NVMe and install the pinned source plus
Reactor Runtime:

```sh
export UV_CACHE_DIR=/opt/dlami/nvme/.cache_uv
export HF_HOME=/opt/dlami/nvme/.cache_hf
export HF_TOKEN="$HF_KEY"

uv venv --python 3.12 /opt/dlami/nvme/.cache_uv/envs/solarwm-wan5
uv pip install --python /opt/dlami/nvme/.cache_uv/envs/solarwm-wan5/bin/python \
  -r requirements.txt reactor-runtime==3.2.5 reactor-sdk==1.2.0
```

Start Reactor Runtime on one selected GPU:

```sh
cd models/solarwm
export CUDA_VISIBLE_DEVICES=7
export PORT=18087
export HF_HOME=/opt/dlami/nvme/.cache_hf
export UV_CACHE_DIR=/opt/dlami/nvme/.cache_uv
export HF_TOKEN="$HF_KEY"
PYTHONPATH=. /opt/dlami/nvme/.cache_uv/envs/solarwm-wan5/bin/python \
  -m reactor_runtime.serve
```

The endpoint is local at `http://localhost:18087`. Check readiness and schema:

```sh
curl -s localhost:18087/health
curl -s localhost:18087/schema
```

## Controls

Generation waits for `set_image`; there is no default or random image.

- `set_image(image, prompt)` uploads an anchor and starts a fresh world.
- `set_prompt(prompt)` restarts from the uploaded anchor because SolarWM keeps
  prompt cross-attention cached for the complete rollout.
- `set_forward`, `set_strafe`, and `set_vertical` control translation.
- `set_pitch`, `set_yaw`, and `set_roll` control rotation.
- `release_camera` returns every held camera axis to zero.
- `reset(seed)` restarts from the selected image without reloading weights.

Camera values are normalized to `[-1, 1]`, with zero neutral. Translation and
rotation axes can be combined within the same chunk.

## Runtime boundary

SolarWM Stage2 is an autoregressive video model. The adapter follows the
upstream self-forcing NFE4 sampler: each call generates three latent frames,
runs timesteps 1000, 750, 500, and 250, and commits clean latents with
`cache_update_policy="commit_detached"`. Self-attention KV and prompt
cross-attention caches persist between calls.

The native rolling self-KV window is 18 latent frames (six chunks), with no
sink tokens. The first chunk includes the uploaded anchor and causally decodes
to 9 RGB frames; every later chunk decodes to 12. `buffer_size` is therefore
12. No fixed Reactor FPS is declared, so playback follows actual generation
throughput.

The upload path accepts JPEG, PNG, WebP, and BMP files up to 25 MiB and 100
million pixels. It applies SolarWM's 864×480 bilinear resize and center crop,
then uses the upstream Wan2.2 VAE and text encoder. Ending a session releases
rollout caches while retaining loaded weights.

## Schema and tests

```sh
PYTHONPATH=. /opt/dlami/nvme/.cache_uv/envs/solarwm-wan5/bin/python \
  -m reactor_runtime.schema --out /tmp/solarwm-schema.json
PYTHONPATH=. /opt/dlami/nvme/.cache_uv/envs/solarwm-wan5/bin/python \
  -m pytest -q
```
