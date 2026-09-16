# LiveAvatar: native YAML build and two-B200 container

## Contract documentation — v0.3.4

The command summaries and input/message descriptions cover validation limits,
input timing, readiness, replies and progress. The generated schema preserves
all command names, fields, defaults, constraints and output tracks. The
description-only comparison passes, together with 42 CPU tests and schema
render reproducibility. README documents the YAML image build and avatar
upload/start flow. The running container remains on its tested image until an
operator explicitly rebuilds and restarts it.

## Interactive idle handling — v0.3.3

Each TPP rank waits on its own CPU command queue between takes. A GPU broadcast
must not wait for human input: with the native 120-second NCCL timeout, an idle
nonzero rank could otherwise time out before the next `start`. Every rank now
receives the same job through its queue; inference collectives, sampling and KV
cache logic are unchanged. A container test waited 135 seconds before starting,
then received all 45 frames of one chunk and audio over a TCP-only isolated
client. The schema is unchanged and all 40 CPU tests pass.

## Acceptance — 2026-09-16

`reactor build --no-dockerfile` and actual `reactor run` acceptance completed on
GPU 0/1. Two independent SDK sessions uploaded different identity images, audio
and prompts and both reported `generation_ended: complete`, six generated clips,
285 generated frames each. The SDK captured 285 fashion-blogger frames and 284
kitchen-grandmother frames. The original PR's source-based run showed the same
285/284 receive counts: the single-frame receive discrepancy is recorded, not
hidden or claimed to be lossless transport. Its exact transport cause was not
isolated in this deployment test.

Both received audio tracks are mono 48000 Hz, finite and unclipped (peaks 0.679
and 0.596). Five speech templates in the fashion sample matched the resampled
reference with correlations 0.959–0.998, confirming the correct playback rate.
Four grandmother templates matched at 0.979–0.992; the 2.25-second template was
only 0.351, also low in the native run. These checks do not establish perfect
lip sync or gap-free startup: the transport inserts silence while waiting for
generation. Inspected temporal sheets retained the two identities/backgrounds
with plausible speech gestures.

After the first two clips, observed worker build times were about 1.43–1.51 s
per 48 frames (1.92 s of playback). Those timings exclude demand-ack waiting and
allow native in-flight stages; they are not an independent unpaced FPS benchmark.
First-clip preparation/compilation was 77.64 s in the first session and 28.12 s
with a different prompt/subject. Cold startup is not real time.

CPU checks: 40 tests passed, Ruff passed, `reactor validate` passed, and the
client-facing schema remained byte-identical. The image import preflight checks
both FA2 and FA4. GPU logs recorded nonzero FA4 execution on both ranks, with
the existing supported FA2 fallback retained.

Evidence directory:
`/opt/dlami/nvme/ruixing/liveavatar-test-kit-20260916/verification/pr53-reactor-run-final/`.
It contains `complete.json`, `container.json`, service/client logs, per-subject
videos/received PCM, `media-verification.json` and `contact-sheet.jpg`.
The earlier `pr53-reactor-run` failed on the FA2/FA4 namespace collision, which
the build now fixes; `pr53-reactor-run-fa4` was interrupted by the GPU safety
guard, not counted as successful. With user approval, later checks watch only
the selected GPUs; unrelated tasks on other GPUs are left alone.

Based on PR #53 head `42611d7` and the Reactor
[build](https://docs.reactor.inc/deploy/platform/build) and
[weights](https://docs.reactor.inc/deploy/development/weights) contracts.
This is local `reactor build` / `reactor run` deployment. It does not publish an
image, upload weights, or activate a cloud deployment.

## Profile and fidelity

The image enables the colleague's Turbo mode: two B200s, shared streaming VAE,
DiT-only `torch.compile`, FA4, **three denoising steps**. Native KV allocations,
explicit image/audio uploads, `start`, 25 FPS video and genuinely resampled
48 kHz mono audio are retained. Three steps are not numerically identical to the
original four-step model; the upstream PR's quality assessment is not a proof of
pixel parity. No default image or automatic take is introduced.

The image pins Runtime 3.2.5, Python 3.12, CUDA 12.8.1, Torch 2.8.0/cu128,
FlashAttention 2.8.3 and FA4 4.0.0b30. FA2 uses the official ABI-matched binary
wheel so the dependency solver does not need to build it before Torch exists.
Both distributions contain files in `flash_attn.cute`; the YAML build therefore
reinstalls FA4 last and checks both imports. A single concurrent dependency
installation can leave the old FA2 CuTe files in place despite reporting the
correct FA4 package version.
The pinned upstream source is installed by `build.run`; model weights are not.
No Dockerfile exists in this model workspace. `reactor.yaml` is the build source.

## Prepare local weights

Use the existing NVMe environment to materialize the pinned HF snapshots:

```bash
/opt/dlami/nvme/.cache_uv/liveavatar-stage1/bin/python prepare_weights.py \
  --output /opt/dlami/nvme/.cache_hf/reactor_registry/liveavatar-deploy
```

The default is cache-only; pass `--allow-download` if snapshots are absent.
Within the same filesystem this creates hard links to cached blobs, not another
large download. There are no symlinks in the weight bundle, and existing different
files are never overwritten. The bundle contains `wan2_2/` and `liveavatar_lora/`.

Model loading resolves those folders using `reactor_runtime.get_weights_path()`.
The container is offline for HF and fails clearly if weights are missing.
For this local workflow, writable compile/temp caches live in `.runtime/` beside
the mounted weights and persist across restarts. Before a future cloud publish,
stage **only** `wan2_2/` and `liveavatar_lora/` in a clean bundle; `.runtime/` is
local cache, not model weights. This local cache setup expects a writable mount.

## Build and run

Run from this model directory:

```bash
export DOCKER_HOST=unix:///run/reactor-worldmodels-docker/docker.sock
export HF_HOME=/opt/dlami/nvme/.cache_hf
export UV_CACHE_DIR=/opt/dlami/nvme/.cache_uv
reactor validate
reactor build --no-dockerfile

# Select two available GPUs; the helper invokes reactor run, not uv.
LIVEAVATAR_GPUS=0,1 bash run_container.sh
```

The direct equivalent is:

```bash
reactor run --gpus '"device=0,1"' --port 8793 \
  --weights /opt/dlami/nvme/.cache_hf/reactor_registry/liveavatar-deploy
```

The existing dedicated daemon stores image layers and BuildKit caches under
`/opt/dlami/nvme/.cache_docker/worldmodels-data`. It is separate from the public
daemon running the Buildkite agent; no Docker service restart is needed.
Check `docker info --format '{{.DockerRootDir}}'` with the same `DOCKER_HOST`.
The container's `.runtime/` cache is under the mounted NVMe HF bundle; no HF token
is baked into the image or required for the already materialized checkpoints.

Always rebuild after code/dependency edits: `reactor run` reuses the existing
`reactor-local/liveavatar:dev` tag. The helper checks memory but is not an exclusive
GPU reservation. To stop, identify **your own** container with `docker ps`, then
use `docker stop <that-container-id>`; never stop all containers on the daemon.

## Client behavior

Wait for `GET http://127.0.0.1:8793/health` to report `state: available` before
connecting. HTTP 200 alone is not readiness: Runtime also answers during loading.
Upload an identity image and driving audio, optionally set the prompt, then send
`start`. The native Runtime API and existing sandbox instructions are unchanged.
First compilation adds latency; no preloaded/default avatar is used to hide it.

The SDK smoke tests use explicit inputs and capture actual received video/PCM.
Their `take.mp4` is a convenient preview muxed with uploaded audio, whereas
`audio.wav` is the actual 48 kHz received stream and is the audio-verification
artifact. Local host-to-container RTC acceptance does not configure public
internet routing or TURN for a remote browser.
