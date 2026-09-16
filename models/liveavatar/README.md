# LiveAvatar — stage-one SDK adaptation

## Accelerated native TPP + FlashAttention 4

The optional five-B200 path uses the released four-stage temporal denoising
pipeline and a dedicated streaming VAE rank. The external upload / Start /
Stop / Reset contract is unchanged. See [PERFORMANCE.md](PERFORMANCE.md) for
timings, numerical/quality limits and the distinction from the single-GPU path.
See [GPU_COUNT_BENCHMARK.md](GPU_COUNT_BENCHMARK.md) for fewer-GPU stage-packing
benchmarks; those experiments do not change the five-GPU serving launcher.

```bash
# Check GPU ownership and memory first; choose five available devices.
LIVEAVATAR_GPUS=0,1,2,3,4 bash start_parallel.sh
```

The Runtime remains native 3.2.5. It owns five spawned GPU workers; killing other
users' processes is never part of startup. Stop during a take terminates only
these workers to unblock native NCCL safely; the next Start reloads them.
Normal completion retains weights. No default input image or audio is selected.
Use `start_local.sh` for the preserved single-GPU stage-one baseline.

## Two-GPU turbo mode (opt-in)

An optional two-B200 path delivers the same gap-free full-cadence stream on far
less hardware. It packs the four denoising stages 2+2 across two GPUs with the
VAE sharing the second rank (a bit-exact stage repack), `torch.compile`s the DiT
while keeping the streaming VAE eager, and runs three denoising steps. Measured
~32 FPS gap-free steady-state on two GPUs (versus ~50 FPS on five), with quality
measured equivalent to the released 4-step/5-GPU path — `reactor_bench` identity,
temporal-flicker and reference-fidelity within noise, lip-sync held, and no
long-take drift across five subjects including a 52 s take.

```bash
# Two available devices; the first turbo clip pays a one-time DiT compile.
LIVEAVATAR_GPUS=0,1 bash start_turbo.sh
```

Turbo is entirely opt-in: the released five-GPU launcher and defaults are the
quality reference and are unchanged (`liveavatar_turbo.turbo_plan` reproduces the
five-GPU behaviour exactly unless `LIVEAVATAR_TURBO=1`). The external upload /
Start / Stop / Reset contract is identical. The first turbo clip absorbs a
one-time DiT-compilation cost; steady-state clips are gap-free.

## Single-GPU baseline

This workspace provides uploaded-image/audio avatar takes through Reactor Runtime
3.2.5 and the Python Reactor SDK. It does **not** add world-navigation controls,
a default avatar, Docker packaging, FA4, CUDA graphs, or compilation.

**Acceptance status, 2026-09-16:** real single-GPU model loading and SDK generation
passed after installing upstream's required FlashAttention 2.8.3. A three-clip
test delivered all 141 frames at 25 FPS; the final six-clip test delivered all
285 frames (11.4 seconds). Temporal samples and inspected clip boundaries
show stable identity/background and plausible speech gestures. Sixteen CPU tests,
schema rendering, and lint checks pass. See [GPU_TEST_REPORT.md](GPU_TEST_REPORT.md)
for the final longer-take results and precise acceptance limits.

**Audio correction (v0.2.0):** the earlier GPU recording's SDK audio had a
16/48 kHz mismatch, not just waiting silence. Output is now actually resampled
to mono 48 kHz once per take and declared at 48 kHz; native model conditioning
remains 16 kHz. CPU Runtime-to-SDK speech regression passes. The old `take.mp4`
used uploaded audio and was not evidence of correct live playback. See
[AUDIO_FIX_REPORT.md](AUDIO_FIX_REPORT.md).

This is not yet a real-time-speed implementation: the live audio track inserts
silence while waiting for the next generated clip. No FA4, CUDA graphs or compile
optimization has been added. No unrelated GPU task was stopped.

## Start locally

All shell commands below run from this directory:

```bash
cd /opt/dlami/nvme/ruixing/reactor-cookbook-liveavatar-20260916/models/liveavatar
LIVEAVATAR_GPU=0 bash start_local.sh
```

Choose an available GPU index first. The script refuses to launch below 100,000
MiB free VRAM; it never stops other processes. It binds to `127.0.0.1:8791` by
default (`LIVEAVATAR_PORT` overrides the port). Keep this foreground terminal open;
Ctrl-C stops only this service.

The environment is `/opt/dlami/nvme/.cache_uv/liveavatar-stage1`. Source/checkpoint
setup is idempotent:

```bash
/opt/dlami/nvme/.cache_uv/liveavatar-stage1/bin/python liveavatar_assets.py
```

The manifest is used only for the Runtime import and schema, not to build a
container. `liveavatar_pipeline.py` has that suffix to avoid shadowing upstream's
`liveavatar` Python package.

## Upload and record a test take

For browser uploads, see [SANDBOX.md](SANDBOX.md). In v0.3.0, `start` takes no
parameters: configure `set_generation_options` first if needed, then explicitly
send `start`. The model exposes no pause/resume/step commands.

No image is selected at startup. The client must supply image and audio explicitly.
From another terminal, with the service running:

```bash
/opt/dlami/nvme/.cache_uv/liveavatar-stage1/bin/python check_model.py \
  --image /absolute/path/to/uploaded-avatar.png \
  --audio /absolute/path/to/uploaded-speech.wav \
  --prompt 'A person speaking naturally to the camera.' \
  --chunks 3 \
  --output /opt/dlami/nvme/.cache_hf/reactor_registry/liveavatar-stage1/real-test
```

The script uses `reactor_sdk`, uploads both files, reads named audio/video tracks,
and saves command messages. `video.mp4` contains received frames at model-native
25 FPS; `audio.wav` preserves received PCM, including transport-inserted silence.
`take.mp4` pairs the gapless received frames with the uploaded driving audio,
as upstream's offline exporter does. It is a **model-timeline quality preview**,
not a wall-clock recording demonstrating real-time speed. The script fails for
missing media or a generation error. The CPU test video is not a substitute.

## Client contract

| Command | Meaning |
| --- | --- |
| `set_avatar_image` | Upload identity/reference image; no built-in fallback |
| `set_audio` | Upload driving WAV/MP3/FLAC, normalized to mono 16 kHz |
| `set_pose_video` | Upload a prepared pose video, or null to clear it |
| `set_prompt` | Set optional scene text and compatibility negative text |
| `set_generation_options` | Set seed and clip limit while idle; never starts generation |
| `start` | Explicit no-argument start button; requires accepted image and audio |
| `stop` | End the take, retain conditions and flush output |
| `reset` | End the take and clear all selected conditions |

Conditions may be changed only while idle, preventing ambiguous mid-take changes.
`state_update` is the full snapshot on connection, accepted changes and clip
completion. `input_accepted` and `take_changed` are awaited command replies;
`chunk_complete` and `generation_ended` are progress broadcasts. Rejected commands
return `command_error`. Uploaded free-form content is marked for moderation in
the schema; actual moderation is a deployment responsibility.

`main_video` uses native **25 FPS**. `main_audio` carries the normalized input
audio, not synthesized speech. One inference turn emits one native 48-frame
clip (45 frames for the first clip, following the upstream three-frame trim).
The runtime waits for the next turn before allowing the producer to continue.

The selected four-step branch has no classifier-free-guidance application:
upstream computes negative-text embeddings but does not use them in the denoiser.
`negative_prompt` is forwarded for compatibility, **not advertised as an effective
negative-control feature**.

## Input coverage and boundaries

Supported native inference conditions: image, driving audio, positive text,
negative text passthrough, optional prepared pose video, seed, and clip limit.
Resolution uses the released fast single-card recipe's `704*384` area and native
image-aspect fitting. Pose decoding uses upstream unchanged; real pose-conditioned
generation remains untested.

Not claimed as implemented: optional CosyVoice TTS, multi-speaker SAM2 routing,
or external/local-LLM prompt expansion. These require separate auxiliary models
and dependencies; the released TTS bootstrap also references a top-level `wan`
package absent from this checkout. Supply speech audio directly for this recipe.
The upstream flags `init_first_frame`, `use_dataset`, and `drop_motion_noisy` do
not change the released single-GPU denoiser, so they are not exposed as misleading
controls. The four-step Euler sampler is fixed rather than offering unsupported
solver/step combinations.

## Reproducibility and implementation notes

See [ADAPTATION_NOTES.md](ADAPTATION_NOTES.md) for pinned revisions, exact streaming
changes, cache preservation, test evidence and remaining GPU acceptance work.

```bash
PYTHONPATH=. CUDA_VISIBLE_DEVICES='' \
  /opt/dlami/nvme/.cache_uv/liveavatar-stage1/bin/python -m pytest tests -q
/opt/dlami/nvme/.cache_uv/liveavatar-stage1/bin/python \
  -m reactor_runtime.schema --path . --out schema.json
```
