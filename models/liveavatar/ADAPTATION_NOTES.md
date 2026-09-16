# LiveAvatar stage-one implementation record

## Sources

- Fresh cookbook main: `5d03fbebe808a99fdc960e5046ea6428e27df1a8`.
- LiveAvatar upstream: `c3c47d031d8bf3247428333c1fe610579c71a551`.
- Wan2.2-S2V-14B weights: `dab4e9c55bbe4c8c4d03db1c2c98c7f0ac9c454b`.
- Live-Avatar distilled LoRA: `92cdccd12a91e8a63767a7c821b7c75e51d5a172`.
- Reactor Runtime 3.2.5, Reactor SDK 1.5.1, PyTorch 2.8.0/cu128,
  transformers 4.51.3, diffusers 0.35.1, PEFT 0.17.1, FlashAttention 2.8.3.
- Read and applied `reactor-team/ai-skills/models/polish-reactor-model-schema`:
  documented command timing/preconditions, moderated uploads/text, typed replies,
  and a `StateUpdate.from_state` snapshot. Skill checkout is the sibling
  `reactor-ai-skills-20260916` directory.
- Avatar contract references:
  https://docs.reactor.inc/model-api-reference/ltx/schema.md
  https://docs.reactor.inc/deploy/development/overview.md
  https://docs.reactor.inc/deploy/development/local-testing.md

## Streaming fidelity

The released single-GPU `WanS2V.generate` accumulates latent clips and decodes them
after all denoising. `liveavatar-streaming.patch` adds an optional callback and
moves **only the deferred VAE recurrence** to each completed clip. It keeps that
recurrence's `motion_latents_pp` separate from denoising `motion_latents`. Updating
the latter every clip would change the original model, so it is deliberately not
done. With the callback omitted, the old return path remains available.

One take is one upstream `generate` call. A demand-gated worker thread retains
that call's locals and CUDA/no-grad/autocast contexts throughout the take. It
blocks after each emitted clip. Pause does not rebuild caches; stop allows current
work to finish and then cleans this take's caches. No process-killing calls are
used in model code. This stage emits 48-frame clips, not 12-frame latent blocks:
splitting VAE decode further would require a separate decoder adaptation.

Unmodified upstream behavior:

- Four sampler-step-specific KV caches and shared condition cache.
- Cache length computed from native clip latent shape and image-fitted resolution;
  no shortened cap, cache offload, quantization, or automatic periodic reset.
- 73 motion frames, three latent frames per denoising block, 48 output frames per
  clip, four distilled Euler steps, `seed + clip_index`, 25 FPS.
- Native denoising positions, attention implementation, reference/motion prefill,
  and deferred-decoder motion-history recurrence.
- First clip loses three initial decoded frames; input audio begins at time zero,
  matching upstream's final audio mux behavior.

GPU weights stay on one visible device; upstream still moves the audio encoder
to CPU after extracting audio features. `init_on_cpu=False` and
`offload_model=False` avoid the optional repeated DiT/VAE CPU swapping. No FA4,
compile or CUDA-graph optimization has been added.

## Dependency adaptation

Runtime 3.2.5 requires NumPy >=2.1 while upstream requirements request NumPy <2.
This isolated environment uses NumPy 2.2.6. Upstream module import (with only
CUDA device discovery mocked), image/video decoding, audio loading, and the
surrogate control-flow parity tests pass under that version. Real learned-model
GPU generation also passed in three- and six-clip SDK tests. Decord 0.6.0's installed wheel carries an old
cp36 platform tag, which `uv pip check` warns about; its ctypes video reader was
tested decoding all 141 frames successfully on Python 3.12. This warning is not
hidden or represented as a fully clean dependency audit.

## Storage

- Environment and uv downloads: `/opt/dlami/nvme/.cache_uv`.
- HF snapshots (approximately 47 GiB): `/opt/dlami/nvme/.cache_hf/hub`.
- Runtime uploads/temp/test artifacts:
  `/opt/dlami/nvme/.cache_hf/reactor_registry/liveavatar-stage1`.
- Existing Docker cache `/opt/dlami/nvme/.cache_docker` is untouched; no Docker
  daemon restart or data-root change is needed for stage one.
- No global shell configuration or other model's environment was changed.

## Verified

1. Schema renders with the actual Runtime package, without importing GPU models.
2. Twelve CPU tests pass: contract, missing/default input behavior, input rejection,
   explicit start gating/stop/reset, bounded backpressure, cancellation/errors, one clip per
   turn, upstream-versus-patched generation control-flow parity, preserving the
   Runtime shutdown method, and nonempty reporting of upstream assertions.
3. The parity test executes both versions' actual `generate` function with CPU
   surrogate VAE/DiT modules and the real Euler scheduler. Three clips match
   bit-for-bit; denoiser condition tensors, temporal positions and cache allocation
   arguments match. Both upstream online-decode settings are exercised.
   This is **not** full-weight numerical parity or quality validation.
4. A CPU-only service using the actual LiveAvatar command class and synthetic
   backend was connected by Reactor SDK, received uploaded files and commands,
   and delivered 141 video frames plus audio. Artifacts:
   `/opt/dlami/nvme/.cache_hf/reactor_registry/liveavatar-stage1/cpu-transport/received-final`.
5. The GPU launch guard refuses the currently occupied cards.

## Real GPU validation completed on 2026-09-16

After the user reported GPUs available, GPU 0 was verified free and used alone.
The initial run exposed missing upstream FlashAttention 2 in cross-attention.
Installed the official v2.8.3 cu12/torch2.8/cp312/cxx11abiTRUE wheel; attention
math, KV-cache code, lengths, and denoising logic were not changed.

Full-weight model load and two real SDK tests passed: 141 frames, then 285 frames.
Both used explicit uploads of the upstream `fashion_blogger` image/audio, not a
runtime default. Temporal contact sheets and clip-boundary frame sequences were
visually inspected. Identity, background and clothing remain stable; speech-like
mouth motion and gestures are plausible. No gross boundary cuts or collapse were
observed. This is a qualitative sampled-frame inspection, not a scored phoneme
alignment or full numerical-parity evaluation. See `GPU_TEST_REPORT.md`.

The final test used about 92.5 GiB observed GPU memory, with all other GPUs idle.
`stop` remains the wire command but its Python handler is `stop_take`, so it does
not override Runtime's synchronous shutdown hook. The final service was stopped
after the test to release its GPU.

## Remaining limitations

Audio erratum: the original GPU test additionally exposed a 16/48 kHz playback
mismatch. v0.2.0 resamples the whole playback waveform from 16 to 48 kHz before
chunk slicing, and declares the output track at 48 kHz. The uploaded 16 kHz file
passed to upstream inference and all KV-cache logic are unchanged. CPU SDK
loopback validates this correction; the historical GPU audio is not a passing
audio-quality artifact. See AUDIO_FIX_REPORT.md.

- Real-time speed is not achieved: live audio inserts silence between generated
  clips. `take.mp4` is an offline model-timeline preview using uploaded audio;
  raw received PCM is kept separately. No latency-free streaming claim is made.
- Real pose-conditioned generation, multispeaker/SAM2 and TTS remain unvalidated
  or outside the implemented surface as described in README.
- Real-model numerical parity, long-duration memory studies and formal lip-sync
  metrics have not been run. Default cache mechanics remain upstream unchanged.

No unrelated GPU processes were stopped. The earlier synthetic transport fixture
and its video are not used as learned-model quality evidence.
