# LiveAvatar acceleration — 2026-09-16

## What is enabled

- Official `causal_s2v_pipeline_tpp.py`: four denoising-stage ranks plus one
  streaming-VAE rank on five B200s; the remaining three GPUs are unnecessary for
  this tested layout. Do not use the experimental blockwise interactive file,
  which substitutes random audio during initial blocks.
- FlashAttention 4 (`flash-attn-4==4.0.0b30`, tested with CUTLASS DSL 4.7.1), dense
  BF16 attention, preserving scale, valid key lengths, causal/window semantics.
  Non-B=1 or dropout calls retain the original supported path; no sparse route,
  FP8 quantization or shortened KV history is enabled.
- GPU RGB8 conversion before host transfer. Four native 12-frame blocks are
  delivered as one existing 48-frame SDK clip (first clip 45 frames). A one-clip
  acknowledgement prevents unbounded output accumulation; native pipeline stages
  can have their normal bounded in-flight blocks.
- Model assets, IPC files and caches remain under `/opt/dlami/nvme`.
- Native 25 FPS video and actually resampled mono 48 kHz audio remain unchanged.

Official FA4 installation/API reference:
https://github.com/Dao-AILab/flash-attention/blob/main/flash_attn/cute/README.md

## Tutorial ideas evaluated

Read `sol_super_accel_1_1_new/AI_INFRA_TUTORIAL_ZH.html` and its Markdown companion.
Useful transfers were stage parallelism, resident weights, measured rather than
assumed kernel benefits, streaming decode, and GPU-side RGB8 conversion. The
upstream already merges its LoRA at load time. H3's AdaLN tables, partial RoPE,
SwiGLU and spatial VAE tile splitting are not drop-in replacements for LiveAvatar.
Sparse SOL/BSA and compressed communication actively change numerical semantics
and were not enabled for this memory-sensitive adaptation.

An explicit-cache CUDA Graph experiment on the streaming VAE was bit-identical
in five tested decoder calls, but steady timing changed from about 0.150 to 0.154
seconds per 12 frames. It is available only in the benchmark experiment and is
not enabled in the service. No blanket `torch.compile` is enabled.

## Measurements (same 384×704 input canvas, 25 FPS)

Explicit official fashion-blogger image/audio, seed 420, four denoising steps.
`benchmark_acceleration.py` measures the single-GPU path including VAE and host
RGB8 delivery. `benchmark_tpp.py` measures the official TPP path including RGB8
delivery but not final file encoding. Loading weights is separate from take time.

| Path | Observed timing | Playback comparison |
| --- | --- | --- |
| Single GPU, baseline | 12.068–12.082 s per steady 48-frame clip | ~4 FPS, 0.159× realtime |
| Single GPU, FA4 | 11.915–11.960 s per steady 48-frame clip | ~4 FPS, 0.161× realtime |
| Five GPUs, native TPP + FA4, first take | 16.982 s for 285 frames / 11.4 s content | 0.671× including cold preparation |
| Five GPUs, native TPP + FA4, repeated take | 9.198 s for 285 frames / 11.4 s content | 30.98 FPS, 1.239× including input preparation |
| Five GPUs, native TPP + FA4, steady blocks after first two clips | 49.96–50.71 FPS | ~2× realtime |

The slowest steady block in the two runs was 0.2424 seconds for 12 frames
(playback budget 0.48 seconds). Native video first-frame latency and cold setup
are distinct from sustained throughput: **the first cold take is not faster than
playback end-to-end**. Arbitrary resolutions, inputs, or GPU contention are not
covered by this measurement; startup checks are not an exclusive GPU reservation.

A representative B=1, Q=792, K=6000, H=40, D=128 attention test measured FA2
0.312 ms versus FA4 0.158 ms, with relative RMS difference ~0.00302. Most of the
end-to-end gain comes from native temporal pipeline parallelism and streaming
VAE, not from replacing a single attention kernel.

## Fidelity and operational boundaries

The TPP callback patch only exposes each decoded block and avoids accumulating
all frames. The upstream TPP cache allocation, circular history, four denoising
steps, reference refresh and streaming decoder state remain intact. Reset
clears per-take streaming VAE state, including its first-decode flag.

This is the released **multi-GPU** algorithm, not a bit-identical rearrangement of
the prior single-GPU deferred-decode path: TPP refreshes the reference anchor
after its first clip and uses the upstream streaming VAE. Its frames must be
judged against that native route; do not claim pixel parity with the old route.
The 11.4-second temporal contact sheet inspected for the repeated take showed
stable identity/background, changing speech mouth shapes and plausible gestures;
this is not a formal lip-sync or long-horizon quality metric.

Artifacts are in `/opt/dlami/nvme/ruixing/liveavatar-test-kit-20260916/verification/`:
`accel-baseline`, `accel-fa4`, and `accel-tpp-fa4`. Historical measurements are not
overwritten. Browser and SDK commands remain described in SANDBOX.md.

## Native Runtime / SDK acceptance

Two explicit-input tests through Runtime 3.2.5 and the SDK (fashion blogger and
kitchen grandmother) each delivered all 285 frames across six clips, reported
`generation_ended: complete`, and produced received PCM plus video artifacts.
The second person's inspected temporal sheet retained a coherent kitchen,
identity and plausible speech gestures. No formal lip-sync metric is claimed.

In the actual received fashion-blogger PCM, five 0.3-second speech templates at
0.75/2.25/4/8/10 seconds matched with correlations 0.991/0.960/0.996/0.998/0.992.
The received format was mono 48000 Hz, peak 0.696, with no sample-rate speedup.
Transport includes startup/underrun silence; offline `take.mp4` still uses the
original uploaded audio and is not used to validate this PCM result.

The worker's per-clip `build_seconds` excludes demand-acknowledgement waiting,
and other native stages may progress while it waits. Its occasional 3× figure
is therefore NOT used as sustained throughput evidence; use the unpaced native
benchmark above (~2×) for that conclusion. Runtime playback itself is paced at
25 FPS by design, not played at 50 FPS.

The service ran on GPU 0–4. After tests, GPU 5/6 showed unrelated compute PIDs;
our Runtime and its five workers were stopped and their GPU allocations were
confirmed released. No unrelated process was signalled. No GPU service is left
running by this acceleration test. Restart with `start_parallel.sh` when the
chosen devices are available.
