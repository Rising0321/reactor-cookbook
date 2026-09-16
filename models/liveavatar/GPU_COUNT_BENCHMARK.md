# LiveAvatar: minimum-GPU real-time benchmark

## Result — 2026-09-16

**Four B200s are the lowest tested configuration that clearly meets both warm
full-take and sustained real-time throughput.** Three GPUs are borderline, not
a reliable real-time recommendation. This is a result for this implementation
and input size, not a hardware lower bound against future optimizations.

| GPUs | Steady FPS, cold / warm take | Warm full-take time | Result at 25 FPS |
| --- | --- | --- | --- |
| 1 | 10.72 / 10.85 | 29.63 s for 11.40 s video | Below real time |
| 2 | 12.50 / 12.69 | 25.63 s for 11.40 s video | Below real time |
| 3 | 24.98 / 25.32 | 48.82 s for 45.96 s video | Borderline steady, full take below real time |
| 4 | 30.30 / 30.80 | 40.82 s for 45.96 s video | Pass after warmup |
| 5 (earlier native benchmark) | 49.94 / 50.69 | 9.20 s for 11.40 s video | Pass after warmup |

The four-GPU warm take achieved 28.15 FPS including input preparation, with
23% steady throughput headroom. Across its two long runs, none of 176 measured
steady blocks exceeded the 0.48-second playback budget; the slowest was 0.398 s.
Three GPUs exceeded that budget on 35/88 and 1/88 steady blocks respectively.
With startup buffering, three GPUs can approach continuous playback, but have
essentially no margin for contention or jitter.

**Cold start remains separate:** the four-GPU first take required 50.57 seconds
for 45.96 seconds of content, excluding weight loading. This benchmark does not
promise immediate real-time output on a newly started service, other resolutions,
arbitrary inputs, or shared/busy GPUs. The five-GPU row reuses the earlier shorter
benchmark and is not a new long-run result.

Corrected packed-stage results are in `grouped-1gpu-rng`, `grouped-2gpu-rng`,
`grouped-3gpu-long-rng` and `grouped-4gpu-shared`. Machine-readable measurements
and comparisons are in `gpu-count-summary.json` in the verification directory.
Decoded MP4 frames were identical between one and two GPUs (all 285 frames),
and between three and four GPUs (all 1149 frames): mean and maximum frame MSE
were both zero. This checks final delivered pixels, not intermediate tensor
bitwise equivalence. Four-GPU temporal samples retained a coherent person and
background across 46 seconds; no formal lip-sync metric is claimed.

All benchmark GPU processes exited and their allocations were confirmed released.
No serving process was started or switched. CPU regression: 36 tests passed;
the guard's own test verifies that an unrelated process is left running.

## Scope

These are unpaced **inference benchmarks**, not a change to the production
Runtime launcher. `start_parallel.sh` still uses the validated five-GPU backend.
All tests use B200, FA4, native BF16, four Euler denoising steps, unchanged native
KV cache allocations, a 384×704 canvas and 25 FPS. The grouped method is compiled
in memory from the pinned upstream method; neither its checkout nor the serving
backend is rewritten.

Physical layouts:

| GPUs | Layout |
| --- | --- |
| 1 | Four sequential DiT stages and streaming VAE on one GPU |
| 2 | Four sequential DiT stages on GPU 0; VAE on GPU 1 |
| 3 | DiT stages 0/1 on GPU 0, stages 2/3 on GPU 1, VAE on GPU 2 |
| 4 | One DiT stage per GPU, with VAE sharing the last GPU |
| 5 | Released four-DiT-rank plus dedicated-VAE-rank pipeline |

Each denoising stage retains **its own KV cache, cross-attention cache and Python
RNG stream**. The last item matters: upstream samples a conditional RoPE offset
with `random.randint(4, 30)` on each inference call. Packing stages without
isolating those streams changes the sample, even with the same nominal seed.
Early `grouped-2gpu` and `grouped-3gpu` short runs predate that correction and
must not be used as final fidelity evidence. The interrupted `grouped-3gpu-long`
run is also excluded. Corrected packed-stage runs have `-rng` in their paths.
The four-GPU run already had one stage per rank and does not have this issue.

When VAE shares a DiT rank, its generated reference anchor is kept separate
until the next clip. The ongoing first clip still uses the original reference,
matching the native separate-rank execution.

## Measurement

Short inputs produce 285 frames / 11.4 seconds. For long tests, the same explicit
12.8-second audio is repeated four times and generation is capped at 24 clips:
1149 frames / 45.96 seconds. This is a throughput stress input, not additional
distinct speech. No default image is introduced into the serving API.

Steady throughput uses frames after the first two clips divided by elapsed wall
time between block 7 and the final block, including callback overhead. A native
12-frame block has a 0.48-second playback budget. Full-take time includes input
preparation but excludes loading model weights and final MP4 encoding. Each
configuration runs twice in the same loaded process to separate cold and warm
take behavior.

## Reproduction

From this model directory, on otherwise idle GPUs:

```bash
CUDA_VISIBLE_DEVICES=0,1,2,3 LIVEAVATAR_MODE=tpp OMP_NUM_THREADS=4 \
  /opt/dlami/nvme/.cache_uv/liveavatar-stage1/bin/python benchmark_guard.py \
  --log /opt/dlami/nvme/.cache_hf/reactor_registry/liveavatar-stage1/new-four-gpu.log \
  -- /opt/dlami/nvme/.cache_uv/liveavatar-stage1/bin/torchrun \
  --standalone --nproc_per_node=4 benchmark_tpp.py \
  --grouped --shared-vae --fa4 --runs 2 --chunks 24 \
  --image /opt/dlami/nvme/ruixing/liveavatar-test-kit-20260916/inputs/fashion_blogger/image.jpg \
  --audio /opt/dlami/nvme/ruixing/liveavatar-test-kit-20260916/verification/grouped-long-audio.wav \
  --output /opt/dlami/nvme/ruixing/liveavatar-test-kit-20260916/verification/new-four-gpu
```

Use fresh log/output paths to retain historical evidence. For three or two GPUs,
change the visible devices and process count and omit `--shared-vae`. One GPU
automatically shares its decoder. Five GPUs need neither grouped flag.
The guard refuses to start while other compute processes are present, polls
every two seconds, and terminates only its own benchmark if a foreign process
appears or the run approaches ten minutes. This is not an exclusive reservation.

Raw JSON, videos and contact sheets live under
`/opt/dlami/nvme/ruixing/liveavatar-test-kit-20260916/verification/`.
