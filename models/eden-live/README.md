# EdenLive on Reactor

Play EdenLive's code-driven games through [Reactor](https://reactor.inc). Choose a game,
use its original reference or upload your own, and steer with keyboard or on-screen controls.
Every completed chunk shows the matching native draft on the left and EdenLive's render on
the right, in one synchronized video stream.

![EdenLive running with synchronized draft and generated views](assets/demo.png)

This recipe requires authorized access to EdenLive's source and weights. It does not
download private source during startup, publish the model, or call an external generation API.

## What you need

- Linux, one NVIDIA B200, Docker and the NVIDIA Container Toolkit.
- About 37 GB for the 5B weights, plus space for the image and source.
- The authorized EdenLive source bundle and access to `VAST-AI/EdenLive`.
- Node.js 20+ for the demo and the original games' JavaScript dependencies.

## 1. Prepare the source and weights

Install the current [Reactor CLI](https://docs.reactor.inc/deploy) and the Hugging Face CLI.
Prepare a directory on a disk with sufficient space:

```sh
export EDEN_BUNDLE=/path/to/edenlive-bundle
mkdir -p "$EDEN_BUNDLE"
cp -a /path/to/EdenLive-Official-master "$EDEN_BUNDLE/source"
npm --prefix "$EDEN_BUNDLE/source/realtime/games" install --ignore-scripts --no-package-lock
hf auth login
hf download VAST-AI/EdenLive --revision fc9ad921b7fab8fa6a82fe66071fe6d3bd55b429 \
  --include 'dit-5b/*' 'vae-5b/*' 'decoder-5b/*' 'umt5/*' 'vision-encoder/*' 'scheduler-5b/*' \
  --local-dir "$EDEN_BUNDLE/weights/EdenLive"
```

The source bundle must contain `realtime/flashdreams`, `realtime/integrations_v2`,
and `realtime/games`. Its model implementations and game assets are used directly.
Keep credentials out of source files and Docker build arguments.

## 2. Start the model

From this directory:

```sh
reactor build
reactor run --gpus device=0 --shm-size 2g --weights "$EDEN_BUNDLE" --port 8080
```

Choose a free GPU on a shared machine. The image is built from `reactor.yaml` and runs
Reactor Runtime 3.8.3. There is no separate Dockerfile. The model stays on one GPU;
the original game engine runs in headless Chromium on the CPU.
Use a CLI version that supports `--shm-size` (tested with v1.20261014.332): Docker's
64 MiB default is too small for the full-resolution chunk exchange.

**Check:** `curl http://localhost:8080/health` reports `"state":"available"` after loading.
Leave the terminal running; `Ctrl+C` stops the service. Loading and the first chunk take
longer than subsequent chunks. The default B200 profile enables dense FlashAttention 4,
PyTorch compilation and the author's CUDA graph dispatch. Initial compilation can take
several minutes; the steady-state numbers below exclude this startup cost.

To use the native eager profile, set `attention_backend: native`, `cuda_graph: false`
and `compile_model: false` in `edenlive.yaml`, then rebuild and restart. Both profiles
keep the same weights, four sampling steps, precision policy and history length.

## 3. Open the demo

```sh
cd demo
npm ci
VITE_REACTOR_URL=http://localhost:8080 npm run dev
```

Open the URL printed by Vite. Click **Connect**, choose an **Official game**, optionally
upload an **Appearance reference**, and press **Start**. Selecting or uploading alone
does not generate video. Match a custom reference to the game's opening composition.

For a remote GPU server, the browser must reach both HTTP and WebRTC. SSH forwarding
of the HTTP port alone does not carry video; host networking alone does not solve this.
Use a TURN-over-TCP relay and forward its TCP listener along with the frontend and HTTP
ports. Keep machine-specific relay credentials and network overrides outside this recipe.

## Controls and playback

| Control | Effect |
| --- | --- |
| Official game | Stage one of seven original games with its matching cover and prompt. Pause before changing it. |
| Appearance reference | Stage a PNG, JPEG or WebP, up to 25 MiB / 16 megapixels. The image is resized and center-cropped to 1280 × 704 at Start. |
| Start / Start new world | Start the selected game and a fresh video with the chosen seed. |
| Keyboard / on-screen keys | Hold to move or act; release to stop that control. Available actions vary by game and are listed in the demo. |
| Update prompt | Change upcoming appearance text while keeping the same world. Native game-event descriptions are added automatically. |
| Pause / Resume | Hold both simulation and generation, then continue the same world. A prompt change while paused stays paused. |

The first chunk contains 9 frames; later chunks contain 12. Each completed chunk is
returned immediately. The native simulation advances at 60 Hz and is sampled at 24 FPS;
the combined 2560 × 704 output plays at 24 FPS. Each left/right pair uses the same
simulation frame, so neither view runs ahead of the other. When generation takes longer
than playback, both views wait together.

The demo polls held keys every 33 ms, with one command in flight and one replaceable
latest state. It does not replay old key transitions. The native renderer samples
the latest accepted state before each draft frame, preparing at most one chunk while
the GPU generates the preceding chunk. Pause freezes capture at a frame boundary;
Resume keeps any partially captured chunk and the same model history. Hold an action
long enough to be sampled: a press and release between samples is intentionally
coalesced. Already captured, generated or buffered frames retain their original input.
Runtime accepts commands between inference steps, so 33 ms polling is not a claim
of 33 ms input-to-video latency.

Prompt edits preserve video history and may change appearance gradually. They do not
rewrite the game geometry, controls, or objectives. A terminal game pauses generation;
press **Start new world** to begin again.

## Validation and current limits

Tested on one B200 with all seven games, a 13-chunk interactive rollout, ten keyboard
actions, live prompt edits, pause/resume and reconnect through TURN/TCP. A controlled
fixed-kernel comparison of the native pipeline and adapter model matched all 141 raw
RGB frames across 12 consecutive chunks. The native autotuned default has run-to-run
variation; that profile is not claimed to be bit-exact.

In a separate two-round, 32-chunk-per-round model benchmark, the default B200 profile
measured about **57.6 FPS** (0.208 s per 12-frame steady-state chunk), versus **25.8 FPS**
for the native eager profile. These timings include DINO conditioning, generation,
cache finalization, decoding and RGB transfer; they exclude game capture, transport,
warmup and prompt re-encoding. Playback remains **24 FPS**.
FA4 changes floating-point reduction order; the accelerated profile is not claimed to
produce bit-identical video. The raw-frame parity check above applies to the native
fixed-kernel profile.

Capture and inference use a bounded pipeline; both views display the same simulation frames after inference completes.
End-to-end responsiveness includes capture, inference, transport and playback buffering.
Custom-reference quality and arbitrary long sessions still
need broader evaluation; the integration does not invoke the optional coding agent.

## Development

`edenlive_app.py` owns commands and simulation; `edenlive_types.py` defines the client
contract. `edenlive_model.py` and `edenlive_backend.py` own the original model pipeline
and its streaming state. The application entrypoint uses an `_app` suffix to avoid
shadowing the upstream `edenlive` package.

This recipe uses the author's 5B LightTAE real-time pipeline with four sampling steps
and its original history and cache update rules. `edenlive_attention.py` supplies the
single-GPU dense FA4 implementation; compilation preserves explicit precision casts,
and graph capture follows the author's cache-aware dispatch. Multi-GPU serving is not
implemented.

CPU tests require Runtime, Pillow, pytest and pytest-asyncio; no weights, upstream checkout
or GPU are needed:

```sh
PYTHONPATH=. python -m pytest tests/ -q --asyncio-mode=auto
python -m reactor_runtime.schema --path .
reactor validate
```

## Credits

EdenLive's model, game engine and assets are provided by its authors. Thanks to
[Rising0321](https://github.com/Rising0321) for porting EdenLive to Reactor and building
the synchronized interactive demo. Source and checkpoint access remain subject to
the authors' permissions.
