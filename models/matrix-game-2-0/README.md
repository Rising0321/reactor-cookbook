# Play Matrix-Game-2.0 through Reactor Runtime

Run SkyworkAI's public
[Matrix-Game-2.0](https://github.com/SkyworkAI/Matrix-Game) universal distilled
model as an interactive Reactor backend. Use this recipe when a client needs to
start from a public demo or uploaded image, apply native keyboard and
mouse-camera controls, stream autoregressive video, and record the session.

The adapter and upstream implementation remain separate. This directory owns
only the Reactor integration. On first load it fetches the exact tested source
revision and checkpoint into Reactor's mounted weights cache without editing the
upstream checkout. Reactor builds the model image from the `build:` block in
`reactor.yaml`; this recipe has no Dockerfile.

## Prerequisites

- The `reactor` CLI and Docker, configured as described in the
  [Reactor build documentation](https://deploy-docs.reactor.inc/platform/build).
- An NVIDIA GPU, NVIDIA driver, and NVIDIA Container Toolkit. CPU inference is
  intentionally unsupported. The published resource request targets one B200.
- About 12 GB in Reactor's weights cache, plus approximately 30 GB of Docker
  image and build working space.

When the system disk is small, place both Reactor's weights cache and the Docker
daemon's data root on a high-capacity volume. `runtime.weights_path` controls the
host directory mounted for source and checkpoints; Docker image and BuildKit
storage are configured on the Docker daemon itself.

## Run

This directory is a `reactor` workspace. `reactor.yaml` names the model,
declares its generated image build, mounts the persistent weights cache, and
configures recording. `requirements.txt` contains inference dependencies only;
`build.runtime_version` is the single Reactor Runtime version pin.

Check the host, build the image, and expose one free GPU to the container:

```sh
cd models/matrix-game-2-0
reactor doctor
reactor build
reactor run --gpus device=0 --port 8080
```

The CLI renders the YAML build definition into a Dockerfile in memory and hands
it to BuildKit. It installs Reactor Runtime 3.1.2, Python 3.12, CUDA 12.8, the
model dependencies, and the required system packages without writing a
Dockerfile into the workspace. `reactor run` reuses the local image, building it
automatically when its tag is absent.

On first load the adapter sparsely clones the pinned `Matrix-Game-2` source and
downloads only the universal distilled checkpoint, Wan 2.1 VAE, image encoder,
and tokenizer files from the pinned Hugging Face snapshot. Interrupted downloads
resume from the mounted cache, and later runs reuse the same assets.

Check readiness and inspect the generated command schema with:

```sh
curl -s http://localhost:8080/health
curl -s http://localhost:8080/schema
```

Rebuild after editing files copied into the model image:

```sh
reactor build && reactor run --gpus device=0 --port 8080
```

## Controls

A session starts paused with no scene selected. Selecting an image automatically
generates the first chunk even while paused.

- `set_key_state(key, pressed)` holds or releases `w`, `a`, `s`, or `d` for
  forthcoming chunks. Held keys persist and can be combined; W+A and W+D become
  the same multi-hot actions used by the official universal model.
- `set_pitch(pitch)` holds normalized look-down or look-up velocity in
  `[-1, 1]`.
- `set_yaw(yaw)` holds normalized turn-left or turn-right velocity in
  `[-1, 1]`.
- `release_controls` returns keyboard and camera conditions to neutral.
- `set_paused(false)` begins continuous generation one native chunk at a time;
  `set_paused(true)` stops before the next chunk.
- `step` generates exactly one chunk while paused.
- `reset(seed)` clears every autoregressive cache, rebuilds the selected image
  conditioning, and automatically generates a fresh first chunk. Pass `-1` to
  keep the active seed.

Keyboard and camera values are sampled together at the next chunk boundary.
Changing a control while one chunk is in flight applies it to the following
chunk. Pausing or ending the session releases all controls.

## Start from a public demo

`random_image` selects one of the public universal example images in the pinned
Matrix-Game checkout. It clears the active rollout and all controls, rebuilds
image conditioning, and automatically queues one chunk. Repeated calls choose a
different configured image when possible.

## Start from an image

Upload a JPEG, PNG, WebP, or BMP through Reactor's upload protocol, then pass its
upload reference to `set_image`. The adapter applies EXIF orientation, validates
the decoded image, and uses the official centered aspect-ratio crop and 352x640
resize before creating the visual condition.

Uploads are limited to 25 MiB and 100 million decoded pixels. Uploaded bytes and
their rollout state are session-scoped and released when the session ends.

## Autoregressive inference

The adapter loads the official model, image VAE/CLIP encoder, causal VAE decoder,
and universal distilled checkpoint in the Runtime process. Each `step` advances
the same native three-latent block as upstream `inference_streaming.py`; it does
not rerun an offline rollout or rebuild history from rendered frames.

The active rollout preserves the diffusion model's 30 rolling KV caches, the
keyboard and mouse action KV caches, the visual cross-attention cache, and all
32 causal VAE decoder cache tensors. The official `local_attn_size: 6`,
three-step distilled denoising schedule, context-cache commit, and 360-latent
horizon remain unchanged.

The first causal decode emits 9 RGB frames and each later decode emits 12 at 25
FPS. One rollout therefore provides 120 interactive chunks before the adapter
pauses and requires `reset`, `set_image`, or `random_image`.

## Model messages

Commands and generation publish typed messages for the client timeline:

- `action_changed` contains the addressed key, whether it is held, the complete
  held-key set, and the first chunk that will sample it.
- `camera_motion_changed` contains both camera axes and the first chunk that will
  sample them.
- `state_update` is a complete snapshot of image selection, pause and queue
  state, rollout progress, seed, and active controls. A joining viewer receives
  one immediately; successful state changes broadcast another.
- `chunk_complete` contains the chunk index, decoded frame count, sampled
  controls, and measured inference time.
- `rollout_limit_reached` reports the exhausted official chunk horizon.

Message delivery stays outside the synchronous inference path.

## Recording

`reactor.yaml` records `main_video` as H.264 in four-second chunks and allows
clips up to five minutes. The generated image includes FFmpeg. Matrix-Game-2.0
does not emit audio.

## Notes

- `matrix_game_2.yaml` pins the upstream source revision, checkpoint snapshot,
  universal distilled variant, 360-latent horizon, seed, and public demo images.
- Set `MATRIX_GAME_2_PATH` to an existing `Matrix-Game-2` source directory to
  reuse a checkout. Its Git revision must match the pinned revision.
- Matrix-Game-2.0 removed the text-conditioning branch from its released 2.0
  inference model, so this adapter does not expose a prompt command the
  checkpoint cannot consume.
- The image uses Matrix-Game-2.0's native Flash Attention 2.8.3 dependency. This
  recipe deliberately does not add Flash Attention 4, CUDA Graphs, or warmup;
  those are separate inference-optimization work.
- Stop `reactor run` to remove the container and release its GPU memory.
