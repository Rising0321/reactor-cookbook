# Matrix-Game-3.5 example

Serve the public Matrix-Game-3.5 distilled first-person model through Reactor
Runtime. Matrix is conditioned on an anchor image, a text prompt, and camera
trajectories rather than keyboard tokens. This adapter exposes the image and
prompt through Runtime commands and expands normalized six-axis camera motion
into the camera-to-world matrices consumed by the model.

## Runtime boundary

The 5B model is loaded once in a persistent worker process. The worker is
separate because Matrix and Reactor both expose a top-level Python package named
`examples`; process isolation lets each repository retain its public imports.

One worker rollout remains alive for the Reactor session. Every request supplies
one native three-latent causal chunk, equivalent to 12 RGB camera slots, and
receives 12 decoded frames. The rollout preserves its rolling KV cache, absolute
RoPE/PRoPE timeline, generated dynamic visual context, and FrustumHandler Patch
Memory across requests. `context_chunks: 7` remains the rolling KV window; it
does not force seven chunks into one interactive request.

The anchor image initializes Matrix's causal visual state, so `set_image` starts
a fresh rollout at chunk 1 without reloading model weights. Text conditioning is
sampled per causal chunk. `set_prompt` re-encodes only the text context for the
next chunk while retaining the current camera pose, rolling KV cache, dynamic
visual context, and Patch Memory. As the rolling window advances, chunks made
under the new prompt naturally replace older cached chunks.

The adapter emits each finished chunk with single-frame backpressure. WebRTC
playout and session recording consume the same complete 16 FPS sequence, while
camera axes are sampled again before the next expensive chunk begins.

## Set up the repository

From the cookbook repository, enter this example directory and choose one
location for the pinned public Matrix source. This is the only local path the
adapter needs; the environment variable overrides the checked-in YAML default:

```sh
cd examples/matrix-game-3-5
MATRIX_ADAPTER_ROOT=$PWD
export MATRIX_GAME_3_5_PATH=/path/to/Matrix-Game-3.5
git clone https://github.com/Riemann-Dynamics/Matrix-Game-3.5 \
  "$MATRIX_GAME_3_5_PATH"
git -C "$MATRIX_GAME_3_5_PATH" checkout \
  fa6d2b628ac9b0f1657dc24689536d74bfeb51da
git -C "$MATRIX_GAME_3_5_PATH" apply \
  "$MATRIX_ADAPTER_ROOT/stateful_rollout.patch"
```

The adapter verifies the immutable upstream revision before it starts the model.
The local patch adds only the resumable chunk boundary used by the worker; the
model forward pass, scheduler, cache policy, memory queries, and registration
logic remain in the upstream rollout. Alternatively, edit only `source.path` in
`matrix_game_3_5.yaml`; worker, model, tokenizer, inference, and sample paths are
all derived from that root.

Create the Matrix worker environment with the upstream Python version. Reactor
Runtime uses a separate Python 3.12 serving environment; only the isolated model
worker uses Python 3.10:

```sh
uv venv --python 3.10 "$MATRIX_GAME_3_5_PATH/.venv"
uv pip install --python "$MATRIX_GAME_3_5_PATH/.venv/bin/python" \
  torch==2.10.0 torchvision==0.25.0 \
  --index-url https://download.pytorch.org/whl/cu128
uv pip install --python "$MATRIX_GAME_3_5_PATH/.venv/bin/python" \
  -r "$MATRIX_GAME_3_5_PATH/requirements.txt" ftfy==6.3.1
```

## Model assets

No separate checkpoint command is required. On first startup, the adapter uses
the isolated Matrix environment to download these public snapshots at the
immutable revisions recorded in `matrix_game_3_5.yaml`:

- Matrix-Game-3.5 distilled first-person checkpoint
- Wan2.2 TI2V 5B model and UMT5 tokenizer
- Depth-Anything-3 model

Downloads resume through the Hugging Face cache and are skipped once every
required file is present. Allow roughly 48 GB beneath
`$MATRIX_GAME_3_5_PATH/checkpoints`. `HF_TOKEN` and `hf auth login` are honored
if the Hub requests authentication.

The default first-person image, prompt, intrinsics, and camera pose are tracked
by the pinned Matrix source commit, so there is no separate sample dataset to
download. Missing default sample files are restored from that checkout at
startup. They are session fallbacks rather than a restriction: a client can
select its own anchor and prompt at runtime.

## Run

Matrix requires Linux, CUDA, and approximately 40 GB of VRAM at 704x1280. From
the adapter directory, select one GPU. `uv` creates an isolated Python 3.12
serving environment from this example's requirements, while the configured
worker launches from the Matrix Python 3.10 environment prepared above:

```sh
CUDA_VISIBLE_DEVICES=0 uv run --no-project --python 3.12 \
  --with-requirements requirements.txt \
  python -m reactor_runtime.serve
```

Startup waits for the Matrix worker to load the weights. Once it is ready:

```sh
curl -s localhost:8080/health
curl -s localhost:8080/schema
curl -s -X POST localhost:8080/start_session \
  -H 'content-type: application/json' -d '{}'
```

A WebRTC client consumes the `main_video` track at 16 FPS. Recording is enabled
for that video track and requires `ffmpeg` on `PATH`.

## Controls

- `set_image` accepts uploaded JPEG, PNG, WebP, or BMP bytes plus an optional
  prompt, then starts a fresh rollout from that image.
- `set_prompt` applies a new non-empty text condition at the next chunk boundary
  without resetting visual history. `PromptQueued.applies_to_chunk` identifies
  that boundary.

The camera API matches AlayaWorld's normalized six-axis convention. Every value
is in `[-1, 1]`, where zero is neutral:

- `set_forward`: backward to forward translation
- `set_strafe`: left to right translation
- `set_vertical`: down to up translation
- `set_pitch`: down to up pitch
- `set_yaw`: left to right yaw
- `set_roll`: counterclockwise to clockwise roll

The frontend owns device mapping. A WASD frontend sends `set_forward(1)` for W,
`set_forward(-1)` for S, `set_strafe(-1)` for A, and `set_strafe(1)` for D. On
key release it recomputes that axis from the currently held keys and sends the
new value. Pointer, arrow, touch, and gamepad input can drive yaw and pitch;
vertical and roll can use additional keys or analog controls. Simultaneous
translation and rotation axes are normalized independently.

Every six-axis command returns `CameraMotionChanged` with the complete current
motion state and the one-based `applies_to_chunk` boundary. `set_paused` and
`reset` return the same message after releasing all axes, so clients can record
and reconcile every action-state transition outside the inference loop.

Additional commands:

- `set_paused` stops before another expensive chunk starts and holds playback.
- `step` generates and plays one complete 12-frame chunk while paused.
- `reset` restores the selected anchor and prompt; an optional non-negative `seed`
  selects the next reproducible rollout.

Camera axes are sampled at a chunk boundary and apply to the next 12 camera slots,
or 0.75 seconds at 16 FPS. Commands received during inference or playback affect
the following chunk. An in-flight CUDA chunk cannot be interrupted; reset and
disconnect take effect when inference returns to Runtime.

`stream.max_chunks` bounds the preallocated PRoPE camera timeline. The default
512 chunks cover 6.4 minutes. Reset starts a fresh timeline, releases the prior
KV and memory state, and preserves the loaded model weights.

## Image uploads

The `set_image` command declares an `UploadedFile` parameter in Runtime's
schema. A Reactor client reserves a session upload slot, writes the raw bytes to
its returned URL, and sends the resulting upload reference with the command. A
schema-driven frontend can therefore render a file picker without embedding
binary data in a command message.

A ready-to-upload copy of the public first-person demo input lives in
[`example_image`](example_image).

Uploads are limited to 25 MiB and 100 million pixels. Runtime verifies the
declared media type, actual JPEG/PNG/WebP/BMP codec, dimensions, and decodability
before a rollout reset. An empty `set_image` prompt keeps the current prompt;
an empty `set_prompt` is rejected because Matrix requires text conditioning.
