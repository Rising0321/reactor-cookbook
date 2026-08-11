# Play OpenDreamer through Reactor Runtime

Run the public [OpenDreamer world model](https://github.com/next-state/open-dreamer)
as an interactive Minecraft Reactor backend. Use this recipe when a client
needs to start from a dataset demo or uploaded Minecraft frame, apply native
keyboard and mouse actions, stream generated video, and record the session.

The adapter and upstream implementation remain separate. This directory owns
only the Reactor integration; `OPENDREAMER_PATH` points to an OpenDreamer
checkout at any filesystem location. The adapter never edits tracked upstream
source files.

## Prerequisites

- [uv](https://docs.astral.sh/uv/getting-started/installation/) and Git.
- An NVIDIA GPU with CUDA. CPU inference is intentionally unsupported.
- About 8 GB of cache space for the checkpoint and 184 MB for the public VPT
  sample.

Clone the upstream source, pin the revision tested by this adapter, and export
its absolute path:

```sh
git clone https://github.com/next-state/open-dreamer.git /path/to/open-dreamer
git -C /path/to/open-dreamer checkout 797e41f052b5996740938fd2fe8161f1866de3a2
export OPENDREAMER_PATH=/path/to/open-dreamer
```

`OPENDREAMER_PATH` must resolve to the repository root containing `dreamer/`.
The adapter validates the required source files and exact Git revision before
importing OpenDreamer.

## Run

From this example directory, select one CUDA device and start the backend:

```sh
cd examples/open-dreamer
CUDA_VISIBLE_DEVICES=0 uv run --python 3.12 \
  --with-requirements requirements.txt \
  python -m reactor_runtime.serve
```

`uv` installs Python 3.12, Reactor Runtime 3.1.1, and the model dependencies in
an isolated environment. The first run automatically downloads the pinned
`reactor-team/open-dreamer` checkpoint and missing demo assets, then compiles
the JAX generation and conditioning paths. The backend listens on
`0.0.0.0:8080`.

This is a backend service rather than a bundled web application. Open the
[Reactor Sandbox](https://reactor-sandbox.vercel.app), enter
`http://localhost:8080`, and start a session. You can also verify readiness
directly:

```sh
curl http://localhost:8080/health
```

## Controls

- `set_key_state(key, pressed)` holds or releases `w`, `a`, `s`, `d`, `space`,
  `shift`, `ctrl`, `e`, `q`, `escape`, `f`, `1`-`9`, or `f3`.
- `set_mouse_button_state(button, pressed)` holds or releases `left`, `right`,
  or `middle`.
- `mouse_move(delta_x, delta_y)` supplies native relative camera movement for
  one generated frame.
- `mouse_wheel(delta)` supplies a `-1`, `0`, or `1` scroll event for one frame.
- `set_demo(demo)` starts from one reproducible dataset window.
- `random_demo` starts from a randomly selected dataset window.
- `set_conditioning_image(image)` starts from one uploaded Minecraft frame.
- `set_paused(paused)` stops model inference.
- `reset(seed)` clears the incremental caches and optionally changes the seed.

Keyboard keys and mouse buttons remain held until released. Mouse movement and
wheel input are consumed after one successful generation step. Pausing and
disconnecting release all controls.

## Start from a dataset demo

The first connection selects one of three dataset demos at random. Each demo is
a 16-frame window with frame-aligned VPT actions from one public OpenAI
recording. `set_demo` selects `demo_1`, `demo_2`, or `demo_3`; `random_demo`
chooses another window randomly. Both reset the model's incremental KV caches.

Missing demo data downloads automatically into
`$OPENDREAMER_PATH/samples/vpt`. Existing files are reused. To populate the
external checkout before starting the backend, run the same downloader
explicitly:

```sh
cd examples/open-dreamer
uv run --python 3.12 --with-requirements requirements.txt \
  python download_demo.py
```

The automatic and explicit paths share one atomic downloader. The paired MP4
and JSONL are internal dataset conditioning assets; clients do not upload them.

## Start from an image

Ready-to-upload dataset frames live in [`example_images`](example_images).
Upload one through the client, then call `set_conditioning_image` with its
upload reference:

```js
const image = await uploadFile(file);
await sendCommand("set_conditioning_image", { image });
```

The adapter center-crops the image to 640x360, pads it to OpenDreamer's 640x368
tokenizer shape, repeats it for the 16-frame context, and pairs each frame with
a neutral VPT action. The next inference boundary resets the caches and starts
automatically from that image.

A single image has no motion history, so its rollout may be less stable than a
dataset demo backed by consecutive frames and aligned actions. Arbitrary images
are accepted, but Minecraft frames from the model's training distribution give
the most reliable results.

## Model messages

Every control command returns a typed, command-correlated message for the
client timeline:

- `ActionChanged` contains the originating control, pause state, held keys and
  mouse buttons, and the mouse or wheel delta received.
- `ConditioningChanged` contains the selected demo or uploaded image filename.
- `RolloutReset` contains the selected seed and retained conditioning source.

Message delivery stays outside the synchronous inference loop.

## Recording

`reactor.yaml` records `main_video` by default in four-second chunks and allows
clips up to five minutes. Recording requires `ffmpeg` on `PATH`. OpenDreamer
does not emit audio.

## Notes

- `opendreamer.yaml` pins the upstream source, checkpoint, sampling schedule,
  and demo offsets. It contains no machine-specific source path.
- The adapter follows the upstream CUDA 12 dependency lock and keeps
  training-only packages out of the runtime environment.
- Selecting the same demo and reset seed reproduces the same rollout. The
  automatic demo selected on a new connection is intentionally random.
- Disconnecting releases client controls but keeps the loaded model available
  for the next connection. Stop the backend process to release its GPU memory.
