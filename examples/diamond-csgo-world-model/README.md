# Play DIAMOND CSGO through Reactor Runtime

Run the public [DIAMOND CSGO world model](https://github.com/eloialonso/diamond/tree/csgo)
as an interactive Reactor backend. Use this recipe when a client needs to start
from an official spawn or an uploaded CSGO frame, apply native keyboard and
mouse actions, stream generated video, and record the session.

The adapter and upstream implementation remain separate. This directory owns
only the Reactor integration; `DIAMOND_PATH` points to an unmodified DIAMOND
checkout.

## Prerequisites

- [uv](https://docs.astral.sh/uv/getting-started/installation/) and Git.
- An Apple Silicon Mac with MPS, an NVIDIA GPU with CUDA, or a CPU. CPU inference
  is supported but slow.
- About 1.4 GB of free cache space for the pinned checkpoint and spawn bundle.

Clone the upstream source and pin the revision this adapter was tested against:

```sh
git clone https://github.com/eloialonso/diamond.git /path/to/diamond
git -C /path/to/diamond checkout 851cefb497733d27f1b85c804104638765860fca
export DIAMOND_PATH=/path/to/diamond
```

`DIAMOND_PATH` must resolve to the repository root. The adapter validates its
CSGO source and configuration files. It automatically downloads the model
checkpoint, model configuration, and official spawn data from Hugging Face on
the first run, then reuses the local cache.

## Run

From this example directory:

```sh
cd examples/diamond-csgo-world-model
PYTORCH_ENABLE_MPS_FALLBACK=1 uv run --python 3.12 \
  --with-requirements requirements.txt \
  python -m reactor_runtime.serve
```

`uv` installs Python 3.12, Reactor Runtime 3.1.2, and the model dependencies in
an isolated environment. `PYTORCH_ENABLE_MPS_FALLBACK=1` is required on Apple
Silicon and may be omitted on CUDA or CPU machines. The backend listens on
`0.0.0.0:8080`.

This is a backend service rather than a bundled web application. Open the
[Reactor Sandbox](https://reactor-sandbox.vercel.app), enter
`http://localhost:8080`, and start a session. You can also verify readiness
directly:

```sh
curl http://localhost:8080/health
```

## Adapter layout

`diamond_adapter/pipeline.py` owns model loading, lifecycle hooks, commands, and
inference. `diamond_adapter/types.py` defines the Reactor contract, while
`diamond_adapter/support.py` contains configuration, import, image, and tensor
helpers. The package remains independent from the upstream checkout.

## Controls

- `set_spawn_image(image)` starts from an uploaded CSGO image. The adapter
  center-crops it to the native aspect ratio, builds the four conditioning
  frames DIAMOND requires, uses neutral action history, and selects human input.
- `random_scene` starts from a random official DIAMOND spawn, including its
  recorded action trajectory for replay mode.
- `set_key_state(key, pressed)` holds or releases a native CSGO key.
- `set_mouse_button_state(button, pressed)` holds or releases fire or scope.
- `mouse_move(delta_x, delta_y)` supplies native relative movement for one step.
- `set_controller(controller)` selects human input or the recorded spawn replay.
- `set_paused(paused)` stops model inference; `step` advances once while paused.
- `reset` starts from another spawn state.

Control events return an `ActionChanged` message containing the acknowledged
controller, current held keys and mouse buttons, and any mouse delta accepted by
that command. Message delivery therefore stays outside the synchronous
inference loop.

## Start from an image

Ready-to-upload CSGO frames live in [`example_images`](example_images). Upload
one through the client, then pass its upload reference to `set_spawn_image`:

```js
const image = await uploadFile(file);
await sendCommand("set_spawn_image", { image });
```

The next inference boundary resets the environment and emits the uploaded frame
before consuming the first action. This frame is also emitted while paused,
without running an expensive model step. A single image has no motion history,
so the adapter repeats it four times; arbitrary non-CSGO images are accepted but
may produce unstable results because they are outside the model's training
distribution.

Use `random_scene` when the client should choose a dataset-backed initial
condition instead. Both scene commands return `SceneChanged` with the selected
source and filename or dataset scene identifier.

## Recording

`reactor.yaml` records `main_video` by default in four-second chunks and allows
clips up to five minutes.

## Test the adapter contract

The focused tests use a fake world, so they verify lifecycle, first-frame, and
schema behavior without downloading the checkpoint:

```sh
uv run --python 3.12 --with 'reactor-runtime==3.1.2' \
  --with pytest --with numpy pytest -q tests
```

## Notes

- `human` applies client input. `replay` follows the selected spawn's recorded
  action trajectory; DIAMOND's CSGO checkpoint does not contain a learned
  player policy.
- Keyboard keys and mouse buttons are held until released. `mouse_move`
  contains relative deltas that are consumed by exactly one inference step.
- The frontend owns keyboard, pointer, touch, and gamepad mappings. The backend
  exposes DIAMOND's native action semantics without prescribing a control
  layout or sensitivity.
- Checkpoint loading and GPU allocation happen once during `load()`. Each
  Reactor session owns one shared world: temporary client disconnects preserve
  it, while session end discards queued scenes and control state.
