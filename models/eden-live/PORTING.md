# EdenLive — port notes

## What the model is

EdenLive 5B (VAST-AI, built on FlashDreams) is a causal draft-to-video model.
Each chunk it reads three draft frames through a DINO encoder, denoises in four
steps against a cache opened from a reference image and a prompt, and decodes
9 frames (first chunk) or 12 with LightTAE. Upstream serves it behind its own
WebRTC stack, driven by code games (`game2v`) or a camera (`cam2v`).

## Where the port started

Ruixing's port (`reactor-team/reactor-edenlive@e11bca2`, imported as 0.1.0)
was already a `ReactorApp` in two halves on runtime 3.8.3:

- `edenlive_model.py` / `edenlive_backend.py` / `edenlive_attention.py`: the
  model half. Upstream's 5B pipeline behind `load` / `generate` / `reset`,
  one `EdenInput` in and one `EdenResult` out, nothing from
  `reactor_runtime`, run in its own process by `DistributedRunner`. A new
  `world_id` opens a new world; the result reports `frames_wanted`.
- `edenlive_app.py` / `edenlive_game.py`: the application half. The game is
  simulated and rendered headless on the server, and keys are sampled once per
  captured frame from a latest-value mailbox.
- Refusals already raised `CommandError`.

## What 1.0.0 changed, and why

**A client video source next to the game.** The model only ever sees three
draft frames per chunk, and the game's draft is its own render at chunk-local
frames 0, 4, 8 (upstream `GameTimeline` samples chunk-locally on purpose). So
any RGB video can drive it. The port gained a `source_video` input track and a
`set_source` command, and the game and the client became two implementations
of one draft-source protocol (`prepare` / `take` / `set_keys` / `set_paused` /
`close`) returning a `Draft`. The step derives the condition frames, the
prompt, the paired output, and the outcome from the `Draft` alone, so there is
one path for both sources. The condition stride moved into the model half
(`CONDITION_STRIDE`, `condition_frames`). The game's separate condition stack
is gone; `EdenGame.take()` checks that the native flags match the shared slice.

**A switch keeps the world.** `set_source` is a hand-written command, not a
public state field, because a switch during a world has to close one source
and open the other between steps. It does not bump `world_id`: the model keeps
its cache and the look carries over, and the new draft takes over gradually.
Switching to the game opens it from the start, because a game's simulation
cannot resume where a client stream left off. An outcome belongs to the game,
so switching away clears `ended`.

**`client` mode needs a reference, not a game.** `start` asks for a reference
(`image_required`), and `set_image` skips the game requirement. `select_game`
still works in `client` mode as a quick way to borrow a game's cover and
prompt as the look.

**The contract was polished** (`polish-reactor-model-schema`): every command
says what, when, what it emits, and which codes it fails with. The snapshot
message lost the model prefix (`eden_status` to `state_update`) and gained
`source`. That rename is the breaking change behind the major bump.

**Kept as is.** The app still resolves the model's execution flags
(`attention_backend`, `cuda_graph`, `compile_model`) from `edenlive.yaml` and
passes them to `load()` as plain values; the model half reads no config.
`@disconnected` still releases held keys, because keys are the only thing a
client holds, and only for the client that holds them.

## What came after 1.0.0, and why

These shipped as `edenlive` 1.0.1 to 1.0.3 and are folded into `eden-live`
1.0.0 (see "The rename" below). None of them moves the schema.

**Load pays for compilation (`warmup: true`).** The first chunks of the first
world used to compile for about a minute in front of a client. `EdenModel.load()`
now generates throwaway worlds with random input, past the CUDA graph capture
point and through a prompt change and a fresh world, then calls `reset()`. The
warmup lives in the model half because it only drives the model's own
`generate()`; the application passes the flag through like the other execution
flags.

**Chunk timing is measured on the application side** (`edenlive_timing.py`).
Only the application sees the whole cycle (the blocking `emit()`, refused
steps, `generate()`, the source's capture), so the timer lives there and the
model half stays unaware of it. A summary covers one measurement run: a pause,
`start`, `set_source`, and a new session all start a new run and drop the
samples not yet summarized, so a summary never mixes worlds, sources, or
sessions.

**The game renders on the GPU, and each chunk is captured on demand.** On
SwiftShader the game took 0.62-0.65 s per 12-frame chunk at 1280x704, more than
the 0.5 s a chunk plays for, and it slowed further under CPU contention.
Chromium could not reach the GPU because the image lacked the GLVND EGL loader
NVIDIA's driver plugs into and the container did not ask for the graphics
driver capability. With `libegl1` and `libgles2` installed,
`NVIDIA_DRIVER_CAPABILITIES` including `graphics`, and `EDEN_GAME_GPU=1`, the
game renders on the model's B200 and a chunk captures in about 0.2 s at the
full size. `game_capture` stays a setting: if a deployment falls back to
SwiftShader (the renderer logs which one it chose), `[832, 458]` keeps a
software capture faster than real time, and `EdenGame.take()` scales it to the
1280x704 contract inside the source, never past `take()`. With the capture that
fast, the application no longer prepares the next chunk while `generate()`
runs: each chunk is captured when its step asks for it, so a key accepted
between steps reaches the next chunk. Key to received frame dropped from
1.24-1.32 s to 0.69-0.86 s, and the serial cycle still fits 24 fps. A future
port keeps both: render on the GPU, and capture on demand unless the capture
cannot fit beside `generate()` in one chunk's playout time.

**The draft and the generated frames are two tracks.** `main_video` carries
the generated frames and `draft_video` the draft each was made from, both
1280x704, emitted together with the same timestamps and metadata. One
2560x704 frame forced every client to decode and display both halves, doubled
the width the encoder and the network carry, and could not be shown apart.

## The rename

The model was imported and released as `reactor/edenlive` (0.1.0 to 1.0.3).
It is now the `eden-live` folder and registers as `reactor/eden-live`,
starting again at 1.0.0 with everything `edenlive` 1.0.3 shipped. The Python
modules keep their `edenlive_` prefix and `runtime.import` stays
`edenlive_app:EdenLive`: the name only has to differ from the upstream
`edenlive` package, which these modules must not shadow.

## Not verified on a GPU

The client source is covered by CPU tests (fitting, latest-frame reads, the
shared step, mid-world switches), not by a live B200 run. The reskin quality of
arbitrary client video, and how fast the look follows a mid-world switch, still
need a session against the deployment.
