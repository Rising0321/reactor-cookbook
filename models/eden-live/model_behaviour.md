# EdenLive — model behaviour

EdenLive 5B reskins a **draft video** into a generated world, chunk by chunk.
The client stages an appearance reference and a prompt, picks where the draft
comes from, and steers it. Every chunk returns the generated frames on
`main_video` and the draft frames they were made from on `draft_video`, frame
for frame.

The draft has two sources, switched with one command:

- **`game`**: one of the seven official code games, simulated and rendered
  headless on the server. The client steers it with `set_keys`. This is the
  native EdenLive setting.
- **`client`**: the client's own `source_video` track (a webcam, a screen
  capture, a three.js canvas, a file). The model reskins whatever the client
  sends.

Both sources feed the model through the same path (see "The shared draft path"
below). The model does not know which source drew a chunk.

---

## Tracks

| Direction | Name | Type | Rate | Format |
|---|---|---|---|---|
| In | `source_video` | Video | 24 fps | `(H, W, 3)` uint8, any size; read only while the source is `client` |
| Out | `main_video` | Video | 24 fps | `(N, 704, 1280, 3)` uint8: the generated frames |
| Out | `draft_video` | Video | 24 fps | `(N, 704, 1280, 3)` uint8: the draft each `main_video` frame was made from |

`N` is 9 for a world's first chunk and 12 after it. The two output tracks are
emitted together with the same timestamps, and each frame carries the same
metadata on both, so a client pairs them by `frame_id` or timestamp. Each
output frame carries metadata: `world_id`, `chunk`, `source`, and per source either `frame_id`,
`simulation_tick`, `keys` (game) or `pts` (client).

Client frames are scaled to cover 1280x704 and center-cropped. A chunk reads
the **latest** frames the client sent, never a backlog, so the draft follows
the live video.

**Send `source_video` at 24 fps.** Output plays at a pinned 24 fps, because a
game chunk is 0.5 s of the game's 24 fps clock. A client chunk is the latest
9 or 12 frames, whatever time they span: at 30 fps a chunk spans 0.4 s, plays
slightly slow, and the frames between chunks are skipped. The client source
does not resample by timestamp.

---

## Session lifecycle

The runtime drives one step at a time: `process_input()` (the application
decides whether a step can happen and builds the step input), `generate()`
(one chunk on the GPU), `process_output()` (pairing, metadata, `state_update`).

```
  connect ─────► state_update to the joiner
     │
  ┌──▼──────────────────────────────────────────────────────────────┐
  │ STAGING (paused, no world)                                       │
  │ set_source, select_game, set_image, set_prompt                   │
  │ process_input refuses: "Press Start, or resume the paused world." │
  └──┬──────────────────────────────────────────────────────────────┘
     │ start(seed)   game: needs select_game; client: needs a reference
  ┌──▼──────────────────────────────────────────────────────────────┐
  │ FIRST CHUNK                                                      │
  │ source delivers 9 draft frames, then the step carries the anchor │
  │ (reference, prompt, seed); the model opens a new world            │
  └──┬──────────────────────────────────────────────────────────────┘
  ┌──▼──────────────────────────────────────────────────────────────┐
  │ STEADY CHUNKS (12 frames)                                        │
  │ set_prompt, set_keys, set_source, set_paused apply between chunks │
  │ game: each chunk is captured when its step asks for it           │
  │ client: each chunk reads the latest 12 client frames             │
  └──┬──────────────────────────┬───────────────────────────────────┘
     │ set_paused(true)          │ game reaches an outcome
  ┌──▼────────────────┐   ┌──────▼─────────────────────────────────────┐
  │ PAUSED            │   │ ENDED (paused)                              │
  │ resume continues  │   │ set_paused(false) fails `game_ended`;        │
  │ the same world    │   │ start, or set_source away from the game     │
  └───────────────────┘   └─────────────────────────────────────────────┘
```

`select_game` and `set_image` need a paused world and end it; `start` begins
the next. `start` during a world replaces it. `@session_ended` closes the
source and resets the model. A client leaving releases the held keys only if it is the client that
set them; a viewer leaving changes nothing.

---

## Commands

Every refusal raises `CommandError(code, message)`: the caller's own call
fails with the code, and no other client hears of it. Every accepted command
replies with `state_update` and broadcasts it.

| Command | Valid | Effect | Fails with |
|---|---|---|---|
| `set_source(source)` | any time | `game` or `client`. Before `start`, picks what `start` uses. During a world, the next chunk draws from the new source and the world keeps its history; switching to `game` starts the game from its opening; switching away from an ended game lets resume work. | `game_required` (switch a started world to `game` before `select_game`) |
| `select_game(game)` | paused | Stage the game's cover as the reference and its prompt as `prompt`; ends the world. With source `client`, the cover and prompt style the client's video. | `pause_required` |
| `set_image(image)` | paused | Stage an uploaded reference; ends the world. | `pause_required`, `game_required` (source `game`, no game yet), `invalid_image` |
| `start(seed)` | any time | New world from the reference, `prompt`, `seed`, and the current source; resumes. | `game_required` (source `game`), `image_required` (source `client`), `prompt_required` |
| `set_prompt(prompt)` | any time | Next chunk's prompt; history stays, so the look shifts gradually. | `prompt_required` |
| `set_keys(keys)` | any time | Keys held in the game; each uncaptured game frame reads the latest. Ignored while paused or with source `client`. | none |
| `set_paused(paused)` | any time | Hold or resume the source and generation; releases keys. | `start_required`, `game_ended` |

## Messages

| Message | When | Payload |
|---|---|---|
| `state_update` | connect (joiner only), every accepted command, every chunk | `source`, `game`, `image_name`, `prompt`, `seed`, `keys`, `paused`, `started`, `ended`, `world_id`, `chunks`, `seconds` |

There is no `command_error` message: a refusal fails the call itself.

---

## The shared draft path

A **draft source** is anything with the calls below; the application holds
exactly one in `self._draft_source`:

```python
kind: str                                  # "game" or "client"
def prepare(self, frames: int) -> None      # ask for the next chunk
async def take(self) -> Draft | None        # the chunk, or None while not ready
def set_keys(self, held) -> None            # a source without keys ignores it
def set_paused(self, paused) -> None
def close(self) -> None
```

`EdenGame` (`edenlive_game.py`) and `ClientSource` (`edenlive_draft.py`) are
the two implementations. A `Draft` carries `frames` (RGB uint8
`[T,704,1280,3]`), per-frame `metadata`, prompt `events` for this chunk, and
`ended`.

From `take()` on, one code path serves both sources:

1. `process_input()` derives the model's condition with
   `condition_frames(draft.frames)`, the draft's chunk-local frames 0, 4, 8
   (`CONDITION_STRIDE`, owned by the model half), and the prompt as
   `prompt + draft.events`.
2. `generate()` forwards `EdenInput` to the model half, which never learns
   the source.
3. `process_output()` pairs `draft.frames` with the generated frames, merges
   `draft.metadata`, and pauses the world when `draft.ended`.

The game's native condition frames are its own render at chunk-local frames
0, 4, 8; `EdenGame.take()` refuses any other layout, so the shared slice
reproduces the native pipeline exactly.

## Maintaining this behaviour

Agents and reviewers keep these invariants. A change that breaks one is wrong
even if it runs.

- **One draft path.** No `if source == ...` past `take()`. A difference
  between sources lives inside the source class, behind the protocol above.
  A new source (a recorded clip, a second game engine) is a new class with
  the same calls and one branch in `_open_source()`, nothing else.
- **The model half never learns the source.** `EdenInput` carries condition
  frames, prompt, world id, and anchor only. The stride lives in
  `edenlive_model.py`, and both sources reach it through `condition_frames`.
- **A source switch is not a new world.** `set_source` changes what the next
  chunk is drawn from; only `start` bumps `world_id` and sends an anchor.
- **Refusals raise `CommandError` with a stable code** named in the command's
  description, and change no state. No `command_error` message is published.
- **Every state change ends in `state_update`.** A new observable field goes
  on `StateUpdate` and into `_state_update()`.
- **Keep this file, `PORTING.md`, and the descriptions in step.** A change to
  a command, a code, a track, or the draft path updates this file in the
  same change.

---

## Deployment

One NVIDIA B200, runtime 3.8.3, image built from `reactor.yaml` (no
Dockerfile). The game runs in headless Chromium (Playwright) in the
application process, rendering on the model's GPU; the model runs in its own process behind the runtime's
`DistributedRunner` with `world_size=1`.

**Load compiles and warms up** (`warmup: true` in `edenlive.yaml`). After the
weights load, `EdenModel.load()` generates throwaway worlds with random input:
one past the CUDA graph capture point (upstream captures from chunk 6, after 2
wrapper warmups) and through a prompt change, then a fresh world, then
`reset()`. Measured on a B200 with an empty compile cache: about 30 s of
weights, about 65 s of warmup (the first chunk alone compiles for about 53 s),
110 s in all. After it, a world's first chunk takes about 0.7 s (it encodes the
reference and prompt), chunks 1 to 6 about 0.33 s while the cache fills, and
every chunk after 0.21 s, with no compile spike in any later world. A prompt
change costs one chunk about 20 ms. Peak memory is about 33 GiB.

**Chunk timing in a live session.** Every 12 chunks the application logs
`chunk timing summary` (`edenlive_timing.py`): new frames per second, and the
median cycle split into `emit_s` (the previous chunk's `emit()`, which blocks
while the one-chunk playout queue is full), `draft_wait_s` (refused steps while
the source had no draft), `generate_s` (`generate()` as the runtime measures
it), and `capture_s` (the game's own capture time). A summary covers one
measurement run: a pause, `start`, `set_source`, and a new session each drop
the samples not yet summarized. Measured through the Python SDK on a B200,
game mode, with the game rendered on SwiftShader and each chunk captured while
the previous one generated:

| Game, input | cycle | new fps | capture | generate | emit |
|---|---|---|---|---|---|
| `freeze`, no keys | 0.55 s | 21 | 0.54 s | 0.22 s | 0 |
| `godzilla`, keys every 0.4 s | 0.52 s | 23 | 0.51 s | 0.25 s | 0 |
| `dust2`, keys every 0.4 s | 0.58-0.63 s | 19-21 | 0.57-0.62 s | 0.25 s | 0 |
| `dust2`, keys every 33 ms (the demo's rate) | 0.63-0.67 s | 18-19 | 0.62-0.65 s | 0.25 s | 0 |
| same, game rendered at 832x458 (`game_capture`) | 0.50-0.52 s | 23-24 | 0.35-0.40 s | 0.25 s | 0.08-0.13 s |

On SwiftShader (the table above), the capture is the bottleneck: it rasterizes each
frame on about 9 CPU cores, and the read-back dominates. A chunk plays in
0.5 s at the pinned 24 fps, so the playout queue drains before the next chunk
and the last frame repeats for 0.05-0.15 s at every chunk boundary. Capture
time scales with pixels, and the model conditions on the draft at 832x480, so
rendering the game at 832x458 (the draft's aspect) and scaling the draft to
1280x704 fits the cycle in 0.5 s; an A/B on `dust2` showed the output following
the draft as well as at full size. That is the software fallback
(`game_capture: [832, 458]`); the default renders on the GPU at full size (see
Deployment). Game events change the prompt on almost every chunk, which
costs about 30 ms of `generate()` for the re-encode and graph recapture. The
runtime paces an unpinned model from `generate()` alone, which here is not the
chunk period, so unpinning would play each chunk at about 50 fps and stall
longer.

**Measured choices.** FA4 + `torch.compile` + CUDA graphs is the fastest
profile (0.21 s per 12-frame chunk, about 57 FPS of model throughput against
24 fps playback). Inductor `max_autotune` was slower (0.227 s) and is not used.
The game renders on the model's GPU through headless Chromium (ANGLE over
NVIDIA EGL): the image installs the GLVND loaders (`libegl1`, `libgles2`), the
container asks for the `graphics` driver capability, and `EDEN_GAME_GPU=1`. A
12-frame chunk then captures in 0.20-0.24 s at the full 1280x704, against
0.49-0.67 s on SwiftShader, which is CPU-bound and slows down under contention.
Without the loaders or the capability, Chromium falls back to SwiftShader on
its own; the renderer logs which one it chose (`Game2V capture WebGL
renderer`), and `game_capture: [832, 458]` keeps a software capture faster
than real time. Client mode does not pay any capture cost.

**Capture on demand, for input latency.** Each game chunk is captured when
its step asks for it, not while the previous chunk generates. A key accepted
between steps then reaches the very next chunk, instead of the one after an
already-captured chunk. On the GPU a chunk captures in about 0.2 s and
generates in about 0.26-0.31 s, so the serial cycle still fits 24 fps (23.5-24
new fps measured on `dust2` with keys every 33 ms). Measured from `set_keys`
to the first received frame carrying the key, on the host itself (a remote
browser adds its network hop and jitter buffer): 0.69-0.86 s on demand,
against 1.24-1.32 s when the next chunk was captured ahead. The weights bundle holds the
authorized EdenLive source tree (`source/`, games included) and the HF
checkpoints (`weights/EdenLive`); `edenlive.yaml` names both. The default
profile (FA4, `torch.compile`, CUDA graphs) measured about 57.6 FPS of model
throughput on a B200 against 24 fps playback; the first chunk pays for
compilation.
