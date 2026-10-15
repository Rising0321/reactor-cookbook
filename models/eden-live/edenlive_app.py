# Copyright (c) 2026 Reactor Technologies, Inc. All rights reserved.
"""Serve EdenLive without shadowing the upstream ``edenlive`` Python package."""

import asyncio
import time
from pathlib import Path

import numpy as np
from edenlive_assets import GAMES, resolve_assets
from edenlive_draft import ClientSource, DraftSource
from edenlive_game import EdenGame
from edenlive_images import decode_image, read_image
from edenlive_model import EdenAnchor, EdenInput, EdenModel, EdenResult, condition_frames
from edenlive_timing import ChunkTimer
from edenlive_types import SOURCES, EdenMedia, EdenOutput, EdenState, Key, StateUpdate
from PIL import Image
from reactor_runtime import (
    ApplicationError,
    ClientInfo,
    CommandError,
    InputField,
    ReactorApp,
    StepOutcome,
    TrackPayload,
    UploadedFile,
    connected,
    disconnected,
    event,
    get_logger,
    session_ended,
    session_started,
)
from reactor_runtime.distributed import DistributedRunner
from reactor_runtime.paths import get_weights_path

logger = get_logger(__name__)


class EdenLive(ReactorApp):
    """Reskin a draft video, from a native game or from the client, into a generated world."""

    state: EdenState
    media: EdenMedia
    # One chunk, the smallest bound the runtime applies: the least latency.
    buffer_size = 12
    # A chunk covers 0.5 s of the game's 24 FPS clock, so it must play at 24 FPS.
    fps = 24

    def __init__(self):
        super().__init__()
        self._engine = None
        self._draft_source: DraftSource | None = None
        self._root = None
        self._entries = {}
        self._capture = (1280, 704)
        self._timer = ChunkTimer()
        self._last_prompt = ""

    def load(self, config_path: Path | None) -> None:
        self._root, weights, self._entries, execution, self._capture = resolve_assets(
            config_path, get_weights_path()
        )
        self._engine = DistributedRunner(
            EdenModel,
            world_size=1,
            call_timeout=180,
            load_kwargs={"source": self._root, "weights": weights, **execution},
        )
        self._engine.start()

    @connected
    async def on_connected(self, client: ClientInfo) -> None:
        await client.send(self._state_update())

    @disconnected
    async def on_disconnected(self, client: ClientInfo) -> None:
        # Only the client holding the keys releases them; a viewer leaving
        # changes nothing.
        if self.state._keys_holder != client.id:
            return
        self._release_keys()
        await self.send(self._state_update())

    @session_started
    def on_session_started(self) -> None:
        self._timer = ChunkTimer()
        self._last_prompt = ""

    @session_ended
    def on_session_ended(self) -> None:
        self._close_source()
        self._engine.reset()

    # -- commands -------------------------------------------------------------

    @event(
        name="set_source",
        description=(
            "Choose where the draft video comes from: `game` (the selected native "
            "game, steered with `set_keys`) or `client` (the client's `source_video` "
            "track, sent at 24 fps: each chunk takes the latest frames and plays "
            "them at 24 fps, so another rate plays faster or slower). Valid any "
            "time. Before `start` it picks the source `start` "
            "uses. During a world the switch applies from the next chunk and the "
            "world keeps its visual history; switching to `game` starts the "
            "selected game from its opening, and switching away from a game that "
            "reached an outcome lets `set_paused` resume. Emits `state_update`. "
            "Fails with `game_required` when switching a started world to `game` "
            "before `select_game`."
        ),
    )
    async def set_source(
        self,
        source: str = InputField(
            choices=list(SOURCES),
            description="`game` or `client`.",
        ),
    ) -> StateUpdate:
        state = self.state
        if source == state._source:
            return await self._publish()
        if source == "game" and state._started and not state._game:
            raise CommandError("game_required", "Choose a game before switching to it.")
        state._source = source
        state._ended = False
        self._timer.restart()
        self._release_keys()
        if state._started:
            self._open_source()
        return await self._publish()

    @event(
        name="select_game",
        description=(
            "Stage one of the official games: its cover becomes the appearance "
            "reference and its prompt becomes `prompt`. Valid while paused. Ends "
            "the current world; `start` begins a new one. With source `client` "
            "the cover and prompt style the client's video. Emits `state_update`. "
            "Fails with `pause_required` while a world plays."
        ),
    )
    async def select_game(
        self,
        game: str = InputField(choices=list(GAMES), description="Official game to stage."),
    ) -> StateUpdate:
        self._require_paused()
        entry = self._entries[game]
        path = self._root / "realtime/games" / game / entry["cover"]
        pixels = await asyncio.to_thread(read_image, path)
        self.state._game = game
        self.state.prompt = entry["prompt"]
        self._stage_reference(pixels, f"{game}/{entry['cover']}")
        return await self._publish()

    @event(
        name="set_image",
        description=(
            "Stage an uploaded appearance reference in place of the game cover. "
            "Valid while paused. Ends the current world; `start` begins a new one "
            "from this image. Emits `state_update`. Fails with `pause_required` "
            "while a world plays, `game_required` before `select_game` when the "
            "source is `game`, or `invalid_image` for an empty, oversized, or "
            "undecodable file."
        ),
    )
    async def set_image(
        self,
        image: UploadedFile = InputField(  # noqa: B008 - Runtime schema declaration
            moderate=True,
            description=(
                "PNG, JPEG, or WebP, at most 25 MiB and 16 megapixels. Scaled and "
                "center-cropped to 1280x704 when the world starts. Works best when "
                "its composition matches the draft's opening frame."
            ),
        ),
    ) -> StateUpdate:
        self._require_paused()
        if self.state._source == "game" and not self.state._game:
            raise CommandError("game_required", "Choose a game first.")
        try:
            pixels = await asyncio.to_thread(decode_image, image.data)
        except (OSError, ValueError, Image.DecompressionBombError) as error:
            raise CommandError(
                "invalid_image", "Choose a valid PNG/JPEG/WebP up to 25 MiB and 16 MP."
            ) from error
        self._stage_reference(pixels, image.name)
        return await self._publish()

    @event(
        name="start",
        description=(
            "Start a new world from the staged reference and `prompt`, drawing the "
            "draft from the current source, and resume. Valid any time; a running "
            "world is replaced. The first chunk arrives on `main_video` and `draft_video` once the "
            "source has delivered its draft frames. Emits `state_update`. Fails "
            "with `game_required` before `select_game` when the source is `game`, "
            "`image_required` without a reference when the source is `client`, or "
            "`prompt_required` when `prompt` is blank."
        ),
    )
    async def start(
        self,
        seed: int = InputField(
            default=2026092901,
            ge=0,
            le=2147483647,
            description=(
                "Seed of the new world. The same seed, reference, and draft repeat a world."
            ),
        ),
    ) -> StateUpdate:
        state = self.state
        if state._source == "game" and (not state._game or state._image is None):
            raise CommandError("game_required", "Choose a game and reference before Start.")
        if state._image is None:
            raise CommandError("image_required", "Choose a game or upload a reference first.")
        if not state.prompt.strip():
            raise CommandError("prompt_required", "Enter a scene description before Start.")
        state._seed = seed
        state._world_id += 1
        state._chunks = 0
        state._frames_wanted = 9
        state._seconds = None
        state._ended = False
        state._draft = None
        state.keys = None
        state._started = True
        self._timer.restart()
        state.paused = False
        self._open_source()
        self.output.flush()
        return await self._publish()

    @event(
        name="set_prompt",
        description=(
            "Change `prompt`. Valid any time; applies from the next chunk and keeps "
            "the world's visual history, so the look changes gradually. A paused "
            "world stays paused. Emits `state_update`. Fails with "
            "`prompt_required` for blank text."
        ),
    )
    async def set_prompt(
        self,
        prompt: str = InputField(
            max_length=4096,
            moderate=True,
            description="Scene description, non-empty, up to 4096 characters.",
        ),
    ) -> StateUpdate:
        if not prompt.strip():
            raise CommandError("prompt_required", "Enter a non-empty scene description.")
        self.state.prompt = prompt.strip()
        return await self._publish()

    @event(
        name="set_keys",
        description=(
            "Replace the keys held in the game. Valid any time; the next game "
            "frame not yet captured reads them, and a press released before it "
            "is captured is dropped. Ignored while paused or while the source is "
            "`client`. Emits `state_update`."
        ),
    )
    async def set_keys(
        self,
        keys: list[Key] | None = InputField(  # noqa: B008 - Runtime schema declaration
            default=None,
            max_length=14,
            description="Keys held now; empty or null releases every key.",
        ),
        client: ClientInfo | None = None,
    ) -> StateUpdate:
        self.state.keys = [] if self.state.paused else sorted(set(keys or ()))
        self.state._keys_holder = client.id if client is not None and self.state.keys else None
        if self._draft_source is not None:
            self._draft_source.set_keys(tuple(self.state.keys))
        return await self._publish()

    @event(
        name="set_paused",
        description=(
            "Pause or resume the world. Pausing holds the draft source and "
            "generation and releases held keys; resuming continues the same world. "
            "Emits `state_update`. Fails with `start_required` when resuming before "
            "`start`, or `game_ended` when resuming a game that reached an outcome."
        ),
    )
    async def set_paused(
        self,
        paused: bool = InputField(
            default=True, description="True pauses; false resumes the same world."
        ),
    ) -> StateUpdate:
        if not paused and not self.state._started:
            raise CommandError("start_required", "Press Start before resuming.")
        if not paused and self.state._ended:
            raise CommandError("game_ended", "This game has ended. Press Start new world.")
        self.state.paused = paused
        self._timer.restart()
        self._release_keys()
        if self._draft_source is not None:
            self._draft_source.set_paused(paused)
        return await self._publish()

    # -- the step -------------------------------------------------------------

    async def process_input(self) -> EdenInput:
        state = self.state
        if not state._started or state.paused or state._ended:
            raise ApplicationError("Press Start, or resume the paused world.")
        source = self._draft_source
        self._timer.attempt(time.monotonic())
        source.prepare(state._frames_wanted)
        draft = await source.take()
        if draft is None:
            # Refusing releases the step lock, so commands land while the
            # source captures or receives this chunk.
            raise ApplicationError(
                f"Waiting for {state._frames_wanted} draft frames from the {source.kind}."
            )
        # The next chunk is prepared only when the next step asks for it, so a
        # key accepted between steps reaches the very next chunk.
        state._draft = draft
        self._timer.drafted(time.monotonic())
        anchor = None
        if state._world_id != state._applied_world_id:
            anchor = EdenAnchor(image=state._image, prompt=state.prompt, seed=state._seed)
        prompt = " ".join((state.prompt, *draft.events))
        state._prompt_changed = prompt != self._last_prompt
        self._last_prompt = prompt
        return EdenInput(
            world_id=state._world_id,
            anchor=anchor,
            prompt=prompt,
            condition=condition_frames(draft.frames),
        )

    def generate(self, input: EdenInput) -> EdenResult:
        return self._engine.generate(input)

    async def process_output(self, outcome: StepOutcome) -> EdenOutput:
        if outcome.error is not None:
            # An inference failure ends the session visibly; a silent reset
            # would hide a discontinuity in the world.
            raise outcome.error
        result: EdenResult = outcome.result
        state = self.state
        draft, state._draft = state._draft, None
        if result.world_id != state._world_id or draft.frames.shape != result.frames.shape:
            raise RuntimeError("Generated chunk does not match its draft")
        metadata = [
            {"world_id": result.world_id, "chunk": result.chunk_index, **frame}
            for frame in draft.metadata
        ]
        state._applied_world_id = result.world_id
        state._chunks = result.chunk_index
        state._frames_wanted = result.frames_wanted
        state._seconds = outcome.elapsed
        if draft.ended:
            # Only `start` or `set_source` continues after an outcome.
            state._ended = True
            state.paused = True
            self._release_keys()
            self._draft_source.set_paused(True)
        await self.send(self._state_update())
        self._measure(draft, outcome.elapsed)
        return EdenOutput(
            main_video=TrackPayload(result.frames, metadata=metadata),
            draft_video=TrackPayload(draft.frames, metadata=metadata),
        )

    # -- helpers --------------------------------------------------------------

    def _measure(self, draft, generate_s: float | None) -> None:
        sample = self._timer.finished(
            time.monotonic(),
            frames=len(draft.frames),
            generate_s=generate_s,
            capture_s=draft.capture_seconds,
            prompt_changed=self.state._prompt_changed,
        )
        if sample is not None:
            logger.debug("chunk timing", chunk=self.state._chunks, **sample)
        summary = self._timer.summary()
        if summary is not None:
            logger.info("chunk timing summary", source=self.state._source, **summary)

    def _open_source(self) -> None:
        self._close_source()
        if self.state._source == "game":
            self._draft_source = EdenGame(
                source=self._root, slug=self.state._game, size=self._capture
            )
        else:
            self._draft_source = ClientSource(self.media.source_video)
        self._draft_source.set_paused(self.state.paused)

    def _close_source(self) -> None:
        if self._draft_source is not None:
            self._draft_source.close()
            self._draft_source = None

    def _stage_reference(self, pixels: np.ndarray, name: str) -> None:
        state = self.state
        state._image = pixels
        state._image_name = name
        state._started = False
        state._chunks = 0
        state._seconds = None
        state._ended = False
        self._close_source()
        self.output.flush()

    def _release_keys(self) -> None:
        self.state.keys = None
        self.state._keys_holder = None
        if self._draft_source is not None:
            self._draft_source.set_keys(())

    def _require_paused(self) -> None:
        if not self.state.paused:
            raise CommandError("pause_required", "Pause before changing the game or reference.")

    def _state_update(self) -> StateUpdate:
        state = self.state
        return StateUpdate(
            source=state._source,
            game=state._game,
            image_name=state._image_name,
            prompt=state.prompt,
            seed=state._seed,
            keys=list(state.keys or ()),
            paused=state.paused,
            started=state._started,
            ended=state._ended,
            world_id=state._world_id,
            chunks=state._chunks,
            seconds=state._seconds,
        )

    async def _publish(self) -> StateUpdate:
        message = self._state_update()
        await self.send(message)
        return message
