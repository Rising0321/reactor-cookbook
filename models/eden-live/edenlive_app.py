"""Serve EdenLive without shadowing the upstream ``edenlive`` Python package."""

import asyncio
from pathlib import Path

import numpy as np
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
    session_ended,
)
from reactor_runtime.distributed import DistributedRunner
from reactor_runtime.paths import get_weights_path

from edenlive_assets import GAMES, resolve_assets
from edenlive_game import EdenGame, event_lines
from edenlive_images import decode_image, read_image
from edenlive_model import EdenAnchor, EdenInput, EdenModel, EdenResult
from edenlive_types import EdenOutput, EdenState, EdenStatus, Key


class EdenLive(ReactorApp):
    """Choose a native game or reference, then steer synchronized draft/generated video."""

    state: EdenState
    buffer_size = 12
    # The native 60 Hz simulation samples at 24 FPS; both halves share timestamps.
    fps = 24

    def __init__(self):
        super().__init__()
        self._engine = None
        self._game = None
        self._source = None
        self._entries = {}

    def load(self, config_path: Path | None) -> None:
        self._source, weights, self._entries, execution = resolve_assets(
            config_path, get_weights_path()
        )
        self._engine = DistributedRunner(
            EdenModel,
            world_size=1,
            call_timeout=180,
            load_kwargs={"source": self._source, "weights": weights, **execution},
        )
        self._engine.start()

    @connected
    async def on_connected(self, client: ClientInfo) -> None:
        await client.send(self._status())

    @disconnected
    async def on_disconnected(self) -> None:
        # A viewer leaving releases controls without destroying a shared world.
        self._release_keys()
        await self.send(self._status())

    @session_ended
    def on_session_ended(self) -> None:
        self._close_game()
        self._engine.reset()

    @event(
        name="select_game",
        description="Stage an official game, its cover and prompt. "
        "Waits for Start. Fails with `pause_required` during generation.",
    )
    async def select_game(
        self,
        game: str = InputField(
            choices=list(GAMES), description="Official code game to stage."
        ),
    ) -> EdenStatus:
        self._require_paused()
        entry = self._entries[game]
        path = self._source / "realtime/games" / game / entry["cover"]
        pixels = await asyncio.to_thread(read_image, path)
        self._close_game()
        self.state._game = game
        self.state._image = pixels
        self.state._image_name = f"{game}/{entry['cover']}"
        self.state.prompt = entry["prompt"]
        self.state._started = False
        self.state._chunks = 0
        self.state._seconds = None
        self.state._ended = False
        self.output.flush()
        return await self._publish()

    @event(
        name="set_image",
        description="Stage an uploaded appearance reference; waits for Start. "
        "Fails with `pause_required` during generation, `game_required` before game selection, "
        "or `invalid_image` for an empty, oversized or undecodable PNG/JPEG/WebP.",
    )
    async def set_image(
        self,
        image: UploadedFile = InputField(  # noqa: B008 - Runtime schema declaration
            moderate=True,
            description="PNG/JPEG/WebP, at most 25 MiB and 16 megapixels; resized and center-cropped to 1280×704 at Start.",
        ),
    ) -> EdenStatus:
        self._require_paused()
        if not self.state._game:
            raise CommandError("game_required", "Choose a game first.")
        try:
            pixels = await asyncio.to_thread(decode_image, image.data)
        except (OSError, ValueError, Image.DecompressionBombError) as error:
            raise CommandError(
                "invalid_image", "Choose a valid PNG/JPEG/WebP up to 25 MiB and 16 MP."
            ) from error
        self.state._image = pixels
        self.state._image_name = image.name
        self.state._started = False
        self.state._chunks = 0
        self.state._seconds = None
        self.state._ended = False
        self._close_game()
        self.output.flush()
        return await self._publish()

    @event(
        name="start",
        description="Start a fresh game and video from the staged reference. "
        "Fails with `game_required` before selection or `prompt_required` for empty text.",
    )
    async def start(
        self,
        seed: int = InputField(
            default=2026092901,
            ge=0,
            le=2147483647,
            description="Random seed for the new world.",
        ),
    ) -> EdenStatus:
        if not self.state._game or self.state._image is None:
            raise CommandError(
                "game_required", "Choose a game and reference before Start."
            )
        if not self.state.prompt.strip():
            raise CommandError(
                "prompt_required", "Enter a scene description before Start."
            )
        self._close_game()
        self._game = EdenGame(source=self._source, slug=self.state._game)
        self.state._seed = seed
        self.state._world_id += 1
        self.state._chunks = 0
        self.state._frames_wanted = 9
        self.state._seconds = None
        self.state._ended = False
        self.state._proxy = None
        self.state._packet = None
        self.state.keys = None
        self.state._sampled_keys = ()
        self.state._started = True
        self.state.paused = False
        self.output.flush()
        return await self._publish()

    @event(
        name="set_prompt",
        description="Update the next chunk's scene text while retaining video history. "
        "A paused world stays paused. Fails with `prompt_required` for blank text.",
    )
    async def set_prompt(
        self,
        prompt: str = InputField(
            max_length=4096,
            moderate=True,
            description="Non-empty scene description, up to 4096 characters.",
        ),
    ) -> EdenStatus:
        if not prompt.strip():
            raise CommandError(
                "prompt_required", "Enter a non-empty scene description."
            )
        self.state.prompt = prompt.strip()
        return await self._publish()

    @event(
        name="set_keys",
        description="Replace the current held keys without queuing past transitions. "
        "The next uncaptured draft frame samples this state. Empty releases all controls.",
    )
    async def set_keys(
        self,
        keys: list[Key] | None = InputField(  # noqa: B008 - Runtime schema declaration
            default=None,
            max_length=14,
            description="Current held native keys. Brief taps between samples are not replayed.",
        ),
    ) -> EdenStatus:
        self.state.keys = [] if self.state.paused else sorted(set(keys or ()))
        if self._game is not None:
            self._game.set_keys(tuple(self.state.keys))
        return await self._publish()

    @event(
        name="set_paused",
        description="Pause or resume the current simulation and video. "
        "Releases held controls. Fails with `start_required` before Start or `game_ended` after an outcome.",
    )
    async def set_paused(
        self,
        paused: bool = InputField(
            default=True, description="True pauses; false resumes the same world."
        ),
    ) -> EdenStatus:
        if not paused and not self.state._started:
            raise CommandError("start_required", "Press Start before resuming.")
        if not paused and self.state._ended:
            raise CommandError(
                "game_ended", "This game has ended. Press Start new world."
            )
        self.state.paused = paused
        self._release_keys()
        if self._game is not None:
            self._game.set_paused(paused)
        return await self._publish()

    async def process_input(self) -> EdenInput:
        state = self.state
        if not state._started or state.paused or state._ended:
            raise ApplicationError(
                "Choose a game and press Start, or resume the paused world."
            )
        self._game.prepare(state._frames_wanted)
        if not self._game.ready():
            # Refusing releases the Runtime step lock: commands can replace
            # the mailbox while the native capture thread renders this batch.
            raise ApplicationError("Waiting for the next draft chunk.")
        proxy, condition, packet, sampled = self._game.collect()
        state._proxy, state._packet = proxy, packet
        state._sampled_keys = sampled
        # After the initial chunk the model requests a fixed 12-frame cadence.
        # Prepare just one successor while generate() consumes this chunk.
        # The first successor waits for the model's frames_wanted result.
        if state._world_id == state._applied_world_id:
            self._game.prepare(state._frames_wanted)
        prompt = " ".join((state.prompt, *event_lines(packet)))
        anchor = None
        if state._world_id != state._applied_world_id:
            anchor = EdenAnchor(
                image=state._image, prompt=state.prompt, seed=state._seed
            )
        return EdenInput(
            world_id=state._world_id, anchor=anchor, prompt=prompt, condition=condition
        )

    def generate(self, input: EdenInput) -> EdenResult:
        return self._engine.generate(input)

    async def process_output(self, outcome: StepOutcome) -> EdenOutput:
        if outcome.error is not None:
            # Unknown inference failures must end visibly rather than reset history silently.
            raise outcome.error
        result = outcome.result
        state = self.state
        if (
            result.world_id != state._world_id
            or state._proxy.shape != result.frames.shape
        ):
            raise RuntimeError("Generated chunk does not match the captured draft")
        paired = np.concatenate((state._proxy, result.frames), axis=2)
        metadata = [
            {
                "world_id": result.world_id,
                "chunk": result.chunk_index,
                "frame_id": frame.game_frame_id,
                "simulation_tick": frame.simulation_tick,
                "keys": list(keys),
            }
            for frame, keys in zip(
                state._packet.frames, state._sampled_keys, strict=True
            )
        ]
        state._applied_world_id = result.world_id
        state._chunks = result.chunk_index
        state._frames_wanted = result.frames_wanted
        state._seconds = outcome.elapsed
        last = state._packet.frames[-1].ui_state
        # The shipped games report outcomes through this shared HUD convention.
        # Only Start creates a fresh game; a terminal game cannot silently resume.
        if isinstance(last, dict) and last.get("state", "") not in (
            "",
            "RUNNING",
            "READY",
        ):
            state._ended = True
            state.paused = True
        if state._ended:
            self._release_keys()
            self._game.set_paused(True)
        state._proxy = None
        state._packet = None
        state._sampled_keys = ()
        await self.send(self._status())
        return EdenOutput(main_video=TrackPayload(paired, metadata=metadata))

    def _close_game(self):
        if self._game is not None:
            self._game.close()
            self._game = None

    def _release_keys(self):
        self.state.keys = None
        if self._game is not None:
            self._game.set_keys(())

    def _require_paused(self):
        if not self.state.paused:
            raise CommandError(
                "pause_required", "Pause before changing the game or reference."
            )

    def _status(self) -> EdenStatus:
        state = self.state
        return EdenStatus(
            game=state._game,
            image_name=state._image_name,
            prompt=state.prompt,
            seed=state._seed,
            keys=list(state.keys or ()),
            paused=state.paused,
            started=state._started,
            world_id=state._world_id,
            chunks=state._chunks,
            seconds=state._seconds,
            ended=state._ended,
        )

    async def _publish(self) -> EdenStatus:
        message = self._status()
        await self.send(message)
        return message
