"""Serve the public Matrix-Game 3.0 distilled model through Reactor Runtime.

The adapter keeps the official interactive generation loop intact and supplies
one native keyboard/mouse action at each upstream iteration boundary. The loop
therefore continues to own its rolling image condition, camera-aware memory
latents, full action and pose history, denoising state, and streaming VAE cache.
"""

from __future__ import annotations

import asyncio
import importlib
from collections.abc import AsyncGenerator
from pathlib import Path
from typing import TYPE_CHECKING, Protocol, override

import numpy as np
from numpy.typing import NDArray
from reactor_runtime import (
    ClientInfo,
    CommandError,
    Idle,
    InputField,
    ReactorPipeline,
    UploadedFile,
    connected,
    disconnected,
    event,
    session_ended,
    session_started,
)
from reactor_runtime.log import get_logger

if TYPE_CHECKING:
    from matrix_game_3_0_assets import MatrixGame30Config
    from matrix_game_3_0_backend import NativeAction
    from matrix_game_3_0_schema import (
        MOVEMENT_KEYS,
        ControlsChanged,
        MatrixGame30Output,
        MatrixGame30State,
        MovementKey,
        RolloutLimitReached,
        StateUpdate,
    )
else:
    module_prefix = f"{__package__}." if __package__ else ""
    assets = importlib.import_module(f"{module_prefix}matrix_game_3_0_assets")
    backend_module = importlib.import_module(f"{module_prefix}matrix_game_3_0_backend")
    images = importlib.import_module(f"{module_prefix}matrix_game_3_0_images")
    schema = importlib.import_module(f"{module_prefix}matrix_game_3_0_schema")
    MatrixGame30Config = assets.MatrixGame30Config
    prepare_assets = assets.prepare_assets
    read_config = assets.read_config
    MatrixGame30Backend = backend_module.MatrixGame30Backend
    NativeAction = backend_module.NativeAction
    action_from_controls = backend_module.action_from_controls
    normalize_output_frames = images.normalize_output_frames
    validate_uploaded_image = images.validate_uploaded_image
    MOVEMENT_KEYS = schema.MOVEMENT_KEYS
    ControlsChanged = schema.ControlsChanged
    MatrixGame30Output = schema.MatrixGame30Output
    MatrixGame30State = schema.MatrixGame30State
    MovementKey = schema.MovementKey
    RolloutLimitReached = schema.RolloutLimitReached
    StateUpdate = schema.StateUpdate

logger = get_logger(__name__)

FPS = 17


class _Backend(Protocol):
    """Define the blocking upstream operations used by the Reactor loop."""

    def reset(
        self,
        prompt: str,
        seed: int,
        anchor_image: Path | UploadedFile,
    ) -> None:
        """Start a fresh official autoregressive rollout."""

    def generate_chunk(self, action: NativeAction) -> NDArray[np.uint8]:
        """Generate exactly one native Matrix iteration."""

    def end_session(self) -> None:
        """Release state owned by the completed rollout."""


class MatrixGame30(ReactorPipeline):
    """Generate an image-, prompt-, movement-, and view-controlled Matrix world."""

    state: MatrixGame30State
    output: MatrixGame30Output
    fps = FPS
    buffer_size = 1

    def __init__(self) -> None:
        super().__init__()
        self._config: MatrixGame30Config | None = None
        self._backend: _Backend | None = None
        self._selected_input: Path | UploadedFile | None = None
        self._seed = 0
        self._chunk_index = 0
        self._example_index = 0
        self._example_rng = np.random.default_rng()

    @override
    def load(self, config_path: Path | None) -> None:
        """Prepare public assets and load the distilled upstream model once.

        Args:
            config_path: Path to ``matrix_game_3_0.yaml`` from ``reactor.yaml``.
        """
        config = read_config(config_path)
        prepare_assets(config)
        backend = MatrixGame30Backend(config)
        backend.load()
        self._config = config
        self._backend = backend
        self._seed = config.seed
        logger.info(
            "Matrix-Game 3.0 model ready",
            source_revision=config.source_revision,
            checkpoint_revision=config.checkpoint_revision,
            distilled=True,
            int8=config.use_int8,
            vae=config.vae_type,
            fps=FPS,
        )

    @session_started
    def on_session_started(self) -> None:
        """Initialize an empty paused world that waits for an anchor image."""
        config = self._require_config()
        self._selected_input = None
        self._example_index = -1
        self._example_rng = np.random.default_rng(config.seed)
        self._seed = config.seed
        self._chunk_index = 0
        self.state.prompt = ""
        self.state._pressed_keys = frozenset()
        self.state.pitch = 0.0
        self.state.yaw = 0.0
        self.state.paused = True
        self.state._restart_requested = False
        self.state._step_requested = False
        self.state._limit_reached = False

    @connected
    async def on_connected(self, client: ClientInfo) -> None:
        """Send the complete shared world state to one joining viewer."""
        await client.send(self._state_update())

    @disconnected
    async def on_disconnected(self) -> None:
        """Release held movement and view controls when their viewer leaves."""
        self._clear_controls()
        await self.send(self._state_update())

    @session_ended
    async def on_session_ended(self) -> None:
        """Release autoregressive state owned by the completed session."""
        backend = self._backend
        try:
            if backend is not None:
                await asyncio.to_thread(backend.end_session)
        finally:
            self._selected_input = None
            self._chunk_index = 0

    @event(
        name="set_prompt",
        description=(
            "Replace the scene prompt and start a fresh rollout from the selected image. "
            "Valid after an image is selected; the prompt is encoded before the next "
            "57-frame `main_video` chunk, which is queued automatically while playback "
            "remains paused. Returns `state_update` on success, or `command_error` when "
            "`prompt` is empty or no image is selected."
        ),
    )
    def set_prompt(
        self,
        prompt: str = InputField(
            default="",
            max_length=4096,
            description=(
                "Non-empty scene description, up to 4096 characters. Matrix encodes it once "
                "before autoregressive generation, so changing it restarts visual memory "
                "instead of modifying an in-progress rollout."
            ),
        ),
    ) -> StateUpdate:
        """Queue a fresh rollout with a replacement prompt."""
        normalized = prompt.strip()
        if not normalized:
            raise CommandError("prompt_required", "Matrix-Game 3.0 requires a prompt.")
        if self._selected_input is None:
            raise CommandError(
                "image_required", "Select an image before setting a prompt."
            )
        self.state.prompt = normalized
        self._request_fresh_rollout(auto_step=True)
        return self._state_update()

    @event(
        name="set_image",
        description=(
            "Replace the anchor image and start a fresh rollout. Valid at any time; the "
            "uploaded image and optional prompt apply before an automatically queued "
            "57-frame `main_video` chunk while playback remains paused, and all prior "
            "latents, camera memory, actions, poses, and VAE cache are discarded. Returns "
            "`state_update` on success, or `command_error` for an invalid image."
        ),
    )
    def set_image(
        self,
        image: UploadedFile = InputField(  # noqa: B008 - schema field declaration
            description=(
                "Anchor image supplied through Reactor's upload protocol. JPEG, PNG, WebP, "
                "or BMP; at most 25 MiB and 100 million pixels."
            )
        ),
        prompt: str = InputField(
            default="",
            max_length=4096,
            description=(
                "Optional scene description, up to 4096 characters, encoded for the fresh "
                "rollout started from this uploaded image. Leave empty to condition only "
                "on the uploaded image and controls."
            ),
        ),
    ) -> StateUpdate:
        """Select an uploaded anchor and queue its first native chunk."""
        validate_uploaded_image(image)
        normalized = prompt.strip()
        self._selected_input = image
        self.state.prompt = normalized
        self._request_fresh_rollout(auto_step=True)
        return self._state_update()

    @event(
        name="random_image",
        description=(
            "Select a configured public example image and its paired prompt at random, then "
            "start a fresh rollout. Valid at any time; a different example is selected when "
            "more than one is configured, and a 57-frame `main_video` chunk is queued while "
            "playback remains paused. Returns `state_update` on success."
        ),
    )
    def random_image(self) -> StateUpdate:
        """Select a random built-in example and queue its first native chunk."""
        config = self._require_config()
        candidates = [
            index
            for index in range(len(config.examples))
            if index != self._example_index
        ]
        if candidates:
            self._example_index = int(self._example_rng.choice(candidates))
        example = config.examples[self._example_index]
        self._selected_input = example.image
        self.state.prompt = example.prompt
        self._request_fresh_rollout(auto_step=True)
        return self._state_update()

    @event(
        name="set_key_state",
        description=(
            "Hold or release one native W/S/A/D keyboard token for subsequent chunks. Valid "
            "before the 12-chunk rollout limit, including while paused; the complete binary "
            "key state is sampled at the next 57- or 40-frame chunk boundary. Emits "
            "`controls_changed` and broadcasts `state_update` on success, or `command_error` "
            "with `rollout_limit_reached` at the limit."
        ),
    )
    async def set_key_state(
        self,
        key: MovementKey = InputField(  # noqa: B008 - schema field declaration
            default="w",
            choices=MOVEMENT_KEYS,
            description=(
                "Native Matrix movement key. Each held key directly sets its corresponding "
                "binary W/S/A/D action channel; perpendicular pairs produce diagonals."
            ),
        ),
        pressed: bool = InputField(
            default=True,
            description=(
                "Set true to hold `key` for forthcoming chunks or false to release it. The "
                "state persists until another key command or control release."
            ),
        ),
    ) -> ControlsChanged:
        """Update one native keyboard token and report all held controls."""
        self._require_available_rollout()
        if pressed:
            self.state._pressed_keys = self.state._pressed_keys.union((key,))
        else:
            self.state._pressed_keys = self.state._pressed_keys.difference((key,))
        message = self._controls_changed("set_key_state")
        await self.send(self._state_update())
        return message

    @event(
        name="set_pitch",
        description=(
            "Set continuous camera pitch for subsequent native chunks. Valid before the "
            "12-chunk rollout limit, including while paused; the normalized value is scaled "
            "to Matrix's native mouse-x range and sampled at the next chunk boundary. Emits "
            "`controls_changed` and broadcasts `state_update` on success, or `command_error` "
            "with `rollout_limit_reached` at the limit."
        ),
    )
    async def set_pitch(
        self,
        pitch: float = InputField(
            default=0.0,
            ge=-1.0,
            le=1.0,
            description=(
                "Normalized camera pitch from down (-1) to up (1), held until changed. Full "
                "scale maps to the official interactive value ±0.1; zero is neutral."
            ),
        ),
    ) -> ControlsChanged:
        """Set normalized native pitch and report all held controls."""
        self._require_available_rollout()
        self.state.pitch = pitch
        message = self._controls_changed("set_pitch")
        await self.send(self._state_update())
        return message

    @event(
        name="set_yaw",
        description=(
            "Set continuous camera yaw for subsequent native chunks. Valid before the "
            "12-chunk rollout limit, including while paused; the normalized value is scaled "
            "to Matrix's native mouse-y range and sampled at the next chunk boundary. Emits "
            "`controls_changed` and broadcasts `state_update` on success, or `command_error` "
            "with `rollout_limit_reached` at the limit."
        ),
    )
    async def set_yaw(
        self,
        yaw: float = InputField(
            default=0.0,
            ge=-1.0,
            le=1.0,
            description=(
                "Normalized camera yaw from left (-1) to right (1), held until changed. Full "
                "scale maps to the official interactive value ±0.1; zero is neutral."
            ),
        ),
    ) -> ControlsChanged:
        """Set normalized native yaw and report all held controls."""
        self._require_available_rollout()
        self.state.yaw = yaw
        message = self._controls_changed("set_yaw")
        await self.send(self._state_update())
        return message

    @event(
        name="set_paused",
        description=(
            "Pause continuous generation before the next native chunk, or resume it. Valid "
            "at any time when pausing; resuming requires an available chunk. Either value "
            "cancels a queued `step`, releases all held controls, and preserves rollout "
            "memory. Returns `state_update` on success, or `command_error` with "
            "`rollout_limit_reached` when resuming at the limit."
        ),
    )
    def set_paused(
        self,
        paused: bool = InputField(
            default=True,
            description=(
                "Set true to stop before the next chunk, or false to generate continuously. "
                "An in-progress native chunk and its remaining frames can finish. Both values "
                "release W/S/A/D, pitch, and yaw controls."
            ),
        ),
    ) -> StateUpdate:
        """Set playback mode and return the complete shared state."""
        if not paused:
            self._require_available_rollout()
        self.state.paused = paused
        self.state._step_requested = False
        self._clear_controls()
        return self._state_update()

    @event(
        name="step",
        description=(
            "Queue exactly one native `main_video` chunk without leaving paused mode. Valid "
            "only while paused and before the 12-chunk limit; the first chunk contains 57 "
            "frames and later chunks contain 40. Returns `state_update` on success, or "
            "`command_error` when playback is running or the rollout limit is reached."
        ),
    )
    def step(self) -> StateUpdate:
        """Queue one paused native chunk and return the complete shared state."""
        self._require_available_rollout()
        if not self.state.paused:
            raise CommandError(
                "pause_required", "Pause Matrix-Game 3.0 before stepping."
            )
        self.state._step_requested = True
        return self._state_update()

    @event(
        name="reset",
        description=(
            "Restart from the selected anchor image and active prompt. Valid after an image "
            "is selected; the reset clears all autoregressive memory and held controls, "
            "preserves paused mode, and queues the fresh 57-frame `main_video` chunk. Returns "
            "`state_update` on success, or `command_error` when no image is selected."
        ),
    )
    def reset(
        self,
        seed: int = InputField(
            default=-1,
            ge=-1,
            le=2_147_483_647,
            description=(
                "Random seed from 0 to 2147483647 for the fresh rollout. Use -1 to retain "
                "the active seed."
            ),
        ),
    ) -> StateUpdate:
        """Queue a fresh rollout from the selected anchor and prompt."""
        if self._selected_input is None:
            raise CommandError("image_required", "Select an image before resetting.")
        if seed >= 0:
            self._seed = seed
        self._request_fresh_rollout(auto_step=True)
        return self._state_update()

    @override
    async def inference(self) -> AsyncGenerator[object, None]:
        """Generate one official Matrix iteration at a time and emit every RGB frame."""
        backend = self._backend
        config = self._config
        if backend is None or config is None:
            raise RuntimeError("Matrix-Game 3.0 was not loaded")

        while True:
            if self.state._restart_requested:
                selected = self._selected_input
                if selected is None:
                    yield Idle
                    continue
                prompt = self.state.prompt.strip()
                self.state._restart_requested = False
                await asyncio.to_thread(backend.reset, prompt, self._seed, selected)
                self._chunk_index = 0

            if self.state._limit_reached:
                yield Idle
                continue

            if self.state.paused and not self.state._step_requested:
                yield Idle
                continue

            self.state._step_requested = False
            action = action_from_controls(
                self.state._pressed_keys,
                self.state.pitch,
                self.state.yaw,
            )
            chunk_index = self._chunk_index
            frames = await asyncio.to_thread(backend.generate_chunk, action)
            frames = normalize_output_frames(frames, chunk_index)
            self._chunk_index += 1
            if self._chunk_index >= config.max_chunks:
                self.state._limit_reached = True
                self.state.paused = True
                self._clear_controls()
                await self.send(
                    RolloutLimitReached(
                        completed_chunks=self._chunk_index,
                        max_chunks=config.max_chunks,
                    )
                )
            await self.send(self._state_update())

            for frame in frames:
                if self.state._restart_requested:
                    break
                yield MatrixGame30Output(main_video=frame)

    def _request_fresh_rollout(self, *, auto_step: bool) -> None:
        """Queue a fresh upstream rollout and clear controls and progress."""
        self._clear_controls()
        self.state._restart_requested = True
        self.state._step_requested = auto_step
        self.state._limit_reached = False
        self._chunk_index = 0

    def _clear_controls(self) -> None:
        """Release native keyboard and camera conditions."""
        self.state._pressed_keys = frozenset()
        self.state.pitch = 0.0
        self.state.yaw = 0.0

    def _require_config(self) -> MatrixGame30Config:
        """Return the loaded configuration or fail before session mutation."""
        if self._config is None:
            raise RuntimeError("Matrix-Game 3.0 was not loaded")
        return self._config

    def _require_available_rollout(self) -> None:
        """Reject controls that cannot apply until a fresh rollout begins."""
        if self._selected_input is None:
            raise CommandError(
                "image_required",
                "Upload an image or select Random Image before controlling Matrix-Game 3.0.",
            )
        if self.state._limit_reached:
            raise CommandError(
                "rollout_limit_reached",
                "Start a fresh Matrix-Game 3.0 rollout before requesting another chunk.",
            )

    def _state_update(self) -> StateUpdate:
        """Return a complete client-facing snapshot of the shared world state."""
        config = self._config
        max_chunks = config.max_chunks if config is not None else 0
        selected = self._selected_input
        image_source = (
            "none"
            if selected is None
            else "upload"
            if isinstance(selected, UploadedFile)
            else "built_in"
        )
        next_chunk = (
            None
            if selected is None or self.state._limit_reached
            else self._chunk_index + 1
        )
        return StateUpdate(
            prompt=self.state.prompt,
            image_source=image_source,
            image_name=selected.name if selected is not None else "",
            seed=self._seed,
            paused=self.state.paused,
            step_queued=self.state._step_requested,
            restart_queued=self.state._restart_requested,
            limit_reached=self.state._limit_reached,
            completed_chunks=self._chunk_index,
            next_chunk=next_chunk,
            next_chunk_frames=(57 if self._chunk_index == 0 else 40)
            if next_chunk is not None
            else None,
            max_chunks=max_chunks,
            pressed_keys=self._ordered_pressed_keys(),
            pitch=self.state.pitch,
            yaw=self.state.yaw,
        )

    def _controls_changed(self, control: str) -> ControlsChanged:
        """Return the native controls held after one frontend command."""
        next_chunk = (
            None
            if self._selected_input is None or self.state._limit_reached
            else self._chunk_index + 1
        )
        return ControlsChanged(
            control=control,
            pressed_keys=self._ordered_pressed_keys(),
            pitch=self.state.pitch,
            yaw=self.state.yaw,
            applies_to_chunk=next_chunk,
        )

    def _ordered_pressed_keys(self) -> list[str]:
        """Return held keys in stable native W/S/A/D channel order."""
        return [key for key in ("w", "s", "a", "d") if key in self.state._pressed_keys]
