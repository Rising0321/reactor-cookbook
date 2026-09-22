"""Serve Matrix-Game-2.0 distilled autoregressive inference through Reactor."""

from __future__ import annotations

import secrets
import time
from dataclasses import dataclass
from pathlib import Path

import numpy as np
from reactor_runtime import (
    ApplicationError,
    ClientInfo,
    CommandError,
    InputField,
    ReactorApp,
    StepOutcome,
    UploadedFile,
    connected,
    disconnected,
    event,
    session_ended,
    session_started,
)
from reactor_runtime.log import get_logger

from matrix_game_2_assets import (
    load_input_image,
    prepare_runtime_assets,
    read_config,
    validate_uploaded_image,
)
from matrix_game_2_backend import FRAMES_PER_CHUNK, ChunkAction, MatrixGame2Backend
from matrix_game_2_types import (
    KEYBOARD_KEYS,
    ActionChanged,
    CameraMotionChanged,
    ChunkComplete,
    MatrixGame2Config,
    MatrixGame2Output,
    MatrixGame2State,
    RolloutLimitReached,
    StateUpdate,
)

logger = get_logger(__name__)


@dataclass(frozen=True)
class MatrixGame2Input:
    """Snapshot image conditioning and one chunk's controls."""

    anchor_image: Path | UploadedFile
    seed: int
    restart: bool
    action: ChunkAction


@dataclass(frozen=True)
class MatrixGame2Result:
    """Carry generated frames and the controls actually consumed."""

    frames: np.ndarray
    seconds: float
    action: ChunkAction


class MatrixGame2(ReactorApp):
    """Generate a keyboard- and mouse-camera-controlled Matrix world from an image."""

    state: MatrixGame2State
    buffer_size = FRAMES_PER_CHUNK

    def __init__(self) -> None:
        super().__init__()
        self._config: MatrixGame2Config | None = None
        self._backend: MatrixGame2Backend | None = None
        self._selected_input: Path | UploadedFile | None = None
        self._image_source = "none"
        self._seed = 0
        self._chunk_index = 0
        self._last_chunk_frames = 0

    def load(self, config_path: Path | None) -> None:
        """Load the pinned universal distilled checkpoint once.

        Args:
            config_path: Path to ``matrix_game_2.yaml`` from ``reactor.yaml``.
        """
        config = read_config(config_path)
        model_path = prepare_runtime_assets(config)
        backend = MatrixGame2Backend(
            source_path=config.source_path,
            model_path=model_path,
            checkpoint_file=config.checkpoint_file,
            max_latent_frames=config.max_latent_frames,
        )
        self._config = config
        self._backend = backend
        self._seed = config.seed
        logger.info(
            "Matrix-Game-2.0 model ready",
            source_revision=config.source_revision,
            checkpoint_revision=config.model_revision,
            max_chunks=config.max_chunks,
        )

    @session_started
    def on_session_started(self) -> None:
        """Initialize an empty shared world before its first viewer connects."""
        config = self._require_config()
        self._selected_input = None
        self._image_source = "none"
        self._seed = config.seed
        self.state._restart_requested = False
        self.state._limit_reached = False
        self._chunk_index = 0
        self._last_chunk_frames = 0
        self._clear_controls()

    @connected
    async def on_connected(self, client: ClientInfo) -> None:
        """Send the complete shared world state to one joining viewer."""
        await client.send(self._state_update())

    @disconnected
    async def on_disconnected(self) -> None:
        """Release held controls when their viewer leaves the live session."""
        self._clear_controls()
        await self.send(self._state_update())

    @session_ended
    def on_session_ended(self) -> None:
        """Release image data and causal caches owned by the completed session."""
        backend = self._backend
        try:
            if backend is not None:
                backend.end_rollout()
        finally:
            self._selected_input = None
            self._image_source = "none"
            self.state._restart_requested = False
            self.state._limit_reached = False
            self._chunk_index = 0
            self._last_chunk_frames = 0
            self._clear_controls()

    @event(
        name="set_image",
        description=(
            "Select an uploaded starting image and initialize a fresh autoregressive world. "
            "Valid at any time; the image is center-cropped to Matrix's 352x640 input, clears "
            "the prior rollout and controls, and starts continuous `main_video` generation. "
            "Returns `StateUpdate` on success, or `CommandError` when the "
            "upload is not a valid JPEG, PNG, WebP, or BMP within the size limits."
        ),
    )
    def set_image(
        self,
        image: UploadedFile = InputField(  # noqa: B008 - schema field declaration
            moderate=True,
            description=(
                "Starting image sent through Reactor's upload protocol. JPEG, PNG, WebP, or "
                "BMP; at most 25 MiB and 100 million decoded pixels. Selection starts a fresh "
                "rollout and automatically generates its first chunk."
            ),
        ),
    ) -> StateUpdate:
        """Select uploaded image bytes and return the queued world state."""
        validate_uploaded_image(image)
        self._selected_input = image
        self._image_source = "uploaded"
        self._request_restart()
        return self._state_update()

    @event(
        name="random_image",
        description=(
            "Select one configured public Matrix universal example and initialize a fresh "
            "autoregressive world. Valid at any time; it chooses a different built-in image "
            "when possible, clears the prior rollout and controls, and starts continuous "
            "`main_video` generation. Returns `StateUpdate` on success."
        ),
    )
    def random_image(self) -> StateUpdate:
        """Select a public example image and return the queued world state."""
        config = self._require_config()
        choices = list(config.random_images)
        if isinstance(self._selected_input, Path) and len(choices) > 1:
            choices = [path for path in choices if path != self._selected_input]
        self._selected_input = secrets.choice(choices)
        self._image_source = "built_in"
        self._request_restart()
        return self._state_update()

    @event(
        name="set_key_state",
        description=(
            "Hold or release one WASD movement key for forthcoming chunks. Requires a selected "
            "image; the complete held-key set is sampled as the official four-value multi-hot "
            "condition when the next chunk starts. Emits `action_changed` and broadcasts "
            "`state_update` on success, or `command_error` when a reset is required."
        ),
    )
    async def set_key_state(
        self,
        key: str = InputField(
            default="w",
            choices=KEYBOARD_KEYS,
            description=(
                "Movement key to hold or release: W forward, S backward, A left, or D right. "
                "Held keys can be combined, including W+A and W+D."
            ),
        ),
        pressed: bool = InputField(
            default=True,
            description=(
                "Set true on key-down and false on key-up. The resulting held-key set persists "
                "until another key event or `release_controls`."
            ),
        ),
    ) -> ActionChanged:
        """Update one held WASD key and report the complete discrete action."""
        self._require_playable_rollout()
        if pressed:
            self.state._pressed_keys = self.state._pressed_keys.union((key,))
        else:
            self.state._pressed_keys = self.state._pressed_keys.difference((key,))
        message = ActionChanged(
            key=key,
            pressed=pressed,
            pressed_keys=sorted(self.state._pressed_keys),
            applies_to_chunk=self._next_control_chunk(),
        )
        await self.send(self._state_update())
        return message

    @event(
        name="set_pitch",
        description=(
            "Set continuous look-down or look-up camera velocity for forthcoming chunks. "
            "Requires a selected image; the normalized value is sampled at the next chunk "
            "boundary and mapped to Matrix's native vertical mouse condition. Emits "
            "`camera_motion_changed` and broadcasts `state_update` on success."
        ),
    )
    async def set_pitch(
        self,
        pitch: float = InputField(
            default=0.0,
            ge=-1.0,
            le=1.0,
            description=(
                "Look down (-1) to look up (1) velocity held until changed; zero is neutral."
            ),
        ),
    ) -> CameraMotionChanged:
        """Queue normalized pitch and return the complete camera action."""
        self._require_playable_rollout()
        self.state.pitch = pitch
        message = self._camera_motion_changed()
        await self.send(self._state_update())
        return message

    @event(
        name="set_yaw",
        description=(
            "Set continuous turn-left or turn-right camera velocity for forthcoming chunks. "
            "Requires a selected image; the normalized value is sampled at the next chunk "
            "boundary and mapped to Matrix's native horizontal mouse condition. Emits "
            "`camera_motion_changed` and broadcasts `state_update` on success."
        ),
    )
    async def set_yaw(
        self,
        yaw: float = InputField(
            default=0.0,
            ge=-1.0,
            le=1.0,
            description=(
                "Turn left (-1) to turn right (1) velocity held until changed; zero is neutral."
            ),
        ),
    ) -> CameraMotionChanged:
        """Queue normalized yaw and return the complete camera action."""
        self._require_playable_rollout()
        self.state.yaw = yaw
        message = self._camera_motion_changed()
        await self.send(self._state_update())
        return message

    @event(
        name="release_controls",
        description=(
            "Return keyboard and camera conditions to neutral. Valid at any time; neutral "
            "values are sampled by the next generated chunk without resetting visual history. "
            "Returns `StateUpdate` with all three controls cleared."
        ),
    )
    def release_controls(self) -> StateUpdate:
        """Clear held keyboard and camera controls and return shared state."""
        self._clear_controls()
        return self._state_update()

    @event(
        name="reset",
        description=(
            "Restart from the selected image with empty model, keyboard, mouse, cross-attention, "
            "and causal VAE caches. Valid after image selection; it resumes continuous generation, "
            "clears controls and progress, and automatically queues one visible chunk. Returns "
            "`StateUpdate` on success, or `CommandError` when no image is selected."
        ),
    )
    def reset(
        self,
        seed: int = InputField(
            default=-1,
            ge=-1,
            le=2_147_483_647,
            description=(
                "Seed for the fresh rollout from 0 to 2147483647. Use -1 to retain the active "
                "seed; a non-negative value becomes active when reset begins."
            ),
        ),
    ) -> StateUpdate:
        """Queue a cache reset and return the complete shared world state."""
        if self._selected_input is None:
            raise CommandError(
                "image_required", "Select a Matrix image before resetting."
            )
        if seed >= 0:
            self._seed = seed
        self._request_restart()
        return self._state_update()

    async def process_input(self) -> MatrixGame2Input:
        """Snapshot an available rollout at its next chunk boundary."""
        if self._selected_input is None:
            raise ApplicationError("Select an anchor image before generating.")
        if self.state._limit_reached:
            raise ApplicationError("Reset the world after reaching the rollout limit.")
        return MatrixGame2Input(
            anchor_image=self._selected_input,
            seed=self._seed,
            restart=self.state._restart_requested,
            action=ChunkAction(
                pressed_keys=tuple(sorted(self.state._pressed_keys)),
                pitch=self.state.pitch,
                yaw=self.state.yaw,
            ),
        )

    def generate(self, input: MatrixGame2Input) -> MatrixGame2Result:
        """Generate one official causal chunk from an immutable input snapshot."""
        backend = self._require_backend()
        if input.restart:
            backend.reset(load_input_image(input.anchor_image), input.seed)
        started = time.perf_counter()
        frames = backend.generate_chunk(input.action)
        return MatrixGame2Result(frames, time.perf_counter() - started, input.action)

    async def process_output(self, outcome: StepOutcome) -> MatrixGame2Output:
        """Publish the completed chunk and advance successful rollout progress."""
        if outcome.error is not None:
            raise outcome.error
        result: MatrixGame2Result = outcome.result
        self.state._restart_requested = False
        self._chunk_index += 1
        self._last_chunk_frames = int(result.frames.shape[0])
        config = self._require_config()
        if self._chunk_index >= config.max_chunks:
            self.state._limit_reached = True
            self._clear_controls()
            await self.send(
                RolloutLimitReached(
                    completed_chunks=self._chunk_index, max_chunks=config.max_chunks
                )
            )
        await self.send(
            ChunkComplete(
                chunk=self._chunk_index,
                frames=self._last_chunk_frames,
                inference_seconds=result.seconds,
                pressed_keys=list(result.action.pressed_keys),
                pitch=result.action.pitch,
                yaw=result.action.yaw,
            )
        )
        await self.send(self._state_update())
        return MatrixGame2Output(main_video=result.frames)

    def _request_restart(self) -> None:
        """Queue fresh image conditioning and release current controls and progress."""
        self.output.flush()
        self._clear_controls()
        self.state._restart_requested = True
        self.state._limit_reached = False
        self._chunk_index = 0
        self._last_chunk_frames = 0

    def _clear_controls(self) -> None:
        """Return the universal keyboard and mouse-camera conditions to neutral."""
        self.state._pressed_keys = frozenset()
        self.state.pitch = 0.0
        self.state.yaw = 0.0

    def _camera_motion_changed(self) -> CameraMotionChanged:
        """Describe the complete continuous camera state after an axis event."""
        return CameraMotionChanged(
            pitch=self.state.pitch,
            yaw=self.state.yaw,
            applies_to_chunk=self._next_control_chunk(),
        )

    def _require_playable_rollout(self) -> None:
        """Reject controls that cannot affect a generated chunk."""
        if self._selected_input is None:
            raise CommandError("image_required", "Select a Matrix image first.")
        if self.state._limit_reached:
            raise CommandError(
                "rollout_limit_reached",
                "Reset Matrix-Game-2.0 before requesting another chunk.",
            )

    def _next_control_chunk(self) -> int:
        """Return the one-based chunk expected to sample controls accepted now."""
        if self.state._restart_requested:
            return 1
        return self._chunk_index + 1

    def _state_update(self) -> StateUpdate:
        """Return a complete client-facing snapshot of the shared world state."""
        config = self._config
        max_chunks = config.max_chunks if config is not None else 0
        selected = self._selected_input
        return StateUpdate(
            image_source=self._image_source,
            image_name=selected.name if selected is not None else "",
            seed=self._seed,
            reset_queued=self.state._restart_requested,
            chunk_in_flight=False,
            limit_reached=self.state._limit_reached,
            completed_chunks=self._chunk_index,
            next_chunk=None
            if selected is None or self.state._limit_reached
            else self._next_control_chunk(),
            max_chunks=max_chunks,
            last_chunk_frames=self._last_chunk_frames,
            pressed_keys=sorted(self.state._pressed_keys),
            pitch=self.state.pitch,
            yaw=self.state.yaw,
        )

    def _require_config(self) -> MatrixGame2Config:
        """Return loaded configuration or raise a lifecycle error."""
        if self._config is None:
            raise RuntimeError("Matrix-Game-2.0 was not loaded")
        return self._config

    def _require_backend(self) -> MatrixGame2Backend:
        """Return the loaded upstream backend or raise a lifecycle error."""
        if self._backend is None:
            raise RuntimeError("Matrix-Game-2.0 was not loaded")
        return self._backend
