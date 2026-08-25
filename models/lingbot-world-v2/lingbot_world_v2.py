"""Serve public LingBot-World-V2 causal-fast inference through Reactor SDK.

The adapter keeps the released model's rolling self-attention KV, prompt
cross-attention KV, scheduler RNG, causal image-condition encoder state, and
causal decoder state across turns. Each inference turn runs one native
four-latent chunk and exposes the released camera-pose conditioning as held
six-axis controls.
"""

from __future__ import annotations

import asyncio
import importlib
import secrets
import time
from collections.abc import AsyncGenerator
from pathlib import Path
from typing import TYPE_CHECKING, Protocol

import numpy as np
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
    from .lingbot_world_v2_assets import BuiltInScene, LingBotConfig
    from .lingbot_world_v2_types import (
        CameraMotionChanged,
        ChunkCompleted,
        ImageSelected,
        LingBotWorldV2Output,
        LingBotWorldV2State,
        PauseChanged,
        PromptQueued,
        RolloutLimitReached,
        RolloutResetQueued,
        StateUpdate,
        StepQueued,
    )
else:
    module_prefix = f"{__package__}." if __package__ else ""
    assets_module = importlib.import_module(f"{module_prefix}lingbot_world_v2_assets")
    backend_module = importlib.import_module(f"{module_prefix}lingbot_world_v2_backend")
    camera_module = importlib.import_module(f"{module_prefix}lingbot_world_v2_camera")
    images_module = importlib.import_module(f"{module_prefix}lingbot_world_v2_images")
    types_module = importlib.import_module(f"{module_prefix}lingbot_world_v2_types")
    BuiltInScene = assets_module.BuiltInScene
    LingBotConfig = assets_module.LingBotConfig
    read_config = assets_module.read_config
    prepare_runtime_assets = assets_module.prepare_runtime_assets
    LingBotBackend = backend_module.LingBotBackend
    FIRST_CHUNK_FRAMES = backend_module.FIRST_CHUNK_FRAMES
    RGB_FPS = backend_module.RGB_FPS
    STEADY_CHUNK_FRAMES = backend_module.STEADY_CHUNK_FRAMES
    TEMPORAL_STRIDE = backend_module.TEMPORAL_STRIDE
    CameraMotionPlanner = camera_module.CameraMotionPlanner
    load_scene_camera = images_module.load_scene_camera
    validate_uploaded_image = images_module.validate_uploaded_image
    CameraMotionChanged = types_module.CameraMotionChanged
    ChunkCompleted = types_module.ChunkCompleted
    ImageSelected = types_module.ImageSelected
    LingBotWorldV2Output = types_module.LingBotWorldV2Output
    LingBotWorldV2State = types_module.LingBotWorldV2State
    PauseChanged = types_module.PauseChanged
    PromptQueued = types_module.PromptQueued
    RolloutLimitReached = types_module.RolloutLimitReached
    RolloutResetQueued = types_module.RolloutResetQueued
    StateUpdate = types_module.StateUpdate
    StepQueued = types_module.StepQueued

logger = get_logger(__name__)


class _Backend(Protocol):
    """Define blocking public-model operations used by the Reactor loop."""

    def reset(
        self,
        *,
        image: Path | UploadedFile,
        prompt: str,
        seed: int,
        intrinsics: np.ndarray,
    ) -> None:
        """Start a fresh native causal rollout."""

    def generate_chunk(self, *, prompt: str, relative_poses: np.ndarray) -> np.ndarray:
        """Generate one native causal chunk."""

    def end_session(self) -> None:
        """Release rollout state while retaining weights."""


class LingBotWorldV2(ReactorPipeline):
    """Generate an image-, prompt-, and camera-controllable LingBot world."""

    state: LingBotWorldV2State
    output: LingBotWorldV2Output
    fps = RGB_FPS
    buffer_size = 1

    def __init__(self) -> None:
        super().__init__()
        self._config: LingBotConfig | None = None
        self._backend: _Backend | None = None
        self._planner: CameraMotionPlanner | None = None
        self._selected_input: BuiltInScene | UploadedFile | None = None
        self._image_source = "none"
        self._seed = 0
        self._chunk_index = 0
        self._chunk_in_flight = False
        self._limit_reached = False

    def load(self, config_path: Path | None) -> None:
        """Prepare public assets and load LingBot weights once at startup.

        Args:
            config_path: Path to ``lingbot_world_v2.yaml`` from ``reactor.yaml``.
        """
        config = read_config(config_path)
        prepare_runtime_assets(config)
        self._config = config
        self._seed = config.seed
        self._planner = CameraMotionPlanner(
            fps=RGB_FPS,
            rotation_degrees_per_second=config.rotation_degrees_per_second,
        )
        self._backend = LingBotBackend(config)
        logger.info(
            "LingBot-World-V2 model ready",
            source_revision=config.source_revision,
            checkpoint_revision=config.checkpoint_revision,
            chunk_latents=config.chunk_latents,
            max_chunks=config.max_chunks,
        )

    @session_started
    def on_session_started(self) -> None:
        """Initialize an empty shared world before its first viewer connects."""
        config = self._require_loaded()
        self.state.prompt = ""
        self.state.paused = True
        self.state._step_requested = False
        self.state._reset_requested = False
        self._clear_camera()
        self._selected_input = None
        self._image_source = "none"
        self._seed = config.seed
        self._chunk_index = 0
        self._chunk_in_flight = False
        self._limit_reached = False

    @session_ended
    def on_session_ended(self) -> None:
        """Release causal rollout state when the shared session ends."""
        self._clear_camera()
        self.state._step_requested = False
        self.state._reset_requested = False
        self._selected_input = None
        self._image_source = "none"
        self._chunk_index = 0
        self._chunk_in_flight = False
        self._limit_reached = False
        if self._backend is not None:
            self._backend.end_session()

    @connected
    async def on_connected(self, client: ClientInfo) -> None:
        """Send the complete shared world state to one joining viewer."""
        await client.send(self._state_update())

    @disconnected
    async def on_disconnected(self) -> None:
        """Release held camera motion when a viewer disconnects."""
        self._clear_camera()
        await self.send(self._state_update())

    @event(
        name="set_prompt",
        description=(
            "Set the text condition without restarting the current causal world. Requires a "
            "selected image and an available chunk; the normalized text replaces cross-attention "
            "conditioning when the next chunk begins. Emits `prompt_queued` and broadcasts "
            "`state_update` on success, or `command_error` for empty text, a missing image, or "
            "an exhausted rollout."
        ),
    )
    async def set_prompt(
        self,
        prompt: str = InputField(
            default="",
            max_length=4096,
            description=(
                "Non-empty scene description, up to 4096 characters. Whitespace is trimmed and "
                "the result is sampled when the next chunk starts."
            ),
        ),
    ) -> PromptQueued:
        """Queue a prompt and report the chunk expected to consume it."""
        self._require_selected()
        self._require_available()
        normalized = prompt.strip()
        if not normalized:
            raise CommandError("prompt_required", "LingBot-World-V2 requires a prompt.")
        self.state.prompt = normalized
        message = PromptQueued(prompt=normalized, applies_to_chunk=self._next_chunk())
        await self.send(self._state_update())
        return message

    @event(
        name="set_camera",
        description=(
            "Set all six held camera axes atomically for forthcoming chunks. Requires a selected "
            "image and an available chunk; values are sampled together when the next chunk begins "
            "and remain active until changed or released. Emits `camera_motion_changed` and "
            "broadcasts `state_update` on success, or `command_error` when generation is not "
            "available."
        ),
    )
    async def set_camera(
        self,
        forward: float = InputField(
            default=0.0,
            ge=-1.0,
            le=1.0,
            description="Backward (-1) to forward (1) camera direction; zero is neutral.",
        ),
        strafe: float = InputField(
            default=0.0,
            ge=-1.0,
            le=1.0,
            description="Left (-1) to right (1) camera direction; zero is neutral.",
        ),
        vertical: float = InputField(
            default=0.0,
            ge=-1.0,
            le=1.0,
            description="Down (-1) to up (1) camera direction; zero is neutral.",
        ),
        pitch: float = InputField(
            default=0.0,
            ge=-1.0,
            le=1.0,
            description="Downward (-1) to upward (1) pitch rate; zero is neutral.",
        ),
        yaw: float = InputField(
            default=0.0,
            ge=-1.0,
            le=1.0,
            description="Left (-1) to right (1) yaw rate; zero is neutral.",
        ),
        roll: float = InputField(
            default=0.0,
            ge=-1.0,
            le=1.0,
            description="Counterclockwise (-1) to clockwise (1) roll rate; zero is neutral.",
        ),
    ) -> CameraMotionChanged:
        """Set all camera axes and report the complete held state."""
        self._require_selected()
        self._require_available()
        self.state._forward = forward
        self.state._strafe = strafe
        self.state._vertical = vertical
        self.state._pitch = pitch
        self.state._yaw = yaw
        self.state._roll = roll
        message = self._camera_changed()
        await self.send(self._state_update())
        return message

    @event(
        name="release_camera",
        description=(
            "Return every held camera axis to neutral for forthcoming chunks. Requires a selected "
            "image and an available chunk; neutral motion is sampled at the next boundary. Emits "
            "`camera_motion_changed` and broadcasts `state_update` on success, or `command_error` "
            "when generation is not available."
        ),
    )
    async def release_camera(self) -> CameraMotionChanged:
        """Release all camera motion and report the neutral held state."""
        self._require_selected()
        self._require_available()
        self._clear_camera()
        message = self._camera_changed()
        await self.send(self._state_update())
        return message

    @event(
        name="set_paused",
        description=(
            "Pause before the next chunk or resume continuous generation, releasing all held "
            "camera motion and canceling a queued step. Pausing is always valid; resuming requires "
            "a selected image and an available chunk. Emits `pause_changed` and broadcasts "
            "`state_update` on success, or `command_error` when resuming is unavailable."
        ),
    )
    async def set_paused(
        self,
        paused: bool = InputField(
            default=True,
            description=(
                "True pauses before the next chunk; false resumes continuous generation. Both "
                "values release all held camera axes and cancel a queued step."
            ),
        ),
    ) -> PauseChanged:
        """Set playback state, release camera motion, and report the result."""
        if not paused:
            self._require_selected()
            self._require_available()
        self.state.paused = paused
        self.state._step_requested = False
        self._clear_camera()
        message = PauseChanged(paused=paused, camera_motion_released=True)
        await self.send(self._state_update())
        return message

    @event(
        name="step",
        description=(
            "Queue exactly one chunk without leaving paused mode. Requires a selected image, "
            "`paused=true`, and an available chunk. Emits `step_queued` and broadcasts "
            "`state_update` on success, or `command_error` when any precondition is missing."
        ),
    )
    async def step(self) -> StepQueued:
        """Queue one paused chunk and report its rollout position."""
        self._require_selected()
        self._require_available()
        if not self.state.paused:
            raise CommandError(
                "pause_required", "Pause LingBot-World-V2 before stepping."
            )
        self.state._step_requested = True
        message = StepQueued(applies_to_chunk=self._next_chunk())
        await self.send(self._state_update())
        return message

    @event(
        name="reset",
        description=(
            "Queue a fresh causal rollout from the selected image and current prompt. Requires a "
            "selected image; the reset clears progress and the positional limit, releases camera "
            "motion, and preserves paused mode. Emits `rollout_reset_queued` and broadcasts "
            "`state_update` on success, or `command_error` when no image is selected."
        ),
    )
    async def reset(
        self,
        seed: int = InputField(
            default=-1,
            ge=-1,
            le=2_147_483_647,
            description=(
                "Seed for the fresh rollout from 0 through 2147483647. Use -1 to retain the "
                "active seed."
            ),
        ),
    ) -> RolloutResetQueued:
        """Queue a fresh rollout and report the state it replaces."""
        self._require_selected()
        if seed >= 0:
            self._seed = seed
        replaced = self._chunk_index
        self._request_reset()
        message = RolloutResetQueued(
            seed=self._seed,
            replaced_chunks=replaced,
            applies_to_chunk=1,
        )
        await self.send(self._state_update())
        return message

    @event(
        name="set_image",
        description=(
            "Select an uploaded anchor and queue a fresh rollout while preserving playback "
            "state. When paused, chunk one is queued automatically and playback remains paused "
            "after it finishes. Valid at any time; the image replaces prior causal state before "
            "chunk one. Emits `image_selected` and broadcasts `state_update` on success, or "
            "`command_error` when the upload is empty, oversized, mislabeled, or undecodable."
        ),
    )
    async def set_image(
        self,
        image: UploadedFile = InputField(  # noqa: B008 - schema field declaration
            description=(
                "Anchor uploaded through Reactor as JPEG, PNG, WebP, or BMP, up to 25 MiB and "
                "100 million pixels. EXIF orientation is applied before 480p resizing."
            )
        ),
        prompt: str = InputField(
            default="",
            max_length=4096,
            description=(
                "Optional prompt for the fresh rollout. An empty value retains the active prompt "
                "or uses the configured generic continuation prompt."
            ),
        ),
    ) -> ImageSelected:
        """Validate an uploaded image and select it for a fresh rollout."""
        try:
            validate_uploaded_image(image)
        except ValueError as error:
            raise CommandError("invalid_image", str(error)) from error
        config = self._require_loaded()
        self._selected_input = image
        self._image_source = "uploaded"
        self.state.prompt = (
            prompt.strip() or self.state.prompt.strip() or config.upload_default_prompt
        )
        self._request_reset()
        self._queue_initial_chunk_if_paused()
        message = ImageSelected(
            source="uploaded",
            filename=image.name,
            prompt=self.state.prompt,
            applies_to_chunk=1,
        )
        await self.send(self._state_update())
        return message

    @event(
        name="random_image",
        description=(
            "Select a public example image and its matching prompt, queue a fresh rollout, and "
            "preserve playback state. When paused, chunk one is queued automatically and playback "
            "remains paused after it finishes. Valid when configured examples are available. "
            "Emits `image_selected` and broadcasts `state_update` on success, or `command_error` "
            "when no usable example exists."
        ),
    )
    async def random_image(self) -> ImageSelected:
        """Select a different public example when possible."""
        config = self._require_loaded()
        if not config.scenes:
            raise CommandError(
                "image_unavailable", "No LingBot example images are configured."
            )
        candidates = [scene for scene in config.scenes if scene != self._selected_input]
        scene = secrets.choice(candidates or list(config.scenes))
        self._selected_input = scene
        self._image_source = "built_in"
        self.state.prompt = scene.prompt
        self._request_reset()
        self._queue_initial_chunk_if_paused()
        message = ImageSelected(
            source="built_in",
            filename=scene.image.name,
            prompt=scene.prompt,
            applies_to_chunk=1,
        )
        await self.send(self._state_update())
        return message

    async def inference(self) -> AsyncGenerator[object, None]:
        """Generate one native causal chunk per turn and emit it at 16 FPS."""
        config = self._require_loaded()
        backend = self._backend
        planner = self._planner
        if backend is None or planner is None:
            raise RuntimeError("LingBot-World-V2 was not loaded")

        while True:
            if self.state._reset_requested:
                selected = self._selected_input
                if selected is None:
                    yield Idle
                    continue
                if isinstance(selected, BuiltInScene):
                    image = selected.image
                    initial_pose, intrinsics = load_scene_camera(selected)
                else:
                    image = selected
                    initial_pose = np.eye(4, dtype=np.float32)
                    intrinsics = np.asarray(config.upload_intrinsics, dtype=np.float32)
                prompt = self.state.prompt.strip()
                if not prompt:
                    raise RuntimeError(
                        "LingBot-World-V2 requires a prompt before reset"
                    )
                self.state._reset_requested = False
                await asyncio.to_thread(
                    backend.reset,
                    image=image,
                    prompt=prompt,
                    seed=self._seed,
                    intrinsics=intrinsics,
                )
                planner.reset(initial_pose)
                self._chunk_index = 0
                self._limit_reached = False
                await self.send(self._state_update())

            if self._selected_input is None or self._limit_reached:
                yield Idle
                continue
            if self.state.paused and not self.state._step_requested:
                yield Idle
                continue

            self.state._step_requested = False
            sampled_prompt = self.state.prompt
            sampled_controls = {
                "forward": self.state._forward,
                "strafe": self.state._strafe,
                "vertical": self.state._vertical,
                "pitch": self.state._pitch,
                "yaw": self.state._yaw,
                "roll": self.state._roll,
            }
            relative_poses = planner.plan_chunk(
                **sampled_controls,
                latent_frames=config.chunk_latents,
                temporal_stride=TEMPORAL_STRIDE,
            )
            self._chunk_in_flight = True
            started = time.perf_counter()
            try:
                frames = await asyncio.to_thread(
                    backend.generate_chunk,
                    prompt=sampled_prompt,
                    relative_poses=relative_poses,
                )
            finally:
                self._chunk_in_flight = False
            seconds = time.perf_counter() - started
            self._chunk_index += 1
            if self._chunk_index >= config.max_chunks:
                self._limit_reached = True
                self.state.paused = True
                self._clear_camera()
            await self.send(
                ChunkCompleted(
                    chunk=self._chunk_index,
                    frames=int(frames.shape[0]),
                    generation_seconds=round(seconds, 3),
                    prompt=sampled_prompt,
                    **sampled_controls,
                )
            )
            if self._limit_reached:
                await self.send(
                    RolloutLimitReached(
                        completed_chunks=self._chunk_index,
                        max_chunks=config.max_chunks,
                    )
                )
            await self.send(self._state_update())

            for frame in frames:
                if self.state._reset_requested:
                    break
                yield LingBotWorldV2Output(main_video=frame)

    def _camera_changed(self) -> CameraMotionChanged:
        """Return the complete camera state for a successful mutation."""
        return CameraMotionChanged(
            forward=self.state._forward,
            strafe=self.state._strafe,
            vertical=self.state._vertical,
            pitch=self.state._pitch,
            yaw=self.state._yaw,
            roll=self.state._roll,
            applies_to_chunk=self._next_chunk(),
        )

    def _clear_camera(self) -> None:
        """Return every held camera axis to neutral."""
        self.state._forward = 0.0
        self.state._strafe = 0.0
        self.state._vertical = 0.0
        self.state._pitch = 0.0
        self.state._yaw = 0.0
        self.state._roll = 0.0

    def _request_reset(self) -> None:
        """Queue a fresh rollout and clear controls, progress, and limit state."""
        self._clear_camera()
        self.state._step_requested = False
        self.state._reset_requested = True
        self._chunk_index = 0
        self._limit_reached = False

    def _queue_initial_chunk_if_paused(self) -> None:
        """Queue chunk one so a paused image selection produces a visible preview."""
        if self.state.paused:
            self.state._step_requested = True

    def _require_loaded(self) -> LingBotConfig:
        """Return configuration or fail when startup did not complete."""
        if self._config is None:
            raise RuntimeError("LingBot-World-V2 was not loaded")
        return self._config

    def _require_selected(self) -> None:
        """Reject commands that need an anchor before one is selected."""
        if self._selected_input is None:
            raise CommandError(
                "image_required",
                "Upload an image or select a random image before this command.",
            )

    def _require_available(self) -> None:
        """Reject controls after the native temporal position limit."""
        if self._limit_reached:
            raise CommandError(
                "rollout_limit_reached",
                "Reset LingBot-World-V2 or select another image before continuing.",
            )

    def _next_chunk(self) -> int:
        """Return the one-based chunk expected to consume newly accepted input."""
        if self.state._reset_requested:
            return 1
        return self._chunk_index + 1 + int(self._chunk_in_flight)

    def _state_update(self) -> StateUpdate:
        """Return a complete snapshot of shared client-visible state."""
        config = self._config
        max_chunks = config.max_chunks if config is not None else 0
        selected = self._selected_input
        if isinstance(selected, BuiltInScene):
            image_name = selected.image.name
        elif isinstance(selected, UploadedFile):
            image_name = selected.name
        else:
            image_name = ""
        next_chunk = None if self._limit_reached else self._next_chunk()
        next_frames = None
        if next_chunk is not None:
            next_frames = FIRST_CHUNK_FRAMES if next_chunk == 1 else STEADY_CHUNK_FRAMES
        return StateUpdate(
            prompt=self.state.prompt or None,
            image_source=self._image_source,
            image_name=image_name,
            seed=self._seed,
            paused=self.state.paused,
            step_queued=self.state._step_requested,
            reset_queued=self.state._reset_requested,
            completed_chunks=self._chunk_index,
            next_chunk=next_chunk,
            next_chunk_frames=next_frames,
            max_chunks=max_chunks,
            limit_reached=self._limit_reached,
            forward=self.state._forward,
            strafe=self.state._strafe,
            vertical=self.state._vertical,
            pitch=self.state._pitch,
            yaw=self.state._yaw,
            roll=self.state._roll,
        )
