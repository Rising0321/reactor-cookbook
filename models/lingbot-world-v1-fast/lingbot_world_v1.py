"""Serve LingBot-World v1 Fast through Reactor's interactive pipeline API."""

from __future__ import annotations

import importlib
import random
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
    from camera_motion import CameraMotionPlanner, MotionConfig
    from lingbot_config import LingBotConfig, Sample
    from lingbot_schema import (
        LingBotWorldOutput,
        LingBotWorldState,
        RolloutLimitReached,
        StateUpdate,
    )
else:
    module_prefix = f"{__package__}." if __package__ else ""
    camera_motion = importlib.import_module(f"{module_prefix}camera_motion")
    lingbot_config = importlib.import_module(f"{module_prefix}lingbot_config")
    lingbot_images = importlib.import_module(f"{module_prefix}lingbot_images")
    lingbot_schema = importlib.import_module(f"{module_prefix}lingbot_schema")
    upstream_backend = importlib.import_module(f"{module_prefix}upstream_backend")
    CameraMotionPlanner = camera_motion.CameraMotionPlanner
    MotionConfig = camera_motion.MotionConfig
    LingBotConfig = lingbot_config.LingBotConfig
    Sample = lingbot_config.Sample
    prepare_runtime = lingbot_config.prepare_runtime
    read_config = lingbot_config.read_config
    normalize_output_frames = lingbot_images.normalize_output_frames
    validate_uploaded_image = lingbot_images.validate_uploaded_image
    LingBotWorldOutput = lingbot_schema.LingBotWorldOutput
    LingBotWorldState = lingbot_schema.LingBotWorldState
    RolloutLimitReached = lingbot_schema.RolloutLimitReached
    StateUpdate = lingbot_schema.StateUpdate
    LingBotWorkerBackend = upstream_backend.LingBotWorkerBackend
    WorkerSettings = upstream_backend.WorkerSettings

logger = get_logger(__name__)

FPS = 16
FRAMES_PER_CHUNK = 12


class _Backend(Protocol):
    """Define the blocking model operations used by the Reactor loop."""

    def reset(
        self,
        seed: int,
        anchor_image: Path | UploadedFile,
        intrinsics: Path,
        prompt: str,
    ) -> None:
        """Start a fresh image-conditioned rollout."""

    def generate_chunk(self, relative_c2ws: np.ndarray, prompt: str) -> np.ndarray:
        """Generate one causal chunk for camera and text conditions."""

    def end_session(self) -> None:
        """Release state owned by the completed session."""


class LingBotWorldV1(ReactorPipeline):
    """Generate an image-, prompt-, and camera-controllable LingBot world."""

    state: LingBotWorldState
    output: LingBotWorldOutput
    buffer_size = FRAMES_PER_CHUNK

    def __init__(self) -> None:
        super().__init__()
        self._config: LingBotConfig | None = None
        self._backend: _Backend | None = None
        self._planner: CameraMotionPlanner | None = None
        self._selected_input: Path | UploadedFile | None = None
        self._selected_intrinsics: Path | None = None
        self._default_prompt = ""
        self._seed = 0
        self._chunk_index = 0
        self._chunk_in_flight = False
        self._last_chunk_seconds: float | None = None

    def load(self, config_path: Path | None) -> None:
        """Prepare public assets and load LingBot-World-Fast once.

        Args:
            config_path: Path to ``lingbot_world_v1.yaml`` from ``reactor.yaml``.
        """
        config = read_config(config_path)
        prepare_runtime(config)
        self._config = config
        self._default_prompt = config.samples[0].prompt
        self._seed = config.seed
        self._planner = CameraMotionPlanner(
            MotionConfig(
                translation_units_per_latent=config.translation_units_per_latent,
                rotation_degrees_per_latent=config.rotation_degrees_per_latent,
            )
        )
        self._backend = LingBotWorkerBackend(
            WorkerSettings(
                python_executable=config.worker_python,
                source_path=config.source_path,
                checkpoint_dir=config.checkpoint.path,
                runtime_root=config.runtime_root,
                max_chunks=config.max_chunks,
                context_latents=config.context_latents,
                max_area=config.max_area,
                shift=config.shift,
            )
        )
        logger.info(
            "LingBot-World v1 Fast ready",
            source_revision=config.source_revision,
            fast_revision=config.fast_checkpoint.revision,
            context_latents=config.context_latents,
            fps=FPS,
        )

    @session_started
    def on_session_started(self) -> None:
        """Initialize one unpaused shared world and queue its first chunk."""
        config = self._require_config()
        sample = config.samples[0]
        self._select_sample(sample)
        self._seed = config.seed
        self._clear_controls()
        self.state.paused = False
        self.state._step_requested = True
        self.state._restart_requested = True
        self.state._limit_reached = False
        self._chunk_index = 0
        self._chunk_in_flight = False
        self._last_chunk_seconds = None

    @connected
    async def on_connected(self, client: ClientInfo) -> None:
        """Send the complete shared world state to one joining viewer."""
        await client.send(self._state_update())

    @disconnected
    async def on_disconnected(self) -> None:
        """Release held camera axes when their viewer disconnects."""
        self._clear_controls()
        await self.send(self._state_update())

    @session_ended
    def on_session_ended(self) -> None:
        """Release causal caches while retaining loaded weights."""
        backend = self._backend
        try:
            if backend is not None:
                backend.end_session()
        finally:
            self._clear_controls()
            self.state._step_requested = False
            self.state._restart_requested = True
            self.state._limit_reached = False
            self._selected_input = None
            self._selected_intrinsics = None
            self._chunk_index = 0
            self._chunk_in_flight = False
            self._last_chunk_seconds = None

    @event(
        name="set_image",
        description=(
            "Replace the anchor image and queue the fresh world's first `main_video` chunk. "
            "Valid at any time; generation runs continuously, progress and camera axes reset, and "
            "the upload is decoded before the command succeeds. Returns `state_update`, or "
            "`command_error` for invalid image bytes, media type, dimensions, or an empty prompt."
        ),
    )
    def set_image(
        self,
        image: UploadedFile = InputField(  # noqa: B008 - schema field declaration
            description=(
                "Anchor image delivered through Reactor's upload protocol. JPEG, PNG, WebP, or "
                "BMP; non-empty, at most 25 MiB and 100 million pixels."
            )
        ),
        prompt: str = InputField(
            default="",
            max_length=4096,
            description=(
                "Optional scene prompt up to 4096 characters. A non-empty value conditions the "
                "preview chunk; an empty value preserves the active prompt."
            ),
        ),
    ) -> StateUpdate:
        """Select an upload and begin a fresh continuous rollout."""
        validate_uploaded_image(image)
        normalized = prompt.strip() or self.state.prompt.strip() or self._default_prompt
        if not normalized:
            raise CommandError("prompt_required", "LingBot-World requires a prompt.")
        self._selected_input = image
        self.state.prompt = normalized
        self.state.paused = False
        self._request_restart(auto_step=True)
        return self._state_update()

    @event(
        name="random_image",
        description=(
            "Select one public LingBot demo image and its matching prompt and calibration, then "
            "queue the fresh world's first `main_video` chunk. Valid at any time; generation "
            "runs continuously and progress resets. Returns the complete `state_update`."
        ),
    )
    def random_image(self) -> StateUpdate:
        """Select a built-in sample and begin a fresh continuous rollout."""
        sample = random.choice(self._require_config().samples)
        self._select_sample(sample)
        self.state.paused = False
        self._request_restart(auto_step=True)
        return self._state_update()

    @event(
        name="set_prompt",
        description=(
            "Set text conditioning for the next generated chunk without clearing visual self-KV "
            "history. Valid after an anchor is selected and before the rollout limit. Returns "
            "`state_update`, or `command_error` when the prompt is empty, no image is selected, "
            "or a fresh rollout is required."
        ),
    )
    def set_prompt(
        self,
        prompt: str = InputField(
            default="",
            max_length=4096,
            description=(
                "Non-empty scene description up to 4096 characters. It replaces the active "
                "cross-attention context at the next chunk boundary while self-KV history stays."
            ),
        ),
    ) -> StateUpdate:
        """Queue a prompt change and return the complete shared state."""
        normalized = prompt.strip()
        if not normalized:
            raise CommandError("prompt_required", "LingBot-World requires a prompt.")
        if self._selected_input is None:
            raise CommandError(
                "image_required", "Select an anchor image before setting a prompt."
            )
        self._require_available_rollout()
        self.state.prompt = normalized
        return self._state_update()

    @event(
        name="set_forward",
        description=(
            "Set backward-to-forward camera translation for the next chunk. Valid before the "
            "rollout limit; the value is held for later chunks. Returns `state_update`, or "
            "`command_error` until a fresh rollout is started after the limit."
        ),
    )
    def set_forward(
        self,
        forward: float = InputField(
            default=0.0,
            ge=-1.0,
            le=1.0,
            description=(
                "Normalized backward (-1) to forward (1) translation. Zero stops this axis; the "
                "value is sampled at the next chunk boundary and held."
            ),
        ),
    ) -> StateUpdate:
        """Set forward motion and return the complete shared state."""
        return self._set_axis("forward", forward)

    @event(
        name="set_strafe",
        description=(
            "Set left-to-right camera translation for the next chunk. Valid before the rollout "
            "limit; the value is held for later chunks. Returns `state_update`, or "
            "`command_error` until a fresh rollout is started after the limit."
        ),
    )
    def set_strafe(
        self,
        strafe: float = InputField(
            default=0.0,
            ge=-1.0,
            le=1.0,
            description=(
                "Normalized left (-1) to right (1) translation. Zero stops this axis; the value "
                "is sampled at the next chunk boundary and held."
            ),
        ),
    ) -> StateUpdate:
        """Set strafe motion and return the complete shared state."""
        return self._set_axis("strafe", strafe)

    @event(
        name="set_vertical",
        description=(
            "Set down-to-up camera translation for the next chunk. Valid before the rollout "
            "limit; the value is held for later chunks. Returns `state_update`, or "
            "`command_error` until a fresh rollout is started after the limit."
        ),
    )
    def set_vertical(
        self,
        vertical: float = InputField(
            default=0.0,
            ge=-1.0,
            le=1.0,
            description=(
                "Normalized down (-1) to up (1) translation. Zero stops this axis; the value is "
                "sampled at the next chunk boundary and held."
            ),
        ),
    ) -> StateUpdate:
        """Set vertical motion and return the complete shared state."""
        return self._set_axis("vertical", vertical)

    @event(
        name="set_pitch",
        description=(
            "Set downward-to-upward camera pitch for the next chunk. Valid before the rollout "
            "limit; the value is held for later chunks. Returns `state_update`, or "
            "`command_error` until a fresh rollout is started after the limit."
        ),
    )
    def set_pitch(
        self,
        pitch: float = InputField(
            default=0.0,
            ge=-1.0,
            le=1.0,
            description=(
                "Normalized downward (-1) to upward (1) pitch. Zero stops this axis; the value "
                "is sampled at the next chunk boundary and held."
            ),
        ),
    ) -> StateUpdate:
        """Set pitch and return the complete shared state."""
        return self._set_axis("pitch", pitch)

    @event(
        name="set_yaw",
        description=(
            "Set left-to-right camera yaw for the next chunk. Valid before the rollout limit; "
            "the value is held for later chunks. Returns `state_update`, or `command_error` "
            "until a fresh rollout is started after the limit."
        ),
    )
    def set_yaw(
        self,
        yaw: float = InputField(
            default=0.0,
            ge=-1.0,
            le=1.0,
            description=(
                "Normalized left (-1) to right (1) yaw. Zero stops this axis; the value is "
                "sampled at the next chunk boundary and held."
            ),
        ),
    ) -> StateUpdate:
        """Set yaw and return the complete shared state."""
        return self._set_axis("yaw", yaw)

    @event(
        name="set_roll",
        description=(
            "Set counterclockwise-to-clockwise camera roll for the next chunk. Valid before the "
            "rollout limit; the value is held for later chunks. Returns `state_update`, or "
            "`command_error` until a fresh rollout is started after the limit."
        ),
    )
    def set_roll(
        self,
        roll: float = InputField(
            default=0.0,
            ge=-1.0,
            le=1.0,
            description=(
                "Normalized counterclockwise (-1) to clockwise (1) roll. Zero stops this axis; "
                "the value is sampled at the next chunk boundary and held."
            ),
        ),
    ) -> StateUpdate:
        """Set roll and return the complete shared state."""
        return self._set_axis("roll", roll)

    # Keep pause available for future schema re-enablement without exposing it to clients.
    def set_paused(
        self,
        paused: bool = InputField(
            default=False,
            description=(
                "True waits before the next chunk; false generates continuously. The change "
                "takes effect after an in-flight chunk and releases all six camera axes."
            ),
        ),
    ) -> StateUpdate:
        """Set pause state and return the complete shared state."""
        if not paused:
            self._require_available_rollout()
        self.state.paused = paused
        self.state._step_requested = False
        self._clear_controls()
        return self._state_update()

    # Keep single-step generation available for future schema re-enablement.
    def step(self) -> StateUpdate:
        """Queue one paused chunk and return the complete shared state."""
        self._require_available_rollout()
        if not self.state.paused:
            raise CommandError("pause_required", "Pause LingBot-World before stepping.")
        self.state._step_requested = True
        return self._state_update()

    @event(
        name="reset",
        description=(
            "Restart from the selected image and prompt and queue one preview chunk. Valid when "
            "an anchor exists; progress, caches, and camera axes reset while paused mode is "
            "preserved. Returns `state_update`, or `command_error` when no image is selected."
        ),
    )
    def reset(
        self,
        seed: int = InputField(
            default=-1,
            ge=-1,
            le=2_147_483_647,
            description=(
                "Seed from 0 to 2147483647 for the fresh rollout. Use -1 to retain the active "
                "seed; a non-negative value becomes active when reset begins."
            ),
        ),
    ) -> StateUpdate:
        """Queue a reproducible fresh rollout and one paused preview chunk."""
        if self._selected_input is None:
            raise CommandError(
                "image_required", "Select an anchor image before resetting."
            )
        if seed >= 0:
            self._seed = seed
        self._request_restart(auto_step=self.state.paused)
        return self._state_update()

    async def inference(self) -> AsyncGenerator[object, None]:
        """Generate and emit one native chunk per request."""
        backend = self._backend
        planner = self._planner
        config = self._require_config()
        if backend is None or planner is None:
            raise RuntimeError("LingBot-World v1 was not loaded")
        while True:
            if self.state._restart_requested:
                selected = self._selected_input
                intrinsics = self._selected_intrinsics
                if selected is None or intrinsics is None:
                    yield Idle
                    continue
                prompt = self.state.prompt.strip()
                if not prompt:
                    raise RuntimeError("LingBot-World requires a non-empty prompt")
                self.state._restart_requested = False
                backend.reset(
                    self._seed,
                    selected,
                    intrinsics,
                    prompt,
                )
                if self.state._restart_requested:
                    continue
                planner.reset()
                self._chunk_index = 0

            if self.state._limit_reached:
                yield Idle
                continue
            if self.state.paused and not self.state._step_requested:
                yield Idle
                continue

            self.state._step_requested = False
            camera = planner.plan_chunk(
                strafe=self.state.strafe,
                vertical=self.state.vertical,
                forward=self.state.forward,
                pitch=self.state.pitch,
                yaw=self.state.yaw,
                roll=self.state.roll,
            )
            self._chunk_in_flight = True
            started = time.perf_counter()
            try:
                frames = backend.generate_chunk(
                    camera,
                    self.state.prompt,
                )
            finally:
                self._chunk_in_flight = False
            self._last_chunk_seconds = time.perf_counter() - started
            frames = normalize_output_frames(frames)
            if self.state._restart_requested:
                continue
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
            yield LingBotWorldOutput(main_video=frames)

    def _set_axis(self, name: str, value: float) -> StateUpdate:
        """Set one validated axis and return a complete state snapshot."""
        self._require_available_rollout()
        setattr(self.state, name, value)
        return self._state_update()

    def _select_sample(self, sample: Sample) -> None:
        """Make one public sample the active image, calibration, and prompt."""
        self._selected_input = sample.image
        self._selected_intrinsics = sample.intrinsics
        self.state.prompt = sample.prompt

    def _clear_controls(self) -> None:
        """Return every camera axis to neutral."""
        self.state.forward = 0.0
        self.state.strafe = 0.0
        self.state.vertical = 0.0
        self.state.pitch = 0.0
        self.state.yaw = 0.0
        self.state.roll = 0.0

    def _request_restart(self, *, auto_step: bool) -> None:
        """Queue a fresh causal rollout and release active camera motion."""
        self._clear_controls()
        self.state._step_requested = auto_step
        self.state._restart_requested = True
        self.state._limit_reached = False
        self._chunk_index = 0
        self._last_chunk_seconds = None
        self.output.flush()

    def _require_available_rollout(self) -> None:
        """Reject controls that cannot apply until a fresh rollout starts."""
        if self.state._limit_reached:
            raise CommandError(
                "rollout_limit_reached",
                "Reset LingBot-World or select an image before requesting another chunk.",
            )

    def _require_config(self) -> LingBotConfig:
        """Return loaded configuration or fail with a lifecycle error."""
        if self._config is None:
            raise RuntimeError("LingBot-World v1 was not loaded")
        return self._config

    def _next_control_chunk(self) -> int:
        """Return the one-based chunk expected to consume newly accepted controls."""
        if self.state._restart_requested:
            return 1
        return self._chunk_index + 1 + int(self._chunk_in_flight)

    def _state_update(self) -> StateUpdate:
        """Return a complete client-facing snapshot of shared world state."""
        config = self._config
        max_chunks = config.max_chunks if config is not None else 0
        selected = self._selected_input
        image_source = "upload" if isinstance(selected, UploadedFile) else "built_in"
        image_name = selected.name if selected is not None else ""
        if self.state._limit_reached:
            next_chunk = None
            next_chunk_frames = None
        else:
            next_chunk = self._next_control_chunk()
            next_chunk_frames = 9 if next_chunk == 1 else 12
        return StateUpdate(
            prompt=self.state.prompt,
            image_source=image_source,
            image_name=image_name,
            seed=self._seed,
            paused=self.state.paused,
            step_queued=self.state._step_requested,
            limit_reached=self.state._limit_reached,
            completed_chunks=self._chunk_index,
            last_chunk_seconds=self._last_chunk_seconds,
            next_chunk=next_chunk,
            next_chunk_frames=next_chunk_frames,
            max_chunks=max_chunks,
            forward=self.state.forward,
            strafe=self.state.strafe,
            vertical=self.state.vertical,
            pitch=self.state.pitch,
            yaw=self.state.yaw,
            roll=self.state.roll,
        )
