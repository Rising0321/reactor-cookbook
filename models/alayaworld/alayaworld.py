"""Serve AlayaWorld v1.1 distilled autoregressive inference through Reactor.

The adapter keeps the public model, causal VAE handoff, temporal history, and
ViGeo spatial cache in one process. Reactor controls expand normalized six-axis
motion into the camera-to-world trajectory that ViGeo renders for each chunk.
Prompt and camera updates are sampled at chunk boundaries.
"""

from __future__ import annotations

import asyncio
import importlib
import secrets
import time
from collections.abc import AsyncGenerator
from pathlib import Path
from typing import TYPE_CHECKING, Any, Literal

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
    from examples.alayaworld.alayaworld_assets import (
        load_scene_metadata,
        prepare_runtime_assets,
        read_config,
        scene_image_path,
        scene_prompt_path,
        validate_runtime_paths,
    )
    from examples.alayaworld.alayaworld_backend import AlayaWorldV11Backend
    from examples.alayaworld.alayaworld_camera import CameraMotionPlanner, MotionConfig
    from examples.alayaworld.alayaworld_types import (
        AlayaWorldConfig,
        AlayaWorldOutput,
        AlayaWorldState,
        CameraMotionChanged,
        ImageSelected,
        PauseChanged,
        PromptQueued,
        RolloutResetQueued,
        StateUpdate,
        StepQueued,
    )
    from examples.alayaworld.alayaworld_utils import (
        compact_rollout_cache,
        load_upstream_modules,
        resolve_attention_backend,
        set_attention_backend,
        uploaded_image_video,
        validate_uploaded_image,
    )
else:
    module_prefix = f"{__package__}." if __package__ else ""
    assets_module = importlib.import_module(f"{module_prefix}alayaworld_assets")
    backend_module = importlib.import_module(f"{module_prefix}alayaworld_backend")
    camera_motion = importlib.import_module(f"{module_prefix}alayaworld_camera")
    types_module = importlib.import_module(f"{module_prefix}alayaworld_types")
    utils_module = importlib.import_module(f"{module_prefix}alayaworld_utils")
    load_scene_metadata = assets_module.load_scene_metadata
    prepare_runtime_assets = assets_module.prepare_runtime_assets
    read_config = assets_module.read_config
    scene_image_path = assets_module.scene_image_path
    scene_prompt_path = assets_module.scene_prompt_path
    validate_runtime_paths = assets_module.validate_runtime_paths
    AlayaWorldV11Backend = backend_module.AlayaWorldV11Backend
    CameraMotionPlanner = camera_motion.CameraMotionPlanner
    MotionConfig = camera_motion.MotionConfig
    AlayaWorldConfig = types_module.AlayaWorldConfig
    AlayaWorldOutput = types_module.AlayaWorldOutput
    AlayaWorldState = types_module.AlayaWorldState
    CameraMotionChanged = types_module.CameraMotionChanged
    ImageSelected = types_module.ImageSelected
    PauseChanged = types_module.PauseChanged
    PromptQueued = types_module.PromptQueued
    RolloutResetQueued = types_module.RolloutResetQueued
    StateUpdate = types_module.StateUpdate
    StepQueued = types_module.StepQueued
    compact_rollout_cache = utils_module.compact_rollout_cache
    load_upstream_modules = utils_module.load_upstream_modules
    resolve_attention_backend = utils_module.resolve_attention_backend
    set_attention_backend = utils_module.set_attention_backend
    uploaded_image_video = utils_module.uploaded_image_video
    validate_uploaded_image = utils_module.validate_uploaded_image

logger = get_logger(__name__)

FPS = 24
FRAMES_PER_CHUNK = 32
_UPLOAD_DEFAULT_PROMPT = "Continue the visual scene shown in the reference image."


class AlayaWorld(ReactorPipeline):
    """Run distilled AlayaWorld v1.1 with live prompt and camera controls."""

    state: AlayaWorldState
    output: AlayaWorldOutput
    # Declaring no `fps` hands pacing to the measured cost of each chunk, which
    # is what the client should see: a turn is expensive and its length varies,
    # so a fixed rate would drain the queue and stall between turns. One chunk
    # of queued frames is the smallest bound that still holds a whole turn, so a
    # command lands on the next turn generated rather than waiting behind frames
    # already queued.
    buffer_size = FRAMES_PER_CHUNK

    def __init__(self) -> None:
        super().__init__()
        self._config: AlayaWorldConfig | None = None
        self._torch: Any = None
        self._backend: AlayaWorldV11Backend | None = None
        self._upstream_config: Any = None
        self._load_input_sample: Any = None
        self._check_input_resolution: Any = None
        self._cache: Any = None
        self._selected_input: Path | UploadedFile | None = None
        self._seed = 0
        self._ar_index = 0
        self._active_prompt = ""
        self._reset_in_flight = False
        self._chunk_in_flight = False
        self._camera: CameraMotionPlanner | None = None

    def load(self, config_path: Path | None) -> None:
        """Load the public AlayaWorld engine and prepare its initial scene.

        Args:
            config_path: Path to ``alayaworld.yaml`` from ``reactor.yaml``.
        """
        config = read_config(config_path)
        prepare_runtime_assets(config)
        validate_runtime_paths(config)
        modules = load_upstream_modules(config.source_path)
        torch = modules["torch"]
        upstream_config = modules["load_config"](str(config.upstream_config))
        mode_config = next(iter(upstream_config.validation.modes.values()))
        chunk_latents = int(mode_config.layout.output_latent_frames)
        configured_fps = float(upstream_config.sample.fps)
        configured_chunk_frames = chunk_latents * int(
            upstream_config.sample.temporal_stride
        )
        if configured_fps != float(FPS):
            raise ValueError(
                f"AlayaWorld sample FPS must be {FPS}, got {configured_fps}"
            )
        if configured_chunk_frames != FRAMES_PER_CHUNK:
            raise ValueError(
                f"AlayaWorld chunks must contain {FRAMES_PER_CHUNK} frames, "
                f"got {configured_chunk_frames}"
            )

        attention = resolve_attention_backend(
            config.attention_backend,
            pytorch_attention=modules["pytorch_attention"],
            torch_module=torch,
        )
        modules["set_attention_backend"] = set_attention_backend
        backend = AlayaWorldV11Backend(
            config=config,
            modules=modules,
            attention_function=attention,
            compact_cache=compact_rollout_cache,
        )

        self._config = config
        self._torch = torch
        self._backend = backend
        self._upstream_config = upstream_config
        self._load_input_sample = modules["load_input_sample"]
        self._check_input_resolution = modules["check_input_resolution"]
        self._seed = config.seed
        self._warmup()
        logger.info(
            "AlayaWorld model ready",
            source_revision=config.source_revision,
            ar_revision=config.ar_transformer.revision,
            dmd_revision=config.dmd_lora.revision,
            vigeo_revision=config.vigeo_checkpoint.revision,
            chunk_frames=configured_chunk_frames,
            compile_mode=config.compile_mode,
            random_images=len(config.random_inputs),
            max_chunks_per_rollout=config.max_chunks_per_rollout,
        )

    def _warmup(self) -> None:
        """Generate throwaway turns so the first client does not pay for them.

        Compiled kernels are built on the first turn that reaches them, not when
        the engine is constructed, so without this the cost lands on whoever
        selects the first image. Running the same path here moves it inside
        model load, where the runtime keeps the model unavailable until it
        finishes. A built-in scene supplies the image, and the rollout it builds
        is discarded so a session still starts with nothing selected.
        """
        config = self._config
        if config is None:
            raise RuntimeError("AlayaWorld was not loaded")
        if config.warmup_chunks == 0:
            return
        scene = config.random_inputs[0]
        prompt = scene_prompt_path(scene).read_text(encoding="utf-8").strip()
        started = time.perf_counter()
        logger.info("AlayaWorld warming up", chunks=config.warmup_chunks)
        try:
            self._reset_rollout(prompt or _UPLOAD_DEFAULT_PROMPT, config.seed, scene)
            for _ in range(config.warmup_chunks):
                self._generate_chunk(prompt, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0)
        finally:
            self._cache = None
            if self._backend is not None:
                self._backend.cache = None
            self._camera = None
            self._ar_index = 0
            self._active_prompt = ""
        logger.info(
            "AlayaWorld warmup complete",
            chunks=config.warmup_chunks,
            seconds=round(time.perf_counter() - started, 3),
        )

    @session_started
    def on_session_started(self) -> None:
        """Wait for an uploaded or randomly selected image."""
        config = self._config
        if config is None:
            raise RuntimeError("AlayaWorld was not loaded")
        self.state.prompt = ""
        self._clear_camera_controls()
        self.state.paused = True
        self.state._step_requested = False
        self.state._reset_requested = False
        self._seed = config.seed
        self._selected_input = None
        self._cache = None
        if self._backend is not None:
            self._backend.cache = None
        self._camera = None
        self._ar_index = 0
        self._active_prompt = ""
        self._reset_in_flight = False
        self._chunk_in_flight = False

    @session_ended
    def on_session_ended(self) -> None:
        """Release the selected image and rollout at session end."""
        self._clear_camera_controls()
        self.state._step_requested = False
        self.state._reset_requested = False
        self._selected_input = None
        self._cache = None
        if self._backend is not None:
            self._backend.cache = None
        self._camera = None
        self._ar_index = 0
        self._active_prompt = ""
        self._reset_in_flight = False
        self._chunk_in_flight = False

    @connected
    async def on_connected(self, client: ClientInfo) -> None:
        """Send the complete shared world state to one joining viewer."""
        await client.send(self._state_update())

    @disconnected
    async def on_disconnected(self) -> None:
        """Release held camera motion when a viewer leaves the live session."""
        self._clear_camera_controls()
        await self._send_state_update()

    @event(
        name="set_prompt",
        description=(
            "Set the scene prompt for the next 32-frame chunk without resetting the world. "
            "Requires a selected image. Emits `prompt_queued` and broadcasts `state_update` "
            "on success, or `command_error` when the prompt is empty or no image is selected."
        ),
    )
    async def set_prompt(
        self,
        prompt: str = InputField(
            default="",
            max_length=4096,
            description=(
                "Non-empty scene prompt, trimmed and sampled when the next 32-frame chunk "
                "starts. Requires an image selected by `set_image` or `random_image`."
            ),
        ),
    ) -> PromptQueued:
        """Queue a prompt and confirm the chunk expected to consume it."""
        self._require_selected_image()
        normalized = prompt.strip()
        if not normalized:
            raise CommandError(
                "prompt_required", "AlayaWorld requires a non-empty prompt."
            )
        self.state.prompt = normalized
        message = PromptQueued(
            prompt=normalized, applies_to_chunk=self._next_control_chunk()
        )
        await self._send_state_update()
        return message

    @event(
        name="set_forward",
        description=(
            "Set backward or forward camera velocity for forthcoming chunks. Requires a "
            "selected image. Emits `camera_motion_changed` and broadcasts `state_update` on "
            "success, or `command_error` when no image is selected."
        ),
    )
    async def set_forward(
        self,
        forward: float = InputField(
            default=0.0,
            ge=-1.0,
            le=1.0,
            description=(
                "Backward (-1) to forward (1) velocity sampled when the next chunk starts "
                "and held until changed; zero is neutral."
            ),
        ),
    ) -> CameraMotionChanged:
        """Queue forward motion and return the complete camera state."""
        self._require_selected_image()
        self.state.forward = forward
        message = self._camera_motion_changed()
        await self._send_state_update()
        return message

    @event(
        name="set_strafe",
        description=(
            "Set left or right camera velocity for forthcoming chunks. Requires a selected "
            "image. Emits `camera_motion_changed` and broadcasts `state_update` on success, "
            "or `command_error` when no image is selected."
        ),
    )
    async def set_strafe(
        self,
        strafe: float = InputField(
            default=0.0,
            ge=-1.0,
            le=1.0,
            description=(
                "Left (-1) to right (1) velocity sampled when the next chunk starts and held "
                "until changed; zero is neutral."
            ),
        ),
    ) -> CameraMotionChanged:
        """Queue strafe motion and return the complete camera state."""
        self._require_selected_image()
        self.state.strafe = strafe
        message = self._camera_motion_changed()
        await self._send_state_update()
        return message

    @event(
        name="set_vertical",
        description=(
            "Set down or up camera velocity for forthcoming chunks. Requires a selected image. "
            "Emits `camera_motion_changed` and broadcasts `state_update` on success, or "
            "`command_error` when no image is selected."
        ),
    )
    async def set_vertical(
        self,
        vertical: float = InputField(
            default=0.0,
            ge=-1.0,
            le=1.0,
            description=(
                "Down (-1) to up (1) velocity sampled when the next chunk starts and held "
                "until changed; zero is neutral."
            ),
        ),
    ) -> CameraMotionChanged:
        """Queue vertical motion and return the complete camera state."""
        self._require_selected_image()
        self.state.vertical = vertical
        message = self._camera_motion_changed()
        await self._send_state_update()
        return message

    @event(
        name="set_pitch",
        description=(
            "Set down or up camera pitch velocity for forthcoming chunks. Requires a selected "
            "image. Emits `camera_motion_changed` and broadcasts `state_update` on success, "
            "or `command_error` when no image is selected."
        ),
    )
    async def set_pitch(
        self,
        pitch: float = InputField(
            default=0.0,
            ge=-1.0,
            le=1.0,
            description=(
                "Look down (-1) to up (1) velocity sampled when the next chunk starts and held "
                "until changed; zero is neutral."
            ),
        ),
    ) -> CameraMotionChanged:
        """Queue pitch motion and return the complete camera state."""
        self._require_selected_image()
        self.state.pitch = pitch
        message = self._camera_motion_changed()
        await self._send_state_update()
        return message

    @event(
        name="set_yaw",
        description=(
            "Set left or right camera yaw velocity for forthcoming chunks. Requires a selected "
            "image. Emits `camera_motion_changed` and broadcasts `state_update` on success, "
            "or `command_error` when no image is selected."
        ),
    )
    async def set_yaw(
        self,
        yaw: float = InputField(
            default=0.0,
            ge=-1.0,
            le=1.0,
            description=(
                "Turn left (-1) to right (1) velocity sampled when the next chunk starts and "
                "held until changed; zero is neutral."
            ),
        ),
    ) -> CameraMotionChanged:
        """Queue yaw motion and return the complete camera state."""
        self._require_selected_image()
        self.state.yaw = yaw
        message = self._camera_motion_changed()
        await self._send_state_update()
        return message

    @event(
        name="set_roll",
        description=(
            "Set counterclockwise or clockwise camera roll velocity for forthcoming chunks. "
            "Requires a selected image. Emits `camera_motion_changed` and broadcasts "
            "`state_update` on success, or `command_error` when no image is selected."
        ),
    )
    async def set_roll(
        self,
        roll: float = InputField(
            default=0.0,
            ge=-1.0,
            le=1.0,
            description=(
                "Counterclockwise (-1) to clockwise (1) velocity sampled when the next chunk "
                "starts and held until changed; zero is neutral."
            ),
        ),
    ) -> CameraMotionChanged:
        """Queue roll motion and return the complete camera state."""
        self._require_selected_image()
        self.state.roll = roll
        message = self._camera_motion_changed()
        await self._send_state_update()
        return message

    @event(
        name="set_paused",
        description=(
            "Pause before the next chunk or resume continuous generation, releasing all camera "
            "motion in either case. Resuming requires a selected image. Emits `pause_changed` "
            "and broadcasts `state_update` on success, or `command_error` when resuming without "
            "an image."
        ),
    )
    async def set_paused(
        self,
        paused: bool = InputField(
            default=False,
            description=(
                "True pauses before the next chunk; false resumes continuous generation. "
                "Both values reset all camera velocities to zero."
            ),
        ),
    ) -> PauseChanged:
        """Set pause state, release camera motion, and report the result."""
        if not paused:
            self._require_selected_image()
        self.state.paused = paused
        self.state._step_requested = False
        self._clear_camera_controls()
        message = PauseChanged(paused=paused, camera_motion_released=True)
        await self._send_state_update()
        return message

    @event(
        name="step",
        description=(
            "Queue exactly one 32-frame chunk while continuous generation is paused. Requires a "
            "selected image and `paused=true`. Emits `step_queued` and broadcasts `state_update` "
            "on success, or `command_error` when either precondition is missing."
        ),
    )
    async def step(self) -> StepQueued:
        """Request one complete chunk and report its position in the rollout."""
        self._require_selected_image()
        if not self.state.paused:
            raise CommandError(
                "pause_required", "Pause AlayaWorld before requesting one step."
            )
        self.state._step_requested = True
        message = StepQueued(applies_to_chunk=self._next_control_chunk())
        await self._send_state_update()
        return message

    @event(
        name="reset",
        description=(
            "Queue a fresh rollout from the selected image, current prompt, and neutral camera "
            "motion. Requires a selected image. Emits `rollout_reset_queued` and broadcasts "
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
                "Random seed for the fresh rollout. Use -1 to retain the active seed; a "
                "non-negative value replaces it when the reset begins."
            ),
        ),
    ) -> RolloutResetQueued:
        """Request a fresh rollout and report the seed and replaced chunk count."""
        self._require_selected_image()
        if seed >= 0:
            self._seed = seed
        completed_chunks = self._ar_index
        self._clear_camera_controls()
        self.state._step_requested = False
        self.state._reset_requested = True
        message = RolloutResetQueued(
            trigger="manual",
            seed=self._seed,
            completed_chunks=completed_chunks,
            applies_to_chunk=1,
        )
        await self._send_state_update()
        return message

    @event(
        name="set_image",
        description=(
            "Select an uploaded image, queue a fresh rollout, and resume continuous generation. "
            "Can replace the image at any time. Emits `image_selected` and broadcasts "
            "`state_update` on success, or `command_error` when the upload is missing, too "
            "large, mislabeled, or undecodable."
        ),
    )
    async def set_image(
        self,
        image: UploadedFile = InputField(  # noqa: B008 - schema field declaration
            description=(
                "Reference image uploaded through the Reactor upload protocol. JPEG, PNG, WebP, "
                "or BMP up to 25 MiB and 100 million pixels; EXIF orientation is applied before "
                "the image is center-cropped to `main_video` resolution."
            ),
        ),
        prompt: str = InputField(
            default="",
            max_length=4096,
            description=(
                "Optional scene prompt for the fresh rollout. Whitespace is trimmed; an empty "
                "value retains the current prompt or uses a generic continuation prompt when "
                "none exists."
            ),
        ),
    ) -> ImageSelected:
        """Validate uploaded image bytes and select them for the next rollout."""
        validate_uploaded_image(image)
        self._selected_input = image
        if self.state is not None:
            self.state.prompt = (
                prompt.strip() or self.state.prompt.strip() or _UPLOAD_DEFAULT_PROMPT
            )
            self.state.paused = False
            self.state._step_requested = False
            self.state._reset_requested = True
            self._clear_camera_controls()
        message = ImageSelected(
            source="uploaded",
            filename=image.name,
            prompt=self.state.prompt,
            applies_to_chunk=1,
        )
        await self._send_state_update()
        return message

    @event(
        name="random_image",
        description=(
            "Select a built-in image and its matching prompt, queue a fresh rollout, and resume "
            "continuous generation. Valid when built-in examples are configured. Emits "
            "`image_selected` and broadcasts `state_update` on success, or `command_error` when "
            "no usable image or prompt is available."
        ),
    )
    async def random_image(self) -> ImageSelected:
        """Select a different configured example image when possible."""
        config = self._config
        if config is None or not config.random_inputs:
            raise CommandError(
                "image_unavailable", "No built-in images are configured."
            )
        candidates = [
            path for path in config.random_inputs if path != self._selected_input
        ]
        selected = secrets.choice(candidates or list(config.random_inputs))
        self._selected_input = selected
        prompt = scene_prompt_path(selected).read_text(encoding="utf-8").strip()
        if not prompt:
            raise CommandError(
                "prompt_unavailable", "The selected built-in image has no prompt."
            )
        if self.state is not None:
            self.state.prompt = prompt
            self.state.paused = False
            self.state._step_requested = False
            self.state._reset_requested = True
            self._clear_camera_controls()
        message = ImageSelected(
            source="built_in",
            filename=scene_image_path(selected).name,
            prompt=prompt,
            applies_to_chunk=1,
        )
        await self._send_state_update()
        return message

    async def inference(self) -> AsyncGenerator[Any, None]:
        """Generate chunks off-loop and emit their RGB frames at 24 FPS."""
        while True:
            selected_input = self._selected_input
            if selected_input is None:
                yield Idle
                continue

            config = self._config
            if config is None:
                raise RuntimeError("AlayaWorld model was not loaded")
            if (
                self._ar_index >= config.max_chunks_per_rollout
                and not self.state._reset_requested
            ):
                completed_chunks = self._ar_index
                self._clear_camera_controls()
                self.state._step_requested = False
                self.state._reset_requested = True
                await self.send(
                    RolloutResetQueued(
                        trigger="automatic_chunk_limit",
                        seed=self._seed,
                        completed_chunks=completed_chunks,
                        applies_to_chunk=1,
                    )
                )
                await self._send_state_update()
                logger.info(
                    "AlayaWorld rollout reached its chunk limit",
                    completed_chunks=completed_chunks,
                    next_chunk=1,
                )

            if self.state._reset_requested:
                self.state._reset_requested = False
                self._reset_in_flight = True
                # Cut playout at the boundary so queued frames from the world
                # being replaced never play after the new one starts.
                self.output.flush()
                try:
                    await asyncio.to_thread(
                        self._reset_rollout,
                        self.state.prompt,
                        self._seed,
                        selected_input,
                    )
                finally:
                    self._reset_in_flight = False
                await self._send_state_update()

            if self.state._reset_requested:
                continue

            if self.state.paused and not self.state._step_requested:
                yield Idle
                continue

            self.state._step_requested = False
            prompt = self.state.prompt
            strafe = self.state.strafe
            vertical = self.state.vertical
            forward = self.state.forward
            pitch = self.state.pitch
            yaw = self.state.yaw
            roll = self.state.roll
            self._chunk_in_flight = True
            try:
                frames = await asyncio.to_thread(
                    self._generate_chunk,
                    prompt,
                    strafe,
                    vertical,
                    forward,
                    pitch,
                    yaw,
                    roll,
                )
            finally:
                self._chunk_in_flight = False
            await self._send_state_update()
            if self.state._reset_requested:
                continue
            # One batched output per turn, so the runtime pairs all 32 frames
            # with the time they took and plays them at the rate they were
            # produced.
            yield AlayaWorldOutput(main_video=frames)

    def _reset_rollout(
        self,
        prompt: str,
        seed: int,
        selected_input: Path | UploadedFile,
    ) -> None:
        """Build fresh causal VAE and ViGeo state without reloading weights."""
        config = self._config
        backend = self._backend
        if config is None or backend is None:
            raise RuntimeError("AlayaWorld model was not loaded")
        self._cache = None
        self._camera = None
        video, metadata = self._prepare_scene(selected_input)
        cache = backend.reset(
            video_pixels=video,
            metadata=metadata,
            prompt=prompt,
            seed=seed,
        )
        cameras = cache.metadata["cam_c2w"]
        if cameras.dim() == 4:
            cameras = cameras[0]
        initial_pose = (
            cameras[backend.prefix_frames - 1]
            .detach()
            .cpu()
            .to(self._torch.float32)
            .numpy()
        )
        self._camera = CameraMotionPlanner(
            initial_pose,
            MotionConfig(
                fps=backend.fps,
                strafe_units_per_second=config.strafe_units_per_second,
                vertical_units_per_second=config.vertical_units_per_second,
                forward_units_per_second=config.forward_units_per_second,
                pitch_degrees_per_second=config.pitch_degrees_per_second,
                yaw_degrees_per_second=config.yaw_degrees_per_second,
                roll_degrees_per_second=config.roll_degrees_per_second,
            ),
        )
        self._cache = cache
        self._ar_index = 0
        self._active_prompt = prompt

    def _prepare_scene(
        self,
        selected_input: Path | UploadedFile,
    ) -> tuple[Any, dict[str, Any]]:
        """Prepare one built-in or uploaded image for upstream cache initialization."""
        config = self._config
        upstream_config = self._upstream_config
        if config is None or upstream_config is None:
            raise RuntimeError("AlayaWorld model was not loaded")
        target_hw = (
            int(upstream_config.sample.height),
            int(upstream_config.sample.width),
        )
        if isinstance(selected_input, UploadedFile):
            metadata = load_scene_metadata(config.upload_template, self._torch)
            video = uploaded_image_video(
                selected_input,
                metadata,
                target_hw=target_hw,
                torch_module=self._torch,
            )
        else:
            video, _caption, metadata = self._load_input_sample(
                str(selected_input),
                image_target_hw=target_hw,
            )
        self._check_input_resolution(video, upstream_config)
        return video, metadata

    def _generate_chunk(
        self,
        prompt: str,
        strafe: float,
        vertical: float,
        forward: float,
        pitch: float,
        yaw: float,
        roll: float,
    ) -> np.ndarray:
        """Run one native AlayaWorld v1.1 distilled turn."""
        backend = self._backend
        planner = self._camera
        config = self._config
        if backend is None or planner is None or config is None:
            raise RuntimeError("AlayaWorld rollout was not initialized")
        trajectory = planner.plan(
            strafe=strafe,
            vertical=vertical,
            forward=forward,
            pitch=pitch,
            yaw=yaw,
            roll=roll,
            frame_count=FRAMES_PER_CHUNK,
        )
        started = time.perf_counter()
        frames = backend.generate(prompt, trajectory)
        self._ar_index += 1
        self._active_prompt = prompt
        logger.info(
            "AlayaWorld v1.1 chunk ready",
            chunk=self._ar_index,
            frames=int(frames.shape[0]),
            seconds=round(time.perf_counter() - started, 3),
            prompt=prompt[:80],
            strafe=strafe,
            vertical=vertical,
            forward=forward,
            pitch=pitch,
            yaw=yaw,
            roll=roll,
        )
        return frames

    def _state_update(self) -> StateUpdate:
        """Return a complete client-facing snapshot of the shared world state."""
        selected = self._selected_input
        image_source: Literal["uploaded", "built_in"] | None = None
        image_name: str | None = None
        if isinstance(selected, UploadedFile):
            image_source = "uploaded"
            image_name = selected.name
        elif selected is not None:
            image_source = "built_in"
            try:
                image_name = scene_image_path(selected).name
            except FileNotFoundError:
                image_name = selected.name
        config = self._config
        return StateUpdate.from_state(
            self.state,
            image_source=image_source,
            image_name=image_name,
            active_prompt=self._active_prompt or None,
            seed=self._seed,
            generating=self._reset_in_flight or self._chunk_in_flight,
            completed_chunks=self._ar_index,
            next_chunk=None if selected is None else self._next_control_chunk(),
            max_chunks=config.max_chunks_per_rollout if config is not None else 0,
        )

    async def _send_state_update(self) -> None:
        """Broadcast the complete observable session state."""
        await self.send(self._state_update())

    def _next_control_chunk(self) -> int:
        """Return the one-based chunk expected to consume a new control value."""
        starts_new_rollout = (
            self._selected_input is None
            or self._reset_in_flight
            or (self.state is not None and self.state._reset_requested)
        )
        if starts_new_rollout:
            return 1
        return self._ar_index + 1 + int(self._chunk_in_flight)

    def _camera_motion_changed(self) -> CameraMotionChanged:
        """Describe the complete camera state after a control event."""
        return CameraMotionChanged(
            strafe=self.state.strafe,
            vertical=self.state.vertical,
            forward=self.state.forward,
            pitch=self.state.pitch,
            yaw=self.state.yaw,
            roll=self.state.roll,
            applies_to_chunk=self._next_control_chunk(),
        )

    def _clear_camera_controls(self) -> None:
        """Return all camera controls to neutral."""
        if self.state is None:
            return
        self.state.strafe = 0.0
        self.state.vertical = 0.0
        self.state.forward = 0.0
        self.state.pitch = 0.0
        self.state.yaw = 0.0
        self.state.roll = 0.0

    def _require_selected_image(self) -> None:
        """Require a rollout origin before accepting world controls."""
        if self._selected_input is None:
            raise CommandError(
                "image_required", "Upload an image or select Random Image first."
            )
