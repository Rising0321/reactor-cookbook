"""Serve AlayaWorld distilled autoregressive inference through Reactor Runtime.

The adapter keeps AlayaWorld's public ``FlashAlayaPipeline`` intact. Reactor
controls provide normalized six-axis camera motion, which is expanded into the
camera-to-world trajectory consumed by the upstream action and spatial-memory
paths. Prompt updates and camera controls are sampled at chunk boundaries.
"""

from __future__ import annotations

import asyncio
import importlib
import io
import re
import secrets
import subprocess
import sys
import tempfile
import time
from collections.abc import AsyncGenerator
from contextlib import suppress
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Any, Never, cast

import numpy as np
import yaml
from PIL import Image, ImageOps, UnidentifiedImageError

from reactor_runtime import (
    CommandError,
    Idle,
    InputField,
    InputState,
    MessageField,
    ModelMessage,
    Output,
    ReactorPipeline,
    UploadedFile,
    Video,
    disconnected,
    event,
    session_ended,
    session_started,
)
from reactor_runtime.log import get_logger

if TYPE_CHECKING:
    from examples.alayaworld.alayaworld_camera import CameraMotionPlanner, MotionConfig
else:
    module_prefix = f"{__package__}." if __package__ else ""
    camera_motion = importlib.import_module(f"{module_prefix}alayaworld_camera")
    CameraMotionPlanner = camera_motion.CameraMotionPlanner
    MotionConfig = camera_motion.MotionConfig

logger = get_logger(__name__)

FPS = 24
FRAMES_PER_CHUNK = 32
_REVISION_PATTERN = re.compile(r"[0-9a-f]{40}")
_COMPILE_MODES = ["none", "default", "reduce-overhead", "max-autotune"]
_ATTENTION_BACKENDS = ["pytorch", "upstream"]
_UPLOAD_MAX_BYTES = 25 * 1024 * 1024
_UPLOAD_MAX_PIXELS = 100_000_000
_UPLOAD_FORMATS = {"BMP", "JPEG", "PNG", "WEBP"}
_UPLOAD_DEFAULT_PROMPT = "Continue the visual scene shown in the reference image."
_SCENE_IMAGE_EXTENSIONS = (".png", ".jpg", ".jpeg", ".webp", ".bmp")
_SCENE_MEMBER_SUFFIXES = (
    "_video.mp4",
    "_camera.pt",
    "_prompt.txt",
    *tuple(f"_image{extension}" for extension in _SCENE_IMAGE_EXTENSIONS),
)


@dataclass(frozen=True)
class _Asset:
    """Describe one pinned public model asset."""

    path: Path
    repo_id: str
    revision: str


@dataclass(frozen=True)
class _Config:
    """Hold validated AlayaWorld adapter settings."""

    source_path: Path
    source_url: str
    source_revision: str
    upstream_config: Path
    upload_template: Path
    random_inputs: tuple[Path, ...]
    model: _Asset
    gemma: _Asset
    da3_source_path: Path
    da3_source_url: str
    da3_source_revision: str
    da3_model: _Asset
    da3_cache: Path
    seed: int
    compile_mode: str
    attention_backend: str
    flex_attention: bool
    ttc: bool
    bank_taehv: bool
    taehv_path: Path | None
    decode_overlap_latents: int
    max_spatial_frames: int
    recent_spatial_frames: int
    strafe_units_per_second: float
    vertical_units_per_second: float
    forward_units_per_second: float
    pitch_degrees_per_second: float
    yaw_degrees_per_second: float
    roll_degrees_per_second: float


class AlayaWorldOutput(Output):
    """Carry one generated AlayaWorld RGB frame."""

    main_video: Video


class ImageSelected(ModelMessage):
    """Confirm the image selected as the next rollout origin."""

    source: str = MessageField(description="Whether the image was uploaded or built in.")
    filename: str = MessageField(description="Name of the selected image.")


class PromptQueued(ModelMessage):
    """Confirm a prompt queued for an autoregressive chunk boundary."""

    prompt: str = MessageField(description="Prompt accepted by the model.")
    applies_to_chunk: int = MessageField(
        description="One-based chunk number expected to use the prompt."
    )


class CameraMotionChanged(ModelMessage):
    """Describe the complete camera motion queued for a chunk boundary."""

    strafe: float = MessageField(description="Normalized left-to-right translation velocity.")
    vertical: float = MessageField(description="Normalized down-to-up translation velocity.")
    forward: float = MessageField(description="Normalized backward-to-forward velocity.")
    pitch: float = MessageField(description="Normalized down-to-up pitch velocity.")
    yaw: float = MessageField(description="Normalized left-to-right yaw velocity.")
    roll: float = MessageField(description="Normalized counterclockwise-to-clockwise roll.")
    applies_to_chunk: int = MessageField(
        description="One-based chunk number expected to use this motion."
    )


class AlayaWorldState(InputState):
    """Hold prompt, camera, and generation controls for one session."""

    prompt: str = InputField(
        default="",
        max_length=4096,
        description="Prompt applied at the next autoregressive chunk boundary.",
    )
    forward: float = InputField(
        default=0.0,
        ge=-1.0,
        le=1.0,
        description="Normalized backward-to-forward camera velocity.",
    )
    strafe: float = InputField(
        default=0.0,
        ge=-1.0,
        le=1.0,
        description="Normalized left-to-right camera translation velocity.",
    )
    vertical: float = InputField(
        default=0.0,
        ge=-1.0,
        le=1.0,
        description="Normalized down-to-up camera translation velocity.",
    )
    pitch: float = InputField(
        default=0.0,
        ge=-1.0,
        le=1.0,
        description="Normalized down-to-up camera pitch velocity.",
    )
    yaw: float = InputField(
        default=0.0,
        ge=-1.0,
        le=1.0,
        description="Normalized left-to-right camera yaw velocity.",
    )
    roll: float = InputField(
        default=0.0,
        ge=-1.0,
        le=1.0,
        description="Normalized counterclockwise-to-clockwise camera roll velocity.",
    )
    paused: bool = InputField(default=False, description="Pause chunk generation.")
    _step_requested: bool = False
    _reset_requested: bool = True


class AlayaWorld(ReactorPipeline):
    """Run AlayaWorld with live prompt and six-axis camera controls."""

    state: AlayaWorldState
    output: AlayaWorldOutput
    fps = FPS
    buffer_size = 1

    def __init__(self) -> None:
        super().__init__()
        self._config: _Config | None = None
        self._torch: Any = None
        self._engine: Any = None
        self._alaya_pipeline: Any = None
        self._upstream_config: Any = None
        self._load_input_sample: Any = None
        self._check_input_resolution: Any = None
        self._plan_rollout: Any = None
        self._cache: Any = None
        self._selected_input: Path | UploadedFile | None = None
        self._needed_latents = 0
        self._chunk_latents = 0
        self._history_latents = 0
        self._gap_steps = 0
        self._condition_latents = 0
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
        config = _read_config(config_path)
        _prepare_runtime_assets(config)
        _validate_runtime_paths(config)
        modules = _load_upstream_modules(config.source_path)
        torch = modules["torch"]

        upstream_config = modules["load_config"](str(config.upstream_config))
        upstream_config.paths.model = str(config.model.path)
        upstream_config.paths.gemma = str(config.gemma.path)
        upstream_config.paths.da3_repo = str(config.da3_source_path)
        upstream_config.paths.da3_model = config.da3_model.repo_id
        upstream_config.paths.da3_cache = str(config.da3_cache)
        upstream_config.paths.taehv = str(config.taehv_path) if config.taehv_path else ""

        mode_config = next(iter(upstream_config.validation.modes.values()))
        chunk_latents = int(mode_config.layout.output_latent_frames)
        history_latents = int(
            upstream_config.layout.history_latent_frames
            if mode_config.layout.history_latent_frames is None
            else mode_config.layout.history_latent_frames
        )
        gap_steps = int(
            float(mode_config.layout.max_gap_sec or 0.0)
            * float(upstream_config.sample.fps)
            / int(upstream_config.sample.temporal_stride)
        )
        condition_latents = int(mode_config.layout.condition_latent_frames)

        flex_attention = config.flex_attention and config.compile_mode != "none"
        engine = modules["build_engine"](
            upstream_config,
            compile_mode=config.compile_mode,
            compile_aux=False,
            bank_taehv=config.bank_taehv,
            verbose=True,
        )
        if config.attention_backend == "pytorch":
            patched_attention_modules = _set_attention_backend(
                engine,
                modules["pytorch_attention"],
            )
            logger.info(
                "AlayaWorld attention backend selected",
                backend="pytorch",
                modules=patched_attention_modules,
            )
        if modules["apply_da3_robust_scale"]():
            logger.info("AlayaWorld DA3 colinear camera fallback enabled")
        alaya_pipeline = modules["pipeline_type"](
            engine,
            control_modes=list(mode_config.control),
            use_memory=bool(mode_config.use_memory),
            action_cfg_scale=float(mode_config.action_cfg_scale),
            flex_attn=flex_attention,
            seed=config.seed,
            ttc=config.ttc,
            ttc_levels=tuple(int(value) for value in upstream_config.validation.ttc.levels),
            ttc_strength=float(upstream_config.validation.ttc.strength),
            ttc_ref_action=bool(upstream_config.validation.ttc.ref_action),
        )

        self._config = config
        self._torch = torch
        self._engine = engine
        self._alaya_pipeline = alaya_pipeline
        self._upstream_config = upstream_config
        self._load_input_sample = modules["load_input_sample"]
        self._check_input_resolution = modules["check_input_resolution"]
        self._plan_rollout = modules["plan_rollout"]
        self._chunk_latents = chunk_latents
        self._history_latents = history_latents
        self._gap_steps = gap_steps
        self._condition_latents = condition_latents
        self._seed = config.seed
        logger.info(
            "AlayaWorld model ready",
            source_revision=config.source_revision,
            checkpoint_revision=config.model.revision,
            chunk_frames=chunk_latents * int(upstream_config.sample.temporal_stride),
            compile_mode=config.compile_mode,
            random_images=len(config.random_inputs),
        )

    @session_started
    def on_session_started(self) -> None:
        """Wait for an uploaded or randomly selected image."""
        self.state.prompt = ""
        self.state.forward = 0.0
        self.state.strafe = 0.0
        self.state.vertical = 0.0
        self.state.pitch = 0.0
        self.state.yaw = 0.0
        self.state.roll = 0.0
        self.state.paused = True
        self.state._step_requested = False
        self.state._reset_requested = False
        self._selected_input = None
        self._cache = None
        self._camera = None
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
        self._camera = None
        self._reset_in_flight = False
        self._chunk_in_flight = False

    @disconnected
    def on_disconnected(self) -> None:
        """Release camera controls when a client leaves."""
        self._clear_camera_controls()

    @event(
        name="set_prompt",
        description="Queue a prompt for the next autoregressive chunk boundary.",
    )
    def set_prompt(
        self,
        prompt: str = InputField(
            default="",
            max_length=4096,
            description="Prompt queued for the next autoregressive chunk boundary.",
        ),
    ) -> PromptQueued:
        """Queue a prompt and confirm the chunk expected to consume it."""
        if self.state is not None:
            self.state.prompt = prompt
        return PromptQueued(prompt=prompt, applies_to_chunk=self._next_control_chunk())

    @event(name="set_forward", description="Set backward or forward camera translation.")
    def set_forward(
        self,
        forward: float = InputField(
            default=0.0,
            ge=-1.0,
            le=1.0,
            description="Normalized backward-to-forward camera velocity.",
        ),
    ) -> CameraMotionChanged:
        """Queue forward motion and return the complete camera state."""
        self.state.forward = forward
        return self._camera_motion_changed()

    @event(name="set_strafe", description="Set left or right camera translation.")
    def set_strafe(
        self,
        strafe: float = InputField(
            default=0.0,
            ge=-1.0,
            le=1.0,
            description="Normalized left-to-right camera translation velocity.",
        ),
    ) -> CameraMotionChanged:
        """Queue strafe motion and return the complete camera state."""
        self.state.strafe = strafe
        return self._camera_motion_changed()

    @event(name="set_vertical", description="Set down or up camera translation.")
    def set_vertical(
        self,
        vertical: float = InputField(
            default=0.0,
            ge=-1.0,
            le=1.0,
            description="Normalized down-to-up camera translation velocity.",
        ),
    ) -> CameraMotionChanged:
        """Queue vertical motion and return the complete camera state."""
        self.state.vertical = vertical
        return self._camera_motion_changed()

    @event(name="set_pitch", description="Set down or up camera pitch.")
    def set_pitch(
        self,
        pitch: float = InputField(
            default=0.0,
            ge=-1.0,
            le=1.0,
            description="Normalized down-to-up camera pitch velocity.",
        ),
    ) -> CameraMotionChanged:
        """Queue pitch motion and return the complete camera state."""
        self.state.pitch = pitch
        return self._camera_motion_changed()

    @event(name="set_yaw", description="Set left or right camera yaw.")
    def set_yaw(
        self,
        yaw: float = InputField(
            default=0.0,
            ge=-1.0,
            le=1.0,
            description="Normalized left-to-right camera yaw velocity.",
        ),
    ) -> CameraMotionChanged:
        """Queue yaw motion and return the complete camera state."""
        self.state.yaw = yaw
        return self._camera_motion_changed()

    @event(name="set_roll", description="Set counterclockwise or clockwise camera roll.")
    def set_roll(
        self,
        roll: float = InputField(
            default=0.0,
            ge=-1.0,
            le=1.0,
            description="Normalized counterclockwise-to-clockwise camera roll velocity.",
        ),
    ) -> CameraMotionChanged:
        """Queue roll motion and return the complete camera state."""
        self.state.roll = roll
        return self._camera_motion_changed()

    @event(name="set_paused", description="Pause or resume AlayaWorld chunk generation.")
    def set_paused(
        self,
        paused: bool = InputField(default=False, description="Pause before the next chunk."),
    ) -> None:
        """Set pause state and release camera motion at the boundary."""
        if self.state is None:
            return
        self.state.paused = paused
        self.state._step_requested = False
        self._clear_camera_controls()

    @event(name="step", description="Generate and play one AlayaWorld chunk while paused.")
    def step(self) -> None:
        """Request one complete chunk without leaving paused mode."""
        if self.state is None or not self.state.paused:
            return
        if self._selected_input is None:
            raise CommandError("image_required", "Upload an image or select Random Image first.")
        self.state._step_requested = True

    @event(name="reset", description="Reset to the configured scene and camera pose.")
    def reset(
        self,
        seed: int = InputField(
            default=-1,
            ge=-1,
            le=2_147_483_647,
            description="New rollout seed; -1 reuses the active seed.",
        ),
    ) -> None:
        """Request a fresh rollout cache at the next chunk boundary."""
        if self._selected_input is None:
            raise CommandError("image_required", "Upload an image or select Random Image first.")
        if seed >= 0:
            self._seed = seed
        if self.state is None:
            return
        self._clear_camera_controls()
        self.state._step_requested = False
        self.state._reset_requested = True

    @event(
        name="set_image",
        description="Reset from an uploaded JPEG, PNG, WebP, or BMP image.",
    )
    def set_image(
        self,
        image: UploadedFile,
        prompt: str = InputField(
            default="",
            max_length=4096,
            description="Optional scene prompt; empty keeps the current prompt.",
        ),
    ) -> ImageSelected:
        """Validate uploaded image bytes and select them for the next rollout."""
        _validate_uploaded_image(image)
        self._selected_input = image
        if self.state is not None:
            self.state.prompt = (
                prompt.strip() or self.state.prompt.strip() or _UPLOAD_DEFAULT_PROMPT
            )
            self.state.paused = False
            self.state._step_requested = False
            self.state._reset_requested = True
            self._clear_camera_controls()
        return ImageSelected(source="uploaded", filename=image.name)

    @event(
        name="random_image",
        description="Reset from a randomly selected built-in AlayaWorld image.",
    )
    def random_image(self) -> ImageSelected:
        """Select a different configured example image when possible."""
        config = self._config
        if config is None or not config.random_inputs:
            raise CommandError("image_unavailable", "No built-in images are configured.")
        candidates = [path for path in config.random_inputs if path != self._selected_input]
        selected = secrets.choice(candidates or list(config.random_inputs))
        self._selected_input = selected
        prompt = _scene_prompt_path(selected).read_text(encoding="utf-8").strip()
        if self.state is not None:
            self.state.prompt = prompt
            self.state.paused = False
            self.state._step_requested = False
            self.state._reset_requested = True
            self._clear_camera_controls()
        return ImageSelected(source="built_in", filename=_scene_image_path(selected).name)

    async def inference(self) -> AsyncGenerator[Any, None]:
        """Generate chunks off-loop and emit their RGB frames at 24 FPS."""
        while True:
            selected_input = self._selected_input
            if selected_input is None:
                yield Idle
                continue

            if self.state._reset_requested:
                self.state._reset_requested = False
                self._reset_in_flight = True
                try:
                    await asyncio.to_thread(
                        self._reset_rollout,
                        self.state.prompt,
                        self._seed,
                        selected_input,
                    )
                finally:
                    self._reset_in_flight = False

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
            for frame in frames:
                if self.state._reset_requested:
                    break
                yield AlayaWorldOutput(main_video=frame)

    def _reset_rollout(
        self,
        prompt: str,
        seed: int,
        selected_input: Path | UploadedFile,
    ) -> None:
        """Build a fresh upstream cache without reloading model weights."""
        config = self._config
        pipeline = self._alaya_pipeline
        if config is None or pipeline is None:
            raise RuntimeError("AlayaWorld model was not loaded")
        self._cache = None
        self._camera = None
        video, metadata, needed_latents = self._prepare_scene(selected_input)
        pipeline.seed = seed
        cache = pipeline.initialize_cache(
            video,
            prompt,
            metadata,
            rounds=1,
            K=self._chunk_latents,
            cond_end=self._condition_latents,
            needed_latents=needed_latents,
        )
        stride = int(pipeline.cfg.sample.temporal_stride)
        anchor_index = max(0, int(cache.target_base_start) * stride - stride)
        camera = _camera_frames(metadata["cam_c2w"])
        initial_pose = camera[anchor_index].detach().cpu().to(self._torch.float32).numpy()
        self._camera = CameraMotionPlanner(
            initial_pose,
            MotionConfig(
                fps=float(pipeline.cfg.sample.fps),
                strafe_units_per_second=config.strafe_units_per_second,
                vertical_units_per_second=config.vertical_units_per_second,
                forward_units_per_second=config.forward_units_per_second,
                pitch_degrees_per_second=config.pitch_degrees_per_second,
                yaw_degrees_per_second=config.yaw_degrees_per_second,
                roll_degrees_per_second=config.roll_degrees_per_second,
            ),
        )
        self._cache = cache
        self._needed_latents = needed_latents
        self._ar_index = 0
        self._active_prompt = prompt

    def _prepare_scene(
        self,
        selected_input: Path | UploadedFile,
    ) -> tuple[Any, dict[str, Any], int]:
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
            metadata = _load_scene_metadata(config.upload_template, self._torch)
            video = _uploaded_image_video(
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
        video, metadata, rounds, _max_rounds, needed_latents = self._plan_rollout(
            upstream_config,
            video,
            metadata,
            rounds_cap=1,
            K=self._chunk_latents,
            N=self._history_latents,
            gap_steps=self._gap_steps,
            cond_end=self._condition_latents,
        )
        if rounds != 1:
            raise RuntimeError("the selected AlayaWorld image cannot seed one chunk")
        return video, metadata, int(needed_latents)

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
        """Run one native AlayaWorld generate/finalize/decode turn."""
        pipeline = self._alaya_pipeline
        cache = self._cache
        engine = self._engine
        config = self._config
        if pipeline is None or cache is None or engine is None or config is None:
            raise RuntimeError("AlayaWorld rollout was not initialized")
        if prompt != self._active_prompt:
            cache.context = engine.encode_caption(prompt)
            self._active_prompt = prompt

        self._write_camera_trajectory(
            cache,
            strafe=strafe,
            vertical=vertical,
            forward=forward,
            pitch=pitch,
            yaw=yaw,
            roll=roll,
        )
        history = cache.history
        if history is None:
            raise RuntimeError("AlayaWorld interactive decode requires history latents")
        started = time.perf_counter()
        pred = pipeline.generate(self._ar_index, cache)
        pipeline.finalize(self._ar_index, cache, pred)
        _compact_rollout_cache(
            cache,
            max_spatial_frames=config.max_spatial_frames,
            recent_spatial_frames=config.recent_spatial_frames,
        )
        frames = self._decode_new_frames(history, pred)
        self._ar_index += 1
        logger.info(
            "AlayaWorld chunk ready",
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

    def _write_camera_trajectory(
        self,
        cache: Any,
        *,
        strafe: float,
        vertical: float,
        forward: float,
        pitch: float,
        yaw: float,
        roll: float,
    ) -> None:
        """Replace the next chunk's camera slots with frontend-controlled poses."""
        planner = self._camera
        if planner is None:
            raise RuntimeError("AlayaWorld camera planner was not initialized")
        stride = int(self._alaya_pipeline.cfg.sample.temporal_stride)
        target_pixel_start = int(cache.target_start(self._ar_index)) * stride
        target_pixel_end = target_pixel_start + int(cache.K) * stride
        write_start = target_pixel_start
        if self._ar_index == 0:
            write_start = max(0, target_pixel_start - stride + 1)
        trajectory = planner.plan(
            strafe=strafe,
            vertical=vertical,
            forward=forward,
            pitch=pitch,
            yaw=yaw,
            roll=roll,
            frame_count=target_pixel_end - write_start,
        )
        metadata = cast(dict[str, Any], cache.metadata)
        camera = metadata["cam_c2w"]
        camera = _ensure_camera_capacity(camera, target_pixel_end, self._torch)
        values = self._torch.from_numpy(trajectory).to(device=camera.device, dtype=camera.dtype)
        if camera.dim() == 3:
            camera[write_start:target_pixel_end] = values
        else:
            camera[:, write_start:target_pixel_end] = values.unsqueeze(0).expand(
                camera.shape[0], -1, -1, -1
            )
        metadata["cam_c2w"] = camera
        if "cam_c2w_raw" in metadata:
            metadata["cam_c2w_raw"] = camera.clone()
        metadata["frame_end"] = int(_camera_frames(camera).shape[0])

    def _decode_new_frames(self, history: Any, pred: Any) -> np.ndarray:
        """Decode one chunk with bounded left context and return its new frames."""
        config = self._config
        engine = self._engine
        if config is None or engine is None:
            raise RuntimeError("AlayaWorld model was not loaded")
        overlap = min(config.decode_overlap_latents, int(history.shape[2]))
        latent = self._torch.cat(
            [history[:, :, -overlap:].contiguous(), pred.to(history.dtype)],
            dim=2,
        ).contiguous()
        decoded = engine.decode_latent_to_video_frames(latent)
        stride = int(self._alaya_pipeline.cfg.sample.temporal_stride)
        prefix_frames = (overlap - 1) * stride + 1
        frames = decoded[prefix_frames:]
        expected = int(pred.shape[2]) * stride
        if int(frames.shape[0]) != expected:
            raise RuntimeError(
                f"AlayaWorld decoded {int(frames.shape[0])} new frames; expected {expected}"
            )
        return np.ascontiguousarray(frames.numpy(), dtype=np.uint8)

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


def _compact_rollout_cache(
    cache: Any,
    *,
    max_spatial_frames: int,
    recent_spatial_frames: int,
) -> None:
    """Bound generated latents and spatial memory while retaining old keyframes."""
    preds = getattr(cache, "preds", None)
    if isinstance(preds, list) and len(preds) > 1:
        preds[:] = preds[-1:]

    bank = getattr(cache, "spatial_bank", None)
    if bank is None:
        return
    pixels = bank.pixels
    frame_indices = bank.frame_indices
    depths = bank.depths
    total = len(frame_indices)
    if len(pixels) != total or len(depths) != total:
        raise RuntimeError("AlayaWorld spatial bank members have different lengths")
    if total <= max_spatial_frames:
        return

    recent_count = min(recent_spatial_frames, max_spatial_frames)
    historical_count = total - recent_count
    historical_budget = max_spatial_frames - recent_count
    if historical_budget == 1:
        historical = [0]
    elif historical_budget > 1:
        historical = [
            index * (historical_count - 1) // (historical_budget - 1)
            for index in range(historical_budget)
        ]
    else:
        historical = []
    keep = historical + list(range(historical_count, total))

    bank.pixels = [pixels[index] for index in keep]
    bank.frame_indices = [frame_indices[index] for index in keep]
    bank.depths = [depths[index] for index in keep]
    world_points = getattr(bank, "world_points", None)
    if isinstance(world_points, dict):
        bank.world_points = {
            new_index: world_points[old_index]
            for new_index, old_index in enumerate(keep)
            if old_index in world_points
        }


def _read_config(config_path: Path | None) -> _Config:
    """Read and validate the AlayaWorld adapter YAML."""
    if config_path is None:
        raise ValueError("AlayaWorld requires runtime.config in reactor.yaml")
    document = yaml.safe_load(config_path.read_text())
    if not isinstance(document, dict):
        raise ValueError(f"{config_path}: expected a YAML mapping")
    source = _mapping(document.get("source"), "source")
    assets = _mapping(document.get("assets"), "assets")
    inference = _mapping(document.get("inference"), "inference")
    inputs = _mapping(document.get("inputs"), "inputs")
    motion = _mapping(document.get("motion"), "motion")
    decode = _mapping(document.get("decode"), "decode")
    memory = _mapping(document.get("memory"), "memory")
    da3_source = _mapping(assets.get("da3_source"), "assets.da3_source")
    compile_mode = str(inference.get("compile", "reduce-overhead"))
    if compile_mode not in _COMPILE_MODES:
        raise ValueError(f"inference.compile must be one of {', '.join(_COMPILE_MODES)}")
    attention_backend = str(inference.get("attention_backend", "pytorch"))
    if attention_backend not in _ATTENTION_BACKENDS:
        raise ValueError(
            f"inference.attention_backend must be one of {', '.join(_ATTENTION_BACKENDS)}"
        )
    overlap = int(decode.get("overlap_latents", 6))
    if overlap <= 0:
        raise ValueError("decode.overlap_latents must be positive")
    max_spatial_frames = int(memory.get("max_spatial_frames", 320))
    recent_spatial_frames = int(memory.get("recent_spatial_frames", 160))
    if max_spatial_frames < 10:
        raise ValueError("memory.max_spatial_frames must be at least 10")
    if not 1 <= recent_spatial_frames <= max_spatial_frames:
        raise ValueError("memory.recent_spatial_frames must be between 1 and max_spatial_frames")
    motion_rates = {
        "strafe_units_per_second": float(motion.get("strafe_units_per_second", 0.126)),
        "vertical_units_per_second": float(motion.get("vertical_units_per_second", 0.261)),
        "forward_units_per_second": float(motion.get("forward_units_per_second", 1.905)),
        "pitch_degrees_per_second": float(motion.get("pitch_degrees_per_second", 4.039)),
        "yaw_degrees_per_second": float(motion.get("yaw_degrees_per_second", 9.375)),
        "roll_degrees_per_second": float(motion.get("roll_degrees_per_second", 4.094)),
    }
    for name, value in motion_rates.items():
        if value <= 0:
            raise ValueError(f"motion.{name} must be positive")
    source_path = _path(config_path.parent, source["path"])
    taehv_raw = assets.get("taehv")
    taehv_path = _path(source_path, taehv_raw) if taehv_raw else None
    random_inputs_raw = inputs.get("random_images")
    if not isinstance(random_inputs_raw, list) or not random_inputs_raw:
        raise ValueError("inputs.random_images must be a non-empty YAML list")
    return _Config(
        source_path=source_path,
        source_url=_repository_url(source.get("url"), "source.url"),
        source_revision=_revision(source.get("revision"), "source.revision"),
        upstream_config=_path(source_path, inference["config"]),
        upload_template=_path(source_path, inputs["upload_template"]),
        random_inputs=tuple(_path(source_path, value) for value in random_inputs_raw),
        model=_asset(source_path, assets.get("model"), "assets.model"),
        gemma=_asset(source_path, assets.get("gemma"), "assets.gemma"),
        da3_source_path=_path(source_path, da3_source["path"]),
        da3_source_url=_repository_url(da3_source.get("url"), "assets.da3_source.url"),
        da3_source_revision=_revision(da3_source.get("revision"), "assets.da3_source.revision"),
        da3_model=_asset(source_path, assets.get("da3_model"), "assets.da3_model"),
        da3_cache=_path(source_path, assets["da3_cache"]),
        seed=int(inference.get("seed", 1234)),
        compile_mode=compile_mode,
        attention_backend=attention_backend,
        flex_attention=bool(inference.get("flex_attention", True)),
        ttc=bool(inference.get("ttc", False)),
        bank_taehv=bool(inference.get("bank_taehv", False)),
        taehv_path=taehv_path,
        decode_overlap_latents=overlap,
        max_spatial_frames=max_spatial_frames,
        recent_spatial_frames=recent_spatial_frames,
        strafe_units_per_second=motion_rates["strafe_units_per_second"],
        vertical_units_per_second=motion_rates["vertical_units_per_second"],
        forward_units_per_second=motion_rates["forward_units_per_second"],
        pitch_degrees_per_second=motion_rates["pitch_degrees_per_second"],
        yaw_degrees_per_second=motion_rates["yaw_degrees_per_second"],
        roll_degrees_per_second=motion_rates["roll_degrees_per_second"],
    )


def _mapping(value: object, name: str) -> dict[str, Any]:
    """Return a YAML mapping or raise a precise configuration error."""
    if not isinstance(value, dict):
        raise ValueError(f"{name} must be a YAML mapping")
    return cast(dict[str, Any], value)


def _asset(source_path: Path, value: object, name: str) -> _Asset:
    """Read one local path and immutable public repository identity."""
    document = _mapping(value, name)
    repo_id = str(document.get("repo_id", ""))
    if "/" not in repo_id:
        raise ValueError(f"{name}.repo_id must identify a public repository")
    return _Asset(
        path=_path(source_path, document["path"]),
        repo_id=repo_id,
        revision=_revision(document.get("revision"), f"{name}.revision"),
    )


def _path(base_path: Path, value: object) -> Path:
    """Resolve a configured path relative to its owning directory."""
    path = Path(str(value)).expanduser()
    if not path.is_absolute():
        path = base_path / path
    return path.resolve()


def _revision(value: object, name: str) -> str:
    """Return an immutable 40-character revision."""
    revision = str(value or "")
    if _REVISION_PATTERN.fullmatch(revision) is None:
        raise ValueError(f"{name} must be a 40-character commit revision")
    return revision


def _repository_url(value: object, name: str) -> str:
    """Return the HTTPS URL for a public source repository."""
    url = str(value or "")
    if not url.startswith("https://"):
        raise ValueError(f"{name} must be a public HTTPS URL")
    return url


def _verify_repository_revision(path: Path, expected: str, name: str) -> None:
    """Require a local Git checkout to match its configured revision."""
    actual = _run_git(["-C", str(path), "rev-parse", "HEAD"], name).stdout.strip()
    if actual != expected:
        raise RuntimeError(f"{name} revision is {actual}; expected {expected}")


def _prepare_runtime_assets(config: _Config) -> None:
    """Download each missing public source checkout and model asset."""
    _ensure_git_checkout(
        config.source_path,
        url=config.source_url,
        revision=config.source_revision,
        name="AlayaWorld source",
    )
    _ensure_hf_file(config.model, name="AlayaWorld merged checkpoint")
    _ensure_hf_snapshot(config.gemma, name="Gemma text encoder")
    _ensure_git_checkout(
        config.da3_source_path,
        url=config.da3_source_url,
        revision=config.da3_source_revision,
        name="Depth-Anything-3 source",
    )
    _ensure_hf_snapshot(
        config.da3_model,
        name="Depth-Anything-3 checkpoint",
        cache_dir=config.da3_cache / "hub",
    )


def _ensure_git_checkout(path: Path, *, url: str, revision: str, name: str) -> None:
    """Clone a missing public repository and require its pinned revision."""
    if path.exists():
        _verify_repository_revision(path, revision, name)
        return
    logger.info("downloading source checkout", asset=name, url=url, revision=revision)
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix=".reactor-download-", dir=path.parent) as temporary:
        checkout = Path(temporary) / "checkout"
        _run_git(["clone", "--filter=blob:none", "--no-checkout", url, str(checkout)], name)
        _run_git(["-C", str(checkout), "checkout", "--detach", revision], name)
        with suppress(FileExistsError):
            checkout.rename(path)
    _verify_repository_revision(path, revision, name)


def _run_git(arguments: list[str], name: str) -> subprocess.CompletedProcess[str]:
    """Run Git and report an actionable public-resource error."""
    try:
        return subprocess.run(
            ["git", *arguments],
            check=True,
            capture_output=True,
            text=True,
        )
    except FileNotFoundError as error:
        raise RuntimeError(f"Git is required to download {name}") from error
    except subprocess.CalledProcessError as error:
        detail = error.stderr.strip() or error.stdout.strip() or "Git command failed"
        raise RuntimeError(f"Unable to prepare {name}: {detail}") from error


def _ensure_hf_file(asset: _Asset, *, name: str) -> None:
    """Download one missing file from a pinned Hugging Face revision."""
    if asset.path.is_file() and asset.path.stat().st_size > 0:
        return
    logger.info(
        "downloading model file",
        asset=name,
        repo_id=asset.repo_id,
        revision=asset.revision,
    )
    asset.path.parent.mkdir(parents=True, exist_ok=True)
    try:
        downloaded = _hf_hub_download(
            repo_id=asset.repo_id,
            filename=asset.path.name,
            revision=asset.revision,
            local_dir=asset.path.parent,
        )
    except Exception as error:
        _raise_hf_download_error(name, asset.repo_id, error)
    if downloaded.resolve() != asset.path.resolve() or not asset.path.is_file():
        raise RuntimeError(f"{name} download did not create {asset.path}")


def _ensure_hf_snapshot(
    asset: _Asset,
    *,
    name: str,
    cache_dir: Path | None = None,
) -> None:
    """Download one missing Hugging Face repository snapshot."""
    if asset.path.exists() and any(asset.path.iterdir()):
        return
    logger.info(
        "downloading model snapshot",
        asset=name,
        repo_id=asset.repo_id,
        revision=asset.revision,
    )
    try:
        if cache_dir is None:
            asset.path.mkdir(parents=True, exist_ok=True)
            _hf_snapshot_download(
                repo_id=asset.repo_id,
                revision=asset.revision,
                local_dir=asset.path,
            )
        else:
            cache_dir.mkdir(parents=True, exist_ok=True)
            _hf_snapshot_download(
                repo_id=asset.repo_id,
                revision=asset.revision,
                cache_dir=cache_dir,
            )
    except Exception as error:
        _raise_hf_download_error(name, asset.repo_id, error)
    if not asset.path.exists() or not any(asset.path.iterdir()):
        raise RuntimeError(f"{name} download did not create {asset.path}")


def _hf_hub_download(
    *,
    repo_id: str,
    filename: str,
    revision: str,
    local_dir: Path,
) -> Path:
    """Download one Hugging Face file without importing model dependencies eagerly."""
    hugging_face = importlib.import_module("huggingface_hub")
    return Path(
        str(
            hugging_face.hf_hub_download(
                repo_id=repo_id,
                filename=filename,
                revision=revision,
                local_dir=local_dir,
            )
        )
    )


def _hf_snapshot_download(
    *,
    repo_id: str,
    revision: str,
    local_dir: Path | None = None,
    cache_dir: Path | None = None,
) -> Path:
    """Download one pinned Hugging Face snapshot lazily."""
    hugging_face = importlib.import_module("huggingface_hub")
    return Path(
        str(
            hugging_face.snapshot_download(
                repo_id=repo_id,
                revision=revision,
                local_dir=local_dir,
                cache_dir=cache_dir,
            )
        )
    )


def _raise_hf_download_error(name: str, repo_id: str, error: Exception) -> Never:
    """Raise an authentication-aware Hugging Face download error."""
    raise RuntimeError(
        f"Unable to download {name} from {repo_id}. Check network access and run "
        "`hf auth login` if the repository is gated."
    ) from error


def _validate_runtime_paths(config: _Config) -> None:
    """Require every prepared source and model asset before loading."""
    required = {
        "AlayaWorld source": config.source_path,
        "AlayaWorld inference config": config.upstream_config,
        "AlayaWorld merged checkpoint": config.model.path,
        "Gemma text encoder": config.gemma.path,
        "Depth-Anything-3 source": config.da3_source_path,
        "Depth-Anything-3 checkpoint": config.da3_model.path,
        "DA3 checkpoint cache": config.da3_cache,
    }
    for name, path in required.items():
        if not path.exists():
            raise FileNotFoundError(f"{name} not found: {path}")
    _validate_scene_triplet(config.upload_template, "upload template")
    for index, path in enumerate(config.random_inputs):
        _validate_scene_triplet(path, f"random image {index}")
    if config.bank_taehv and (config.taehv_path is None or not config.taehv_path.is_file()):
        raise FileNotFoundError("inference.bank_taehv requires assets.taehv")


def _scene_prefix(path: Path) -> Path:
    """Return a scene prefix from a prefix or one of its triplet members."""
    value = str(path)
    for suffix in _SCENE_MEMBER_SUFFIXES:
        if value.endswith(suffix):
            return Path(value[: -len(suffix)])
    return path


def _scene_camera_path(path: Path) -> Path:
    """Return the camera metadata member for a configured scene."""
    return Path(f"{_scene_prefix(path)}_camera.pt")


def _scene_prompt_path(path: Path) -> Path:
    """Return the prompt member for a configured scene."""
    return Path(f"{_scene_prefix(path)}_prompt.txt")


def _scene_image_path(path: Path) -> Path:
    """Return the first supported still-image member for a configured scene."""
    prefix = _scene_prefix(path)
    for extension in _SCENE_IMAGE_EXTENSIONS:
        candidate = Path(f"{prefix}_image{extension}")
        if candidate.is_file():
            return candidate
    raise FileNotFoundError(f"scene image not found for prefix: {prefix}")


def _validate_scene_triplet(path: Path, name: str) -> None:
    """Require the image, camera, and prompt files used by one scene."""
    members = {
        "image": _scene_image_path(path),
        "camera": _scene_camera_path(path),
        "prompt": _scene_prompt_path(path),
    }
    for member_name, member_path in members.items():
        if not member_path.is_file():
            raise FileNotFoundError(f"{name} {member_name} not found: {member_path}")


def _load_scene_metadata(path: Path, torch_module: Any) -> dict[str, Any]:
    """Load a fresh metadata mapping from the upload camera template."""
    value = torch_module.load(
        _scene_camera_path(path),
        map_location="cpu",
        weights_only=False,
    )
    if not isinstance(value, dict) or "cam_c2w" not in value:
        raise ValueError("upload camera template must contain cam_c2w metadata")
    return cast(dict[str, Any], value)


def _validate_uploaded_image(image: UploadedFile) -> None:
    """Reject oversized, mislabeled, or undecodable uploaded image bytes."""
    if not image.mime_type.startswith("image/"):
        raise CommandError("unsupported_media", f"{image.name} is not an image.")
    if not image.data:
        raise CommandError("invalid_image", f"{image.name} is empty.")
    if image.size > _UPLOAD_MAX_BYTES:
        raise CommandError(
            "image_too_large",
            f"{image.name} exceeds the {_UPLOAD_MAX_BYTES // (1024 * 1024)} MiB limit.",
        )
    try:
        with Image.open(io.BytesIO(image.data)) as decoded:
            image_format = decoded.format or ""
            width, height = decoded.size
            if image_format not in _UPLOAD_FORMATS:
                raise CommandError(
                    "unsupported_media",
                    f"{image.name} must be JPEG, PNG, WebP, or BMP.",
                )
            if width <= 0 or height <= 0 or width * height > _UPLOAD_MAX_PIXELS:
                raise CommandError(
                    "image_too_large",
                    f"{image.name} exceeds the {_UPLOAD_MAX_PIXELS}-pixel limit.",
                )
            decoded.verify()
    except CommandError:
        raise
    except (Image.DecompressionBombError, UnidentifiedImageError, OSError) as error:
        raise CommandError("invalid_image", f"{image.name} cannot be decoded.") from error


def _uploaded_image_video(
    image: UploadedFile,
    metadata: dict[str, Any],
    *,
    target_hw: tuple[int, int],
    torch_module: Any,
) -> Any:
    """Decode, center-crop, and repeat an upload over the camera template."""
    target_height, target_width = target_hw
    with Image.open(io.BytesIO(image.data)) as decoded:
        oriented = ImageOps.exif_transpose(decoded).convert("RGB")
        fitted = ImageOps.fit(
            oriented,
            (target_width, target_height),
            method=Image.Resampling.LANCZOS,
        )
        pixels = np.array(fitted, dtype=np.uint8, copy=True)
    frame = torch_module.from_numpy(pixels).permute(2, 0, 1).float() / 127.5 - 1.0
    frame_count = int(_camera_frames(metadata["cam_c2w"]).shape[0])
    return frame.unsqueeze(0).expand(frame_count, -1, -1, -1).contiguous()


def _load_upstream_modules(source_path: Path) -> dict[str, Any]:
    """Import the unmodified public AlayaWorld inference surface lazily."""
    source = str(source_path)
    if source not in sys.path:
        sys.path.insert(0, source)
    rollout = importlib.import_module("flash_alaya.utils.rollout_utils")
    return {
        "torch": importlib.import_module("torch"),
        "load_config": importlib.import_module("flash_alaya.alaya.config.loader").load_config,
        "pipeline_type": importlib.import_module("flash_alaya.utils.pipeline").FlashAlayaPipeline,
        "load_input_sample": rollout.load_input_sample,
        "check_input_resolution": rollout.check_input_resolution,
        "plan_rollout": rollout.plan_rollout,
        "build_engine": rollout.build_engine,
        "apply_da3_robust_scale": importlib.import_module(
            "inference.da3_patch"
        ).apply_da3_robust_scale,
        "pytorch_attention": importlib.import_module(
            "flash_alaya.ltx2.modules.attention"
        ).AttentionFunction.PYTORCH,
    }


def _set_attention_backend(engine: Any, attention_function: Any) -> int:
    """Select an upstream attention callable on every loaded attention module."""
    changed = 0
    for root in (engine.transformer, engine.text_encoder):
        if root is None:
            continue
        for module in root.modules():
            if hasattr(module, "attention_function"):
                module.attention_function = attention_function
                changed += 1
    if changed == 0:
        raise RuntimeError("AlayaWorld exposed no configurable attention modules")
    return changed


def _camera_frames(camera: Any) -> Any:
    """Return camera poses as ``[F, 4, 4]`` from batched or unbatched input."""
    if camera.dim() == 3:
        return camera
    if camera.dim() == 4:
        return camera[0]
    raise ValueError(f"cam_c2w must be [F,4,4] or [B,F,4,4], got {tuple(camera.shape)}")


def _ensure_camera_capacity(camera: Any, frame_count: int, torch_module: Any) -> Any:
    """Extend a camera tensor by repeating its final pose when needed."""
    current = int(_camera_frames(camera).shape[0])
    if current >= frame_count:
        return camera
    missing = frame_count - current
    time_axis = 0 if camera.dim() == 3 else 1
    tail = camera[-1:] if camera.dim() == 3 else camera[:, -1:]
    repeats = [1] * camera.dim()
    repeats[time_axis] = missing
    return torch_module.cat([camera, tail.repeat(*repeats)], dim=time_axis)
