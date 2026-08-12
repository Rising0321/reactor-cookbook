"""Serve public Matrix-Game-3.5 distilled inference through Reactor Runtime.

The adapter accepts a reference image and text prompt, then expands normalized
six-axis camera motion into Matrix's native camera-to-world matrices. A
persistent worker owns the upstream 5B model and keeps its causal rollout alive
while generating 12-frame chunks. Each chunk uses the latest motion state while
reusing the rollout's KV cache, dynamic visual context, and Patch Memory.
"""

from __future__ import annotations

import asyncio
import importlib
import io
import os
import re
import subprocess
from collections.abc import AsyncGenerator
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Any, Protocol, cast

import av
import numpy as np
import yaml

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
    from camera_motion import (
        CameraMotionPlanner,
        MotionConfig,
    )
    from upstream_backend import (
        MatrixWorkerBackend,
        WorkerSettings,
    )
else:
    module_prefix = f"{__package__}." if __package__ else ""
    camera_motion = importlib.import_module(f"{module_prefix}camera_motion")
    upstream_backend = importlib.import_module(f"{module_prefix}upstream_backend")
    CameraMotionPlanner = camera_motion.CameraMotionPlanner
    MotionConfig = camera_motion.MotionConfig
    MatrixWorkerBackend = upstream_backend.MatrixWorkerBackend
    WorkerSettings = upstream_backend.WorkerSettings

logger = get_logger(__name__)

FPS = 16
_CAMERA_POSES_PER_CHUNK = 12
_OUTPUT_FRAMES_PER_CHUNK = 12
_REVISION_PATTERN = re.compile(r"[0-9a-f]{40}")
_SOURCE_ENV = "MATRIX_GAME_3_5_PATH"
_WORKER_PYTHON = Path(".venv/bin/python")
_INFERENCE_CONFIG = Path("configs/infer_distilled.yaml")
_CHECKPOINT_PATH = Path("checkpoints/Matrix-Game-3.5-Distilled/first-person.safetensors")
_WAN_PATH = Path("checkpoints/Wan2.2-TI2V-5B")
_TOKENIZER_PATH = _WAN_PATH / "google/umt5-xxl"
_DEPTH_PATH = Path("checkpoints/DA3NESTED-GIANT-LARGE-1.1")
_DEFAULT_SAMPLE = Path("samples/first_person/case_0")
_ANCHOR_IMAGE = _DEFAULT_SAMPLE / "input.png"
_CAMERA = _DEFAULT_SAMPLE / "camera.npz"
_PROMPT_FILE = _DEFAULT_SAMPLE / "prompt.txt"
_WAN_REQUIRED_FILES = (
    "Wan2.2_VAE.pth",
    "diffusion_pytorch_model-00001-of-00003.safetensors",
    "diffusion_pytorch_model-00002-of-00003.safetensors",
    "diffusion_pytorch_model-00003-of-00003.safetensors",
    "diffusion_pytorch_model.safetensors.index.json",
    "models_t5_umt5-xxl-enc-bf16.pth",
    "google/umt5-xxl/spiece.model",
    "google/umt5-xxl/tokenizer.json",
    "google/umt5-xxl/tokenizer_config.json",
)
_DEPTH_REQUIRED_FILES = ("config.json", "model.safetensors")
_UPLOAD_MAX_BYTES = 25 * 1024 * 1024
_UPLOAD_MAX_PIXELS = 100_000_000
_UPLOAD_CODECS = {"bmp", "mjpeg", "png", "webp"}


@dataclass(frozen=True)
class _Asset:
    """Describe one pinned public model asset and its local location."""

    path: Path
    repo_id: str
    revision: str


@dataclass(frozen=True)
class _Config:
    """Hold validated Matrix adapter settings."""

    worker_python: Path
    source_path: Path
    source_revision: str
    inference_config: Path
    checkpoint: _Asset
    wan: _Asset
    tokenizer_dir: Path
    depth: _Asset
    anchor_image: Path
    camera: Path
    prompt_file: Path
    seed: int
    max_chunks: int
    translation_meters_per_second: float
    rotation_degrees_per_second: float


class _Backend(Protocol):
    """Define the blocking model operations used by the Reactor loop."""

    def reset(
        self,
        seed: int,
        anchor_image: Path | UploadedFile,
        prompt: str,
    ) -> None:
        """Reset the causal rollout from an image and text condition."""

    def generate_chunk(
        self,
        trajectory_c2w: np.ndarray,
        seed: int,
        prompt: str,
    ) -> np.ndarray:
        """Generate one RGB chunk for camera and text conditions."""


class MatrixGame35Output(Output):
    """Carry one generated Matrix-Game-3.5 RGB frame."""

    main_video: Video


class ImageSelected(ModelMessage):
    """Confirm the image selected as the next rollout origin."""

    source: str = MessageField(description="Whether the image was uploaded or built in.")
    filename: str = MessageField(description="Name of the selected image.")


class PromptQueued(ModelMessage):
    """Confirm the prompt queued for a causal chunk boundary."""

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


class MatrixGame35State(InputState):
    """Hold prompt, six-axis camera, and generation controls for one session."""

    prompt: str = InputField(
        default="",
        max_length=4096,
        description="Prompt sampled at the next causal chunk boundary.",
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
        description="Normalized counterclockwise-to-clockwise camera roll.",
    )
    paused: bool = InputField(default=False, description="Pause chunk generation.")
    _step_requested: bool = False
    _restart_requested: bool = True


class MatrixGame35(ReactorPipeline):
    """Generate an image-, prompt-, and camera-controllable Matrix world."""

    state: MatrixGame35State
    output: MatrixGame35Output
    fps = FPS
    buffer_size = 1

    def __init__(self) -> None:
        super().__init__()
        self._config: _Config | None = None
        self._backend: _Backend | None = None
        self._planner: CameraMotionPlanner | None = None
        self._selected_input: Path | UploadedFile | None = None
        self._default_prompt = ""
        self._seed = 0
        self._chunk_index = 0
        self._chunk_in_flight = False

    def load(self, config_path: Path | None) -> None:
        """Validate configuration and load Matrix weights in a persistent worker.

        Args:
            config_path: Path to ``matrix_game_3_5.yaml`` from ``reactor.yaml``.
        """
        config = _read_config(config_path)
        _verify_source_revision(config.source_path, config.source_revision)
        _validate_bootstrap_paths(config)
        _restore_default_sample(config)
        _ensure_model_assets(config)
        _validate_runtime_paths(config)
        initial_pose, intrinsics = _load_initial_camera(config.camera)
        self._config = config
        self._default_prompt = config.prompt_file.read_text(encoding="utf-8").strip()
        if not self._default_prompt:
            raise ValueError(f"prompt file is empty: {config.prompt_file}")
        self._seed = config.seed
        self._planner = CameraMotionPlanner(
            initial_pose,
            MotionConfig(
                fps=FPS,
                translation_meters_per_second=config.translation_meters_per_second,
                rotation_degrees_per_second=config.rotation_degrees_per_second,
            ),
        )
        self._backend = MatrixWorkerBackend(
            WorkerSettings(
                python_executable=config.worker_python,
                source_path=config.source_path,
                inference_config=config.inference_config,
                checkpoint=config.checkpoint.path,
                wan_dir=config.wan.path,
                tokenizer_dir=config.tokenizer_dir,
                da3_dir=config.depth.path,
                anchor_image=config.anchor_image,
                default_camera=config.camera,
                prompt_file=config.prompt_file,
                seed=config.seed,
                max_chunks=config.max_chunks,
            ),
            intrinsics,
        )
        logger.info(
            "Matrix-Game-3.5 model ready",
            source_revision=config.source_revision,
            checkpoint_revision=config.checkpoint.revision,
            fps=FPS,
        )

    @session_started
    def on_session_started(self) -> None:
        """Start each session from the configured anchor and seed."""
        config = self._config
        if config is None:
            raise RuntimeError("Matrix-Game-3.5 was not loaded")
        self._selected_input = config.anchor_image
        self.state.prompt = self._default_prompt
        self._seed = config.seed
        self._clear_controls()
        self.state.paused = False
        self.state._step_requested = False
        self.state._restart_requested = True
        self._chunk_index = 0
        self._chunk_in_flight = False

    @session_ended
    def on_session_ended(self) -> None:
        """Discard controls and generated state when the session ends."""
        self._clear_controls()
        self.state._step_requested = False
        self.state._restart_requested = True
        self._selected_input = None
        self._chunk_index = 0
        self._chunk_in_flight = False

    @disconnected
    def on_disconnected(self) -> None:
        """Release controls when a client leaves."""
        self._clear_controls()

    @event(
        name="set_prompt",
        description="Apply a prompt while continuing the active causal rollout.",
    )
    def set_prompt(
        self,
        prompt: str = InputField(
            default="",
            max_length=4096,
            description="Non-empty text condition for the next chunk.",
        ),
    ) -> PromptQueued:
        """Queue a prompt and confirm the chunk expected to consume it."""
        normalized = prompt.strip()
        if not normalized:
            raise CommandError("prompt_required", "Matrix-Game-3.5 requires a prompt.")
        if self._selected_input is None:
            raise CommandError("image_required", "Select an image before setting a prompt.")
        self.state.prompt = normalized
        return PromptQueued(prompt=normalized, applies_to_chunk=self._next_control_chunk())

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
            description="Optional prompt; empty keeps the current prompt.",
        ),
    ) -> ImageSelected:
        """Validate an uploaded anchor and select it for a fresh causal rollout."""
        _validate_uploaded_image(image)
        normalized = prompt.strip() or self.state.prompt.strip() or self._default_prompt
        if not normalized:
            raise CommandError("prompt_required", "Matrix-Game-3.5 requires a prompt.")
        self._selected_input = image
        self.state.prompt = normalized
        self.state.paused = False
        self._request_restart()
        return ImageSelected(source="uploaded", filename=image.name)

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
            description="Normalized counterclockwise-to-clockwise camera roll.",
        ),
    ) -> CameraMotionChanged:
        """Queue roll motion and return the complete camera state."""
        self.state.roll = roll
        return self._camera_motion_changed()

    @event(name="set_paused", description="Pause or resume chunk generation and playback.")
    def set_paused(
        self,
        paused: bool = InputField(
            default=False,
            description="Pause before the next chunk boundary.",
        ),
    ) -> CameraMotionChanged:
        """Set pause state and return the released camera motion."""
        self.state.paused = paused
        self.state._step_requested = False
        self._clear_controls()
        return self._camera_motion_changed()

    @event(name="step", description="Generate and play one 12-frame chunk while paused.")
    def step(self) -> None:
        """Request exactly one chunk without leaving paused mode."""
        if self.state.paused:
            self.state._step_requested = True

    @event(name="reset", description="Reset the rolling world to its selected anchor image.")
    def reset(
        self,
        seed: int = InputField(
            default=-1,
            ge=-1,
            le=2_147_483_647,
            description="New rollout seed; -1 reuses the active seed.",
        ),
    ) -> CameraMotionChanged:
        """Request a reset and return the released camera motion."""
        if seed >= 0:
            self._seed = seed
        if self._selected_input is None:
            raise CommandError("image_required", "Select an image before resetting.")
        self._request_restart()
        return self._camera_motion_changed()

    async def inference(self) -> AsyncGenerator[object, None]:
        """Generate chunks off-loop and emit their frames at Matrix's native FPS."""
        backend = self._backend
        planner = self._planner
        if backend is None or planner is None:
            raise RuntimeError("Matrix-Game-3.5 was not loaded")

        while True:
            if self.state._restart_requested:
                selected_input = self._selected_input
                if selected_input is None:
                    yield Idle
                    continue
                prompt = self.state.prompt.strip()
                if not prompt:
                    raise RuntimeError("Matrix-Game-3.5 requires a non-empty prompt")
                self.state._restart_requested = False
                await asyncio.to_thread(
                    backend.reset,
                    self._seed,
                    selected_input,
                    prompt,
                )
                if self.state._restart_requested:
                    continue
                planner.reset()
                self._chunk_index = 0

            if self.state.paused and not self.state._step_requested:
                yield Idle
                continue

            self.state._step_requested = False
            trajectory = planner.plan_block(
                strafe=self.state.strafe,
                vertical=self.state.vertical,
                forward=self.state.forward,
                pitch=self.state.pitch,
                yaw=self.state.yaw,
                roll=self.state.roll,
                frame_count=_CAMERA_POSES_PER_CHUNK,
            )
            self._chunk_in_flight = True
            try:
                frames = await asyncio.to_thread(
                    backend.generate_chunk,
                    trajectory,
                    self._seed,
                    self.state.prompt,
                )
            finally:
                self._chunk_in_flight = False
            frames = _normalize_output_frames(frames)
            self._chunk_index += 1

            for frame in frames:
                if self.state._restart_requested:
                    break
                yield MatrixGame35Output(main_video=frame)

    def _clear_controls(self) -> None:
        """Return every camera axis to neutral."""
        self.state.forward = 0.0
        self.state.strafe = 0.0
        self.state.vertical = 0.0
        self.state.pitch = 0.0
        self.state.yaw = 0.0
        self.state.roll = 0.0

    def _request_restart(self) -> None:
        """Queue a fresh causal rollout and release active camera motion."""
        self._clear_controls()
        self.state._step_requested = False
        self.state._restart_requested = True

    def _next_control_chunk(self) -> int:
        """Return the one-based chunk expected to consume new camera motion."""
        if self.state._restart_requested:
            return 1
        return self._chunk_index + 1 + int(self._chunk_in_flight)

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


def _read_config(config_path: Path | None) -> _Config:
    """Read and validate the Matrix adapter YAML."""
    if config_path is None:
        raise ValueError("Matrix-Game-3.5 requires runtime.config in reactor.yaml")
    document = yaml.safe_load(config_path.read_text())
    if not isinstance(document, dict):
        raise ValueError(f"{config_path}: expected a YAML mapping")

    source = _mapping(document.get("source"), "source")
    source_revision = _revision(source.get("revision"), "source.revision")
    assets = _mapping(document.get("assets"), "assets")
    inference = _mapping(document.get("inference"), "inference")
    motion = _mapping(document.get("motion"), "motion")
    stream = _mapping(document.get("stream"), "stream")
    source_path = _source_path(config_path, source["path"])
    checkpoint = _asset(
        source_path,
        assets.get("checkpoint"),
        "assets.checkpoint",
        _CHECKPOINT_PATH,
    )
    wan = _asset(source_path, assets.get("wan"), "assets.wan", _WAN_PATH)
    depth = _asset(source_path, assets.get("depth"), "assets.depth", _DEPTH_PATH)
    translation_speed = float(motion.get("translation_meters_per_second", 1.5))
    rotation_speed = float(motion.get("rotation_degrees_per_second", 45.0))
    if translation_speed <= 0:
        raise ValueError("motion.translation_meters_per_second must be positive")
    if rotation_speed <= 0:
        raise ValueError("motion.rotation_degrees_per_second must be positive")
    max_chunks = int(stream.get("max_chunks", 512))
    if max_chunks < 8:
        raise ValueError("stream.max_chunks must be at least 8")

    return _Config(
        worker_python=source_path / _WORKER_PYTHON,
        source_path=source_path,
        source_revision=source_revision,
        inference_config=source_path / _INFERENCE_CONFIG,
        checkpoint=checkpoint,
        wan=wan,
        tokenizer_dir=source_path / _TOKENIZER_PATH,
        depth=depth,
        anchor_image=source_path / _ANCHOR_IMAGE,
        camera=source_path / _CAMERA,
        prompt_file=source_path / _PROMPT_FILE,
        seed=int(inference.get("seed", 3407)),
        max_chunks=max_chunks,
        translation_meters_per_second=translation_speed,
        rotation_degrees_per_second=rotation_speed,
    )


def _mapping(value: object, name: str) -> dict[str, Any]:
    """Return a YAML mapping or raise a precise configuration error."""
    if not isinstance(value, dict):
        raise ValueError(f"{name} must be a YAML mapping")
    return cast(dict[str, Any], value)


def _asset(source_path: Path, value: object, name: str, relative_path: Path) -> _Asset:
    """Return one pinned public asset under the Matrix source root."""
    document = _mapping(value, name)
    repo_id = str(document.get("repo_id", ""))
    if "/" not in repo_id:
        raise ValueError(f"{name}.repo_id must identify a public repository")
    return _Asset(
        path=source_path / relative_path,
        repo_id=repo_id,
        revision=_revision(document.get("revision"), f"{name}.revision"),
    )


def _revision(value: object, name: str) -> str:
    """Return one full immutable Git-style revision."""
    revision = str(value or "")
    if not _REVISION_PATTERN.fullmatch(revision):
        raise ValueError(f"{name} must be a full 40-character revision")
    return revision


def _source_path(config_path: Path, value: object) -> Path:
    """Resolve the single configured Matrix checkout root."""
    configured = os.environ.get(_SOURCE_ENV)
    path = Path(configured if configured else str(value)).expanduser()
    candidate = path if path.is_absolute() else config_path.parent / path
    return Path(os.path.abspath(candidate))


def _verify_source_revision(source_path: Path, expected: str) -> None:
    """Require the public Matrix checkout to match its configured revision."""
    if not (source_path / ".git").exists():
        raise RuntimeError(f"Matrix source at {source_path} must be a Git checkout")
    result = subprocess.run(
        ["git", "-C", str(source_path), "rev-parse", "HEAD"],
        check=True,
        capture_output=True,
        text=True,
    )
    actual = result.stdout.strip()
    if actual != expected:
        raise RuntimeError(f"Matrix source is {actual}; expected {expected}")


def _restore_default_sample(config: _Config) -> None:
    """Restore missing default inputs from the pinned public source revision."""
    relative_paths = (
        _ANCHOR_IMAGE,
        _CAMERA,
        _PROMPT_FILE,
    )
    missing = [path for path in relative_paths if not (config.source_path / path).is_file()]
    if not missing:
        return
    logger.info(
        "restoring Matrix default sample",
        files=[str(path) for path in missing],
        revision=config.source_revision,
    )
    try:
        subprocess.run(
            [
                "git",
                "-C",
                str(config.source_path),
                "checkout",
                config.source_revision,
                "--",
                *(str(path) for path in missing),
            ],
            check=True,
            capture_output=True,
            text=True,
        )
    except (FileNotFoundError, subprocess.CalledProcessError) as error:
        raise RuntimeError(
            "Unable to restore the default Matrix sample from the pinned source checkout"
        ) from error
    unresolved = [str(path) for path in missing if not (config.source_path / path).is_file()]
    if unresolved:
        raise RuntimeError(f"Matrix default sample remains incomplete: {unresolved}")


def _validate_bootstrap_paths(config: _Config) -> None:
    """Require the source files needed to restore inputs and download weights."""
    files = {
        "Matrix worker Python": config.worker_python,
        "Matrix inference config": config.inference_config,
    }
    for name, path in files.items():
        if not path.is_file():
            raise FileNotFoundError(f"{name} file does not exist: {path}")


def _ensure_model_assets(config: _Config) -> None:
    """Download every missing model snapshot at its pinned public revision."""
    _ensure_hf_snapshot(
        config,
        config.checkpoint,
        name="Matrix distilled checkpoint",
        local_dir=config.checkpoint.path.parent,
        required_files=(config.checkpoint.path,),
    )
    _ensure_hf_snapshot(
        config,
        config.wan,
        name="Wan2.2 base model",
        local_dir=config.wan.path,
        required_files=tuple(config.wan.path / path for path in _WAN_REQUIRED_FILES),
        ignore_patterns=("assets/*", "examples/*"),
    )
    _ensure_hf_snapshot(
        config,
        config.depth,
        name="Depth-Anything-3 model",
        local_dir=config.depth.path,
        required_files=tuple(config.depth.path / path for path in _DEPTH_REQUIRED_FILES),
    )


def _ensure_hf_snapshot(
    config: _Config,
    asset: _Asset,
    *,
    name: str,
    local_dir: Path,
    required_files: tuple[Path, ...],
    ignore_patterns: tuple[str, ...] = (),
) -> None:
    """Download a missing Hugging Face snapshot with the Matrix worker environment."""
    if all(_is_nonempty_file(path) for path in required_files):
        return
    if not config.worker_python.is_file():
        raise FileNotFoundError(
            f"Matrix worker Python does not exist: {config.worker_python}. "
            "Create the documented upstream environment before starting Reactor."
        )
    logger.info(
        "downloading Matrix model asset",
        asset=name,
        repo_id=asset.repo_id,
        revision=asset.revision,
        destination=str(local_dir),
    )
    downloader = Path(__file__).with_name("download_snapshot.py")
    command = [
        str(config.worker_python),
        str(downloader),
        "--repo-id",
        asset.repo_id,
        "--revision",
        asset.revision,
        "--local-dir",
        str(local_dir),
    ]
    for pattern in ignore_patterns:
        command.extend(("--ignore-pattern", pattern))
    try:
        subprocess.run(command, check=True)
    except subprocess.CalledProcessError as error:
        raise RuntimeError(
            f"Unable to download {name} from {asset.repo_id}. Check network access and "
            "run `hf auth login` if Hugging Face requests authentication."
        ) from error
    unresolved = [str(path) for path in required_files if not _is_nonempty_file(path)]
    if unresolved:
        raise RuntimeError(f"{name} download is incomplete; missing files: {unresolved}")


def _is_nonempty_file(path: Path) -> bool:
    """Return whether a model file exists and contains data."""
    return path.is_file() and path.stat().st_size > 0


def _validate_runtime_paths(config: _Config) -> None:
    """Require every prepared source, input, environment, and model asset."""
    files = {
        "Matrix worker Python": config.worker_python,
        "inference config": config.inference_config,
        "distilled checkpoint": config.checkpoint.path,
        "anchor image": config.anchor_image,
        "camera trajectory": config.camera,
        "prompt file": config.prompt_file,
    }
    directories = {
        "Matrix source": config.source_path,
        "Wan2.2 model": config.wan.path,
        "tokenizer": config.tokenizer_dir,
        "Depth-Anything-3 model": config.depth.path,
    }
    for label, path in files.items():
        if not path.is_file():
            raise FileNotFoundError(f"{label} file does not exist: {path}")
    for label, path in directories.items():
        if not path.is_dir():
            raise FileNotFoundError(f"{label} directory does not exist: {path}")


def _load_initial_camera(path: Path) -> tuple[np.ndarray, np.ndarray]:
    """Load the first c2w pose and intrinsics from a public Matrix camera NPZ."""
    with np.load(path) as archive:
        if "extrinsics_c2w" not in archive or "intrinsics" not in archive:
            raise ValueError(f"{path}: expected extrinsics_c2w and intrinsics arrays")
        extrinsics = np.asarray(archive["extrinsics_c2w"], dtype=np.float32)
        intrinsics = np.asarray(archive["intrinsics"], dtype=np.float32)
    if extrinsics.ndim != 3 or extrinsics.shape[1:] != (4, 4):
        raise ValueError(f"{path}: extrinsics_c2w must have shape (N, 4, 4)")
    if int(extrinsics.shape[0]) == 0:
        raise ValueError(f"{path}: camera trajectory is empty")
    return np.ascontiguousarray(extrinsics[0]), intrinsics


def _normalize_output_frames(value: np.ndarray) -> np.ndarray:
    """Return exactly one contiguous uint8 RGB Matrix output chunk."""
    frames = np.asarray(value)
    if frames.ndim != 4 or frames.shape[-1] != 3:
        raise RuntimeError(f"Matrix output must have shape (T, H, W, 3), got {frames.shape}")
    if int(frames.shape[0]) != _OUTPUT_FRAMES_PER_CHUNK:
        raise RuntimeError(
            f"Matrix output must contain {_OUTPUT_FRAMES_PER_CHUNK} frames, "
            f"got {int(frames.shape[0])}"
        )
    if frames.dtype != np.uint8:
        frames = np.clip(frames, 0, 255).astype(np.uint8)
    return np.ascontiguousarray(frames)


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
        with av.open(io.BytesIO(image.data), mode="r") as container:
            if not container.streams.video:
                raise CommandError("invalid_image", f"{image.name} has no image stream.")
            stream = container.streams.video[0]
            codec = stream.codec_context.name
            width = int(stream.codec_context.width)
            height = int(stream.codec_context.height)
            if codec not in _UPLOAD_CODECS:
                raise CommandError(
                    "unsupported_media",
                    f"{image.name} must be JPEG, PNG, WebP, or BMP.",
                )
            if width <= 0 or height <= 0 or width * height > _UPLOAD_MAX_PIXELS:
                raise CommandError(
                    "image_too_large",
                    f"{image.name} exceeds the {_UPLOAD_MAX_PIXELS}-pixel limit.",
                )
            frame = next(container.decode(stream), None)
            if frame is None or frame.width != width or frame.height != height:
                raise CommandError("invalid_image", f"{image.name} cannot be decoded.")
    except CommandError:
        raise
    except (av.FFmpegError, EOFError, OSError, ValueError) as error:
        raise CommandError("invalid_image", f"{image.name} cannot be decoded.") from error
