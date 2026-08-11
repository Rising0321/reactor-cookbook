"""Serve the DIAMOND Counter-Strike world model through Reactor Runtime.

The adapter keeps DIAMOND's inference implementation intact and translates
Reactor commands into the keyboard and mouse action representation expected by
the upstream CSGO model. It produces one generated RGB frame on ``main_video``
for every world-model step.
"""

from __future__ import annotations

import importlib
import math
import os
import sys
from collections.abc import Callable, Iterator
from dataclasses import dataclass
from io import BytesIO
from pathlib import Path
from typing import Any

import numpy as np
import yaml

from reactor_runtime import (
    InputField,
    InputState,
    MessageField,
    ModelMessage,
    Output,
    ReactorPipeline,
    UploadedFile,
    Video,
    event,
)
from reactor_runtime.log import get_logger

logger = get_logger(__name__)

_ROOT = Path(__file__).resolve().parent
_UPSTREAM_ENV = "DIAMOND_PATH"
_KEYS = ["w", "a", "s", "d", "space", "ctrl", "shift", "1", "2", "3", "r"]
_MOUSE_BUTTONS = ["left", "right"]
_CONTROLLERS = ["human", "replay"]
_DELTA_X_MIN = -1000.0
_DELTA_X_MAX = 1000.0
_DELTA_Y_MIN = -200.0
_DELTA_Y_MAX = 200.0


@dataclass(frozen=True)
class _Config:
    """Hold the adapter settings read from ``diamond.yaml``."""

    repo_id: str
    revision: str
    device: str
    profile: str
    seed: int


@dataclass(frozen=True)
class _PreparedScene:
    """Hold one device-ready initial condition for the next reset."""

    obs: Any
    obs_full_res: Any
    act: Any
    next_act: Any | None


class DiamondOutput(Output):
    """Carry the generated Counter-Strike frame."""

    main_video: Video


class ActionChanged(ModelMessage):
    """Describe the latest native control input received from a client."""

    controller: str = MessageField(description="Controller that produced the action.")
    pressed_keys: list[str] = MessageField(description="Native keys held for this action.")
    pressed_mouse_buttons: list[str] = MessageField(
        description="Native mouse buttons held for this action."
    )
    delta_x: float = MessageField(description="Horizontal mouse delta received.")
    delta_y: float = MessageField(description="Vertical mouse delta received.")


class SceneChanged(ModelMessage):
    """Describe the scene selected for the next world-model reset."""

    source: str = MessageField(description="Scene source: upload or dataset.")
    scene: str = MessageField(description="Uploaded filename or dataset scene identifier.")


class DiamondState(InputState):
    """Hold the client-controlled generation state."""

    controller: str = InputField(
        default="human",
        choices=_CONTROLLERS,
        description="Use client actions or replay the spawn's recorded action trajectory.",
    )
    paused: bool = InputField(default=False, description="Pause world-model inference.")
    _pressed_keys: frozenset[str] = frozenset()
    _pressed_mouse_buttons: frozenset[str] = frozenset()
    _delta_x: float = 0.0
    _delta_y: float = 0.0
    _step_requested: bool = False
    _replay_step: int = 0


class Diamond(ReactorPipeline):
    """Run DIAMOND CSGO inference from Reactor's interactive state."""

    state: DiamondState
    output: DiamondOutput

    def __init__(self) -> None:
        super().__init__()
        self._agent: Any = None
        self._world: Any = None
        self._torch: Any = None
        self._action_type: Any = None
        self._encode_action: Callable[..., Any] | None = None
        self._key_codes: dict[str, int] = {}
        self._spawn_dirs: tuple[Path, ...] = ()
        self._rng = np.random.default_rng()
        self._sequence_length = 0
        self._full_resolution = (150, 280)
        self._low_resolution = (30, 56)
        self._pending_scene: _PreparedScene | None = None
        self._reset_requested = True

    def load(self, config_path: Path | None) -> None:
        """Load the upstream DIAMOND model and its CSGO spawn states.

        Args:
            config_path: Path to the adapter YAML named by ``reactor.yaml``.
        """
        config = _read_config(config_path)
        dependencies = _load_adapter_dependencies()
        torch = dependencies["torch"]
        snapshot_download = dependencies["snapshot_download"]
        compose = dependencies["compose"]
        initialize_config_dir = dependencies["initialize_config_dir"]
        instantiate = dependencies["instantiate"]
        omega_conf = dependencies["omega_conf"]
        upstream_root = _upstream_root()
        modules = _load_upstream_modules(upstream_root)
        agent_type = modules["agent"].Agent
        world_type = modules["world"].WorldModelEnv
        action_module = modules["action"]
        pygame = modules["pygame"]

        omega_conf.register_new_resolver("eval", _resolve_upstream_eval, replace=True)
        with initialize_config_dir(
            version_base="1.3",
            config_dir=str(upstream_root / "config"),
        ):
            cfg = compose(
                config_name="trainer",
                overrides=[f"world_model_env={config.profile}"],
            )

        snapshot = Path(
            snapshot_download(
                repo_id=config.repo_id,
                revision=config.revision,
                allow_patterns="csgo/*",
            )
        )
        cfg.agent = omega_conf.load(snapshot / "csgo/config/agent/csgo.yaml")
        cfg.env = omega_conf.load(snapshot / "csgo/config/env/csgo.yaml")

        device = _select_device(config.device, torch)
        torch.manual_seed(config.seed)
        self._agent = agent_type(instantiate(cfg.agent, num_actions=cfg.env.num_actions))
        self._agent = self._agent.to(device).eval()
        self._agent.load(snapshot / "csgo/model/csgo.pt")

        sequence_length = cfg.agent.denoiser.inner_model.num_steps_conditioning
        if self._agent.upsampler is not None:
            sequence_length = max(
                sequence_length,
                cfg.agent.upsampler.inner_model.num_steps_conditioning,
            )
        world_config = instantiate(cfg.world_model_env, num_batches_to_preload=1)
        spawn_root = snapshot / "csgo/spawn"
        self._world = world_type(
            self._agent.denoiser,
            self._agent.upsampler,
            self._agent.rew_end_model,
            spawn_root,
            1,
            sequence_length,
            world_config,
            return_denoising_trajectory=False,
        )

        self._action_type = action_module.CSGOAction
        self._encode_action = action_module.encode_csgo_action
        self._torch = torch
        self._spawn_dirs = tuple(sorted(path for path in spawn_root.iterdir() if path.is_dir()))
        self._rng = np.random.default_rng(config.seed)
        self._sequence_length = int(sequence_length)
        height, width = (int(value) for value in cfg.env.train.size)
        self._full_resolution = (height, width)
        upsampling_factor = int(cfg.agent.upsampler.upsampling_factor)
        self._low_resolution = (height // upsampling_factor, width // upsampling_factor)
        self._key_codes = {
            "w": pygame.K_w,
            "a": pygame.K_a,
            "s": pygame.K_s,
            "d": pygame.K_d,
            "space": pygame.K_SPACE,
            "ctrl": pygame.K_LCTRL,
            "shift": pygame.K_LSHIFT,
            "1": pygame.K_1,
            "2": pygame.K_2,
            "3": pygame.K_3,
            "r": pygame.K_r,
        }
        logger.info(
            "DIAMOND CSGO model ready",
            device=str(device),
            profile=config.profile,
            revision=config.revision,
        )

    @event(name="reset", description="Reset the world model to another CSGO spawn state.")
    def reset(self) -> ActionChanged:
        """Request a reset and return the released native input state."""
        self._pending_scene = None
        self._reset_requested = True
        self._clear_controls()
        self.state._step_requested = False
        return self._action_changed()

    @event(name="random_scene", description="Start from a random DIAMOND dataset scene.")
    def random_scene(self) -> SceneChanged:
        """Queue a random official spawn and return its identifier."""
        if not self._spawn_dirs:
            raise RuntimeError("DIAMOND spawn scenes were not loaded")
        scene_index = int(self._rng.integers(len(self._spawn_dirs)))
        scene_dir = self._spawn_dirs[scene_index]
        self._pending_scene = self._prepare_dataset_scene(scene_dir)
        self._queue_scene_reset()
        logger.info("dataset scene selected", scene=scene_dir.name)
        return SceneChanged(source="dataset", scene=scene_dir.name)

    @event(name="set_spawn_image", description="Start from an uploaded CSGO image.")
    def set_spawn_image(self, image: UploadedFile) -> SceneChanged:
        """Queue an uploaded image as a four-frame neutral initial condition.

        Args:
            image: Uploaded CSGO image fetched by Reactor Runtime.

        Raises:
            ValueError: If the upload is not a decodable image.
        """
        if not image.mime_type.startswith("image/"):
            raise ValueError(f"expected an image upload, got {image.mime_type!r}")
        full_res, low_res = _decode_spawn_image(
            image.data,
            full_resolution=self._full_resolution,
            low_resolution=self._low_resolution,
        )
        self._pending_scene = self._prepare_uploaded_scene(full_res, low_res)
        self.state.controller = "human"
        self._queue_scene_reset()
        logger.info("uploaded scene selected", name=image.name, size=len(image.data))
        return SceneChanged(source="upload", scene=image.name)

    @event(name="set_controller", description="Choose client control or recorded action replay.")
    def set_controller(
        self,
        controller: str = InputField(default="human", choices=_CONTROLLERS),
    ) -> ActionChanged:
        """Switch controller and return the resulting native input state."""
        if self.state.controller != controller:
            if (
                controller == "replay"
                and self._pending_scene is not None
                and self._pending_scene.next_act is None
            ):
                self._pending_scene = None
            self.state.controller = controller
            self.state._step_requested = False
            self._reset_requested = True
            self._clear_controls()
        return self._action_changed()

    @event(name="set_paused", description="Pause or resume world-model inference.")
    def set_paused(
        self,
        paused: bool = InputField(default=False, description="Pause world-model inference."),
    ) -> ActionChanged:
        """Pause or resume and return the released native input state."""
        self.state.paused = paused
        self.state._step_requested = False
        self._clear_controls()
        return self._action_changed()

    @event(name="step", description="Run one world-model step while paused.")
    def step(self) -> None:
        """Request one inference step without leaving paused mode."""
        if self.state is not None and self.state.paused:
            self.state._step_requested = True

    @event(name="set_key_state", description="Press or release a native CSGO input key.")
    def set_key_state(
        self,
        key: str = InputField(
            default="w",
            choices=_KEYS,
            description="Native DIAMOND key name.",
        ),
        pressed: bool = InputField(
            default=True,
            description="True for key down; false for key up.",
        ),
    ) -> ActionChanged:
        """Update one held keyboard key and return the resulting input state."""
        if self.state.controller == "human":
            if pressed:
                self.state._pressed_keys = self.state._pressed_keys.union((key,))
            else:
                self.state._pressed_keys = self.state._pressed_keys.difference((key,))
        return self._action_changed()

    @event(
        name="set_mouse_button_state",
        description="Press or release a native CSGO mouse button.",
    )
    def set_mouse_button_state(
        self,
        button: str = InputField(
            default="left",
            choices=_MOUSE_BUTTONS,
            description="Native mouse button: left fires and right scopes.",
        ),
        pressed: bool = InputField(
            default=True,
            description="True for button down; false for button up.",
        ),
    ) -> ActionChanged:
        """Update one held mouse button and return the resulting input state."""
        if self.state.controller == "human":
            if pressed:
                self.state._pressed_mouse_buttons = self.state._pressed_mouse_buttons.union(
                    (button,)
                )
            else:
                self.state._pressed_mouse_buttons = self.state._pressed_mouse_buttons.difference(
                    (button,)
                )
        return self._action_changed()

    @event(name="mouse_move", description="Apply raw mouse movement to the next model step.")
    def mouse_move(
        self,
        delta_x: float = InputField(
            default=0.0,
            ge=_DELTA_X_MIN,
            le=_DELTA_X_MAX,
            description="Horizontal mouse delta in DIAMOND's native range.",
        ),
        delta_y: float = InputField(
            default=0.0,
            ge=_DELTA_Y_MIN,
            le=_DELTA_Y_MAX,
            description="Vertical mouse delta in DIAMOND's native range.",
        ),
    ) -> ActionChanged:
        """Store one raw mouse delta and return the resulting input state."""
        if self.state.controller == "human":
            self.state._delta_x = delta_x
            self.state._delta_y = delta_y
            return self._action_changed(delta_x=delta_x, delta_y=delta_y)
        return self._action_changed()

    def inference(self) -> Iterator[DiamondOutput | None]:
        """Generate CSGO frames while applying the latest client controls."""
        if self._world is None or self._agent is None or self._encode_action is None:
            raise RuntimeError("DIAMOND model was not loaded")

        self._reset_requested = True
        self._clear_controls()
        while True:
            if self._reset_requested:
                self._world.reset()
                self._apply_pending_scene()
                self.state._replay_step = 0
                self._reset_requested = False

            if self.state.paused and not self.state._step_requested:
                yield None
                continue
            self.state._step_requested = False

            action = self._next_action()
            observation, _reward, ended, truncated, _info = self._world.step(action)
            if bool(ended.item()) or bool(truncated.item()) or self._replay_trajectory_finished():
                self._reset_requested = True
                self._clear_controls()
            yield DiamondOutput(main_video=_to_video_frame(observation))

    def _prepare_uploaded_scene(
        self,
        full_res: np.ndarray,
        low_res: np.ndarray,
    ) -> _PreparedScene:
        """Build a device-ready four-frame condition from one uploaded image."""
        if self._encode_action is None or self._agent is None:
            raise RuntimeError("DIAMOND model was not loaded")
        full_frames = np.repeat(full_res[None], self._sequence_length, axis=0)
        low_frames = np.repeat(low_res[None], self._sequence_length, axis=0)
        neutral = self._encode_action(
            self._action_type([], 0.0, 0.0, False, False),
            device=self._agent.device,
        )
        actions = neutral.reshape(1, 1, -1).repeat(1, self._sequence_length, 1)
        return _PreparedScene(
            obs=self._observation_tensor(low_frames),
            obs_full_res=self._observation_tensor(full_frames),
            act=actions,
            next_act=None,
        )

    def _prepare_dataset_scene(self, scene_dir: Path) -> _PreparedScene:
        """Load one official spawn with its full recorded action trajectory."""
        if self._torch is None or self._agent is None:
            raise RuntimeError("DIAMOND model was not loaded")
        device = self._agent.device
        return _PreparedScene(
            obs=self._observation_tensor(np.load(scene_dir / "low_res.npy")),
            obs_full_res=self._observation_tensor(np.load(scene_dir / "full_res.npy")),
            act=self._torch.tensor(
                np.load(scene_dir / "act.npy"),
                dtype=self._torch.long,
                device=device,
            ).unsqueeze(0),
            next_act=self._torch.tensor(
                np.load(scene_dir / "next_act.npy"),
                dtype=self._torch.long,
                device=device,
            ),
        )

    def _observation_tensor(self, frames: np.ndarray) -> Any:
        """Normalize uint8 TCHW frames into a batched tensor on the model device."""
        if self._torch is None or self._agent is None:
            raise RuntimeError("DIAMOND model was not loaded")
        return (
            self._torch.tensor(frames, device=self._agent.device)
            .div(255)
            .mul(2)
            .sub(1)
            .unsqueeze(0)
        )

    def _queue_scene_reset(self) -> None:
        """Reset controls and request application of the queued scene."""
        self._reset_requested = True
        self.state._step_requested = False
        self.state._replay_step = 0
        self._clear_controls()

    def _apply_pending_scene(self) -> None:
        """Replace the freshly reset upstream buffers with one queued scene."""
        scene = self._pending_scene
        if scene is None:
            return
        self._world.obs_buffer = scene.obs
        self._world.obs_full_res_buffer = scene.obs_full_res
        self._world.act_buffer = scene.act
        if scene.next_act is not None:
            self._world.next_act = scene.next_act
        self._pending_scene = None

    def _next_action(self) -> Any:
        """Return the next client action or recorded replay action."""
        if self.state.controller == "replay":
            self._clear_controls()
            replay_step = self.state._replay_step
            if replay_step == 0:
                action = self._world.act_buffer[0, -1].clone()
            else:
                action = self._world.next_act[replay_step - 1].clone()
            self.state._replay_step += 1
            return action

        assert self._encode_action is not None
        keys = [self._key_codes[key] for key in _KEYS if key in self.state._pressed_keys]
        delta_x = self.state._delta_x
        delta_y = self.state._delta_y
        self.state._delta_x = 0.0
        self.state._delta_y = 0.0
        action = self._action_type(
            keys,
            delta_x,
            delta_y,
            "left" in self.state._pressed_mouse_buttons,
            "right" in self.state._pressed_mouse_buttons,
        )
        return self._encode_action(action, device=self._agent.device)

    def _action_changed(self, *, delta_x: float = 0.0, delta_y: float = 0.0) -> ActionChanged:
        """Describe the current native input state for an event response."""
        return ActionChanged(
            controller=self.state.controller,
            pressed_keys=[key for key in _KEYS if key in self.state._pressed_keys],
            pressed_mouse_buttons=[
                button for button in _MOUSE_BUTTONS if button in self.state._pressed_mouse_buttons
            ],
            delta_x=delta_x,
            delta_y=delta_y,
        )

    def _replay_trajectory_finished(self) -> bool:
        """Return whether the recorded spawn actions were all consumed."""
        return self.state.controller == "replay" and self.state._replay_step > int(
            self._world.next_act.size(0)
        )

    def _clear_controls(self) -> None:
        """Release held controls and discard pending mouse movement."""
        if self.state is None:
            return
        self.state._pressed_keys = frozenset()
        self.state._pressed_mouse_buttons = frozenset()
        self.state._delta_x = 0.0
        self.state._delta_y = 0.0


def _decode_spawn_image(
    data: bytes,
    *,
    full_resolution: tuple[int, int],
    low_resolution: tuple[int, int],
) -> tuple[np.ndarray, np.ndarray]:
    """Decode and center-crop one upload into DIAMOND's CHW frame sizes.

    Args:
        data: Encoded image bytes.
        full_resolution: Target ``(height, width)`` for the upsampler.
        low_resolution: Target ``(height, width)`` for the world model.

    Returns:
        Full-resolution and low-resolution contiguous uint8 RGB arrays.

    Raises:
        ValueError: If Pillow cannot decode the upload.
    """
    image_module = importlib.import_module("PIL.Image")
    image_ops = importlib.import_module("PIL.ImageOps")
    full_height, full_width = full_resolution
    low_height, low_width = low_resolution
    try:
        with image_module.open(BytesIO(data)) as uploaded:
            rgb = image_ops.exif_transpose(uploaded).convert("RGB")
            fitted = image_ops.fit(
                rgb,
                (full_width, full_height),
                method=image_module.Resampling.LANCZOS,
            )
            low = fitted.resize(
                (low_width, low_height),
                resample=image_module.Resampling.LANCZOS,
            )
            full_array = np.asarray(fitted, dtype=np.uint8).transpose(2, 0, 1)
            low_array = np.asarray(low, dtype=np.uint8).transpose(2, 0, 1)
    except (OSError, ValueError) as error:
        raise ValueError("could not decode uploaded spawn image") from error
    return np.ascontiguousarray(full_array), np.ascontiguousarray(low_array)


def _read_config(config_path: Path | None) -> _Config:
    """Read and validate the adapter's model configuration."""
    if config_path is None:
        raise ValueError("DIAMOND requires runtime.config in reactor.yaml")
    document = yaml.safe_load(config_path.read_text())
    if not isinstance(document, dict):
        raise ValueError(f"{config_path}: expected a YAML mapping")

    profile = str(document.get("profile", "fast"))
    if profile not in {"fast", "higher_quality"}:
        raise ValueError("profile must be 'fast' or 'higher_quality'")
    device = str(document.get("device", "auto"))
    if device not in {"auto", "cpu", "mps", "cuda"}:
        raise ValueError("device must be auto, cpu, mps, or cuda")
    return _Config(
        repo_id=str(document.get("repo_id", "eloialonso/diamond")),
        revision=str(document["revision"]),
        device=device,
        profile=profile,
        seed=int(document.get("seed", 0)),
    )


def _upstream_root() -> Path:
    """Return the external DIAMOND checkout configured for this process.

    Raises:
        RuntimeError: If ``DIAMOND_PATH`` is unset or does not identify a CSGO checkout.
    """
    configured = os.environ.get(_UPSTREAM_ENV)
    if not configured:
        raise RuntimeError(
            f"Set {_UPSTREAM_ENV} to the DIAMOND repository checkout before starting Reactor"
        )

    root = Path(configured).expanduser().resolve()
    required = (
        root / "config/trainer.yaml",
        root / "src/agent.py",
        root / "src/csgo/action_processing.py",
    )
    missing = [str(path.relative_to(root)) for path in required if not path.is_file()]
    if missing:
        joined = ", ".join(missing)
        raise RuntimeError(
            f"{_UPSTREAM_ENV}={root} is not a DIAMOND CSGO checkout; missing: {joined}"
        )
    return root


def _load_upstream_modules(upstream_root: Path) -> dict[str, Any]:
    """Import DIAMOND from an external, unmodified ``src`` tree."""
    source = str(upstream_root / "src")
    if source not in sys.path:
        sys.path.insert(0, source)
    return {
        "agent": importlib.import_module("agent"),
        "world": importlib.import_module("envs"),
        "action": importlib.import_module("csgo.action_processing"),
        "pygame": importlib.import_module("pygame"),
    }


def _load_adapter_dependencies() -> dict[str, Any]:
    """Import DIAMOND's optional runtime dependencies only when loading weights."""
    hydra = importlib.import_module("hydra")
    return {
        "torch": importlib.import_module("torch"),
        "snapshot_download": importlib.import_module("huggingface_hub").snapshot_download,
        "compose": hydra.compose,
        "initialize_config_dir": hydra.initialize_config_dir,
        "instantiate": importlib.import_module("hydra.utils").instantiate,
        "omega_conf": importlib.import_module("omegaconf").OmegaConf,
    }


def _select_device(requested: str, torch_module: Any) -> Any:
    """Return the requested accelerator, preferring MPS on Apple Silicon."""
    if requested == "auto":
        if torch_module.cuda.is_available():
            requested = "cuda"
        elif torch_module.backends.mps.is_available():
            requested = "mps"
        else:
            requested = "cpu"
    if requested == "cuda" and not torch_module.cuda.is_available():
        raise RuntimeError("CUDA was requested but is unavailable")
    if requested == "mps" and not torch_module.backends.mps.is_available():
        raise RuntimeError("MPS was requested but is unavailable")
    return torch_module.device(requested)


def _resolve_upstream_eval(expression: str) -> float:
    """Resolve the one trusted expression used by DIAMOND's sampler config."""
    if expression != 'float("inf")':
        raise ValueError(f"unsupported DIAMOND config expression: {expression!r}")
    return math.inf


def _to_video_frame(observation: Any) -> np.ndarray:
    """Convert one DIAMOND NCHW observation into contiguous uint8 RGB."""
    frame = (
        observation[0].detach().clamp(-1, 1).add(1).mul(127.5).byte().permute(1, 2, 0).cpu().numpy()
    )
    return np.ascontiguousarray(frame)
