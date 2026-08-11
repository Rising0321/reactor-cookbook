"""Serve OpenDreamer as an interactive Minecraft world model.

The adapter loads the public OpenDreamer tokenizer and EMA dynamics checkpoint,
seeds their KV caches from consecutive Minecraft frames with aligned VPT
actions, and turns Reactor input events into the action representation used
during training. It emits one RGB frame for every autoregressive model step.
"""

from __future__ import annotations

import importlib
import io
import json
import os
import re
import subprocess
import sys
from collections.abc import Callable, Iterator, Mapping
from contextlib import AbstractContextManager
from dataclasses import dataclass
from pathlib import Path
from typing import Any

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
    connected,
    disconnected,
    event,
)
from reactor_runtime.interface.pipeline.idle import _IdleType
from reactor_runtime.log import get_logger

logger = get_logger(__name__)

_KEYS = [
    "w",
    "a",
    "s",
    "d",
    "space",
    "shift",
    "ctrl",
    "e",
    "q",
    "escape",
    "f",
    "1",
    "2",
    "3",
    "4",
    "5",
    "6",
    "7",
    "8",
    "9",
    "f3",
]
_MOUSE_BUTTONS = ["left", "right", "middle"]
_KEY_TO_VPT_NAME = {
    "w": "key.keyboard.w",
    "a": "key.keyboard.a",
    "s": "key.keyboard.s",
    "d": "key.keyboard.d",
    "space": "key.keyboard.space",
    "shift": "key.keyboard.left.shift",
    "ctrl": "key.keyboard.left.control",
    "e": "key.keyboard.e",
    "q": "key.keyboard.q",
    "escape": "key.keyboard.escape",
    "f": "key.keyboard.f",
    "1": "key.keyboard.1",
    "2": "key.keyboard.2",
    "3": "key.keyboard.3",
    "4": "key.keyboard.4",
    "5": "key.keyboard.5",
    "6": "key.keyboard.6",
    "7": "key.keyboard.7",
    "8": "key.keyboard.8",
    "9": "key.keyboard.9",
    "f3": "key.keyboard.f3",
}
_BUTTON_TO_VPT_NAME = {
    "left": "mouse.0",
    "right": "mouse.1",
    "middle": "mouse.2",
}
_CAMERA_DELTA_MIN = -200.0
_CAMERA_DELTA_MAX = 200.0
_DEMO_CHOICES = ["demo_1", "demo_2", "demo_3"]
_UPSTREAM_ENV = "OPENDREAMER_PATH"
_REVISION_PATTERN = re.compile(r"[0-9a-f]{40}")


@dataclass(frozen=True)
class _DemoConfig:
    """Describe one configured VPT conditioning window."""

    name: str
    video: Path
    actions: Path
    start_frame: int


@dataclass(frozen=True)
class _Config:
    """Hold validated OpenDreamer load settings."""

    source_revision: str
    checkpoint_repo_id: str
    checkpoint_revision: str
    platform: str
    seed: int
    num_steps: int
    tau_ctx_target: float
    conditioning_frames: int
    demos: tuple[_DemoConfig, ...]
    warmup_steps: int
    memory_fraction: float


@dataclass(frozen=True)
class _ConditioningSequence:
    """Pair consecutive Minecraft frames with aligned VPT actions."""

    frames: np.ndarray
    actions: Any


class OpenDreamerOutput(Output):
    """Carry one generated Minecraft frame."""

    main_video: Video


class ActionChanged(ModelMessage):
    """Describe the latest native control input received from a client."""

    control: str = MessageField(description="Command that produced this input state.")
    paused: bool = MessageField(description="Whether world-model inference is paused.")
    pressed_keys: list[str] = MessageField(description="Native VPT keys currently held.")
    pressed_mouse_buttons: list[str] = MessageField(
        description="Native VPT mouse buttons currently held."
    )
    delta_x: float = MessageField(description="Horizontal mouse delta received.")
    delta_y: float = MessageField(description="Vertical mouse delta received.")
    wheel_delta: int = MessageField(description="Mouse-wheel delta received.")


class ConditioningChanged(ModelMessage):
    """Describe the conditioning source selected for the next rollout."""

    source: str = MessageField(description="Conditioning source: demo or upload.")
    selection: str = MessageField(description="Demo identifier or uploaded file pair.")


class RolloutReset(ModelMessage):
    """Describe a requested cache reset before it reaches inference."""

    seed: int = MessageField(description="Generation seed selected for the rollout.")
    conditioning: str = MessageField(description="Conditioning source retained by the reset.")


class OpenDreamerState(InputState):
    """Hold the controls for one OpenDreamer session."""

    paused: bool = InputField(default=False, description="Pause world-model inference.")
    _pressed_keys: frozenset[str] = frozenset()
    _pressed_mouse_buttons: frozenset[str] = frozenset()
    _delta_x: float = 0.0
    _delta_y: float = 0.0
    _wheel_delta: int = 0
    _reset_requested: bool = True
    _seed: int = 0


class OpenDreamer(ReactorPipeline):
    """Generate an interactive Minecraft world with OpenDreamer."""

    state: OpenDreamerState
    output: OpenDreamerOutput

    def __init__(self) -> None:
        super().__init__()
        self._config: _Config | None = None
        self._deps: dict[str, Any] = {}
        self._mesh: Any = None
        self._tokenizer: Any = None
        self._dynamics: Any = None
        self._schedule: Any = None
        self._latent_shape: tuple[int, int, int, int] | None = None
        self._model_frame_shape: tuple[int, int, int] | None = None
        self._empty_dynamics_cache: Any = None
        self._empty_tokenizer_cache: Any = None
        self._next_frame_jit: Callable[..., Any] | None = None
        self._observe_frame_jit: Callable[..., Any] | None = None
        self._key_to_index: dict[str, int] = {}
        self._demos: dict[str, _ConditioningSequence] = {}
        self._conditioning_source = "random"
        self._uploaded_conditioning: _ConditioningSequence | None = None
        self._demo_rng = np.random.default_rng()

    def load(self, config_path: Path | None) -> None:
        """Load the public OpenDreamer source and checkpoint once.

        Args:
            config_path: Path to the model YAML named by ``reactor.yaml``.
        """
        config = _read_config(config_path)
        upstream_root = _upstream_root()
        _verify_source_revision(upstream_root, config.source_revision)
        _ensure_demo_assets(upstream_root, config.demos)
        _prepare_process_environment(config)
        dependencies = _load_dependencies(upstream_root)
        self._config = config
        self._deps = dependencies

        jax = dependencies["jax"]
        nnx = dependencies["nnx"]
        snapshot_download = dependencies["snapshot_download"]
        bundle_type = dependencies["bundle_type"]
        build_parallel = dependencies["build_parallel"]

        checkpoint_path = snapshot_download(
            repo_id=config.checkpoint_repo_id,
            revision=config.checkpoint_revision,
        )
        if jax.default_backend() == "cpu":
            raise RuntimeError("OpenDreamer requires a CUDA accelerator")

        mesh, _data_sharding, mesh_rules = build_parallel("data")
        self._mesh = mesh
        with _mesh_context(jax, mesh):
            bundle = bundle_type.from_pretrained(
                checkpoint_path,
                mesh_rules=mesh_rules,
                rngs=nnx.Rngs(config.seed),
                model_names={"dynamics_ema", "tokenizer"},
            )
            if bundle.dynamics_ema is None or bundle.tokenizer is None:
                raise RuntimeError("checkpoint does not contain dynamics_ema and tokenizer")
            self._dynamics = bundle.dynamics_ema
            self._tokenizer = bundle.tokenizer
            self._configure_inference(config)
            self._warm_inference(config)
            self._validate_action_space()

        assert self._model_frame_shape is not None
        self._demos = {
            demo.name: _read_conditioning_sequence(
                _upstream_asset(upstream_root, demo.video),
                _upstream_asset(upstream_root, demo.actions),
                self._model_frame_shape,
                start_frame=demo.start_frame,
                required_frames=config.conditioning_frames,
                dependencies=self._deps,
            )
            for demo in config.demos
        }
        logger.info(
            "OpenDreamer model ready",
            backend=jax.default_backend(),
            devices=len(jax.devices()),
            checkpoint_revision=config.checkpoint_revision,
            source_revision=config.source_revision,
            demos=len(self._demos),
            conditioning_frames=config.conditioning_frames,
        )

    def _configure_inference(self, config: _Config) -> None:
        """Create schedules, empty caches, and compiled inference callables."""
        jnp = self._deps["jnp"]
        nnx = self._deps["nnx"]
        schedule_type = self._deps["schedule_type"]
        next_frame = self._deps["next_frame"]
        tokenizer_caches_type = self._deps["tokenizer_caches_type"]
        normalize_latents = self._deps["normalize_latents"]

        dynamics_config = self._dynamics.cfg
        tokenizer_config = self._tokenizer.cfg
        self._schedule = schedule_type.init(
            num_steps=config.num_steps,
            k_max=dynamics_config.k_max,
            tau_ctx_target=config.tau_ctx_target,
        )

        n_latents = int(tokenizer_config.decoder.n_latents)
        d_bottleneck = int(tokenizer_config.encoder.d_bottleneck)
        height = int(tokenizer_config.decoder.H)
        width = int(tokenizer_config.decoder.W)
        self._latent_shape = (1, 1, n_latents, d_bottleneck)
        self._model_frame_shape = (height, width, 3)

        self._empty_dynamics_cache = self._dynamics.create_static_caches(
            batch_size=1,
            n_latents=n_latents,
            window_size=int(dynamics_config.context_length),
            n_agent=0,
            dtype=dynamics_config.dtype,
        )
        self._empty_tokenizer_cache = self._tokenizer.create_static_caches(
            batch_size=1,
            H=height,
            W=width,
            window_size=int(tokenizer_config.decoder.context_length),
            dtype=tokenizer_config.decoder.dtype,
        )
        schedule = self._schedule

        def compiled_next_frame(
            tokenizer: Any,
            dynamics: Any,
            action: Any,
            latent_shape: tuple[int, int, int, int],
            dynamics_cache: Any,
            tokenizer_cache: Any,
            rng: Any,
        ) -> tuple[Any, Any, Any, Any]:
            frame, _hidden, new_dynamics_cache, decoder_cache, new_rng = next_frame(
                tokenizer,
                dynamics,
                schedule,
                action,
                latent_shape,
                dynamics_cache,
                tokenizer_cache.decoder,
                rng,
            )
            new_tokenizer_cache = tokenizer_caches_type(
                encoder=tokenizer_cache.encoder,
                decoder=decoder_cache,
            )
            return frame, new_dynamics_cache, new_tokenizer_cache, new_rng

        def compiled_observe_frame(
            tokenizer: Any,
            dynamics: Any,
            frame: Any,
            action: Any,
            dynamics_cache: Any,
            tokenizer_cache: Any,
        ) -> tuple[Any, Any]:
            video = jnp.asarray(frame, dtype=jnp.float32)[None, None, ...]
            latent, _, encoder_cache = tokenizer.encode(
                video,
                deterministic=True,
                caches=tokenizer_cache.encoder,
            )
            normalized = normalize_latents(
                latent,
                dynamics.cfg.latent_mean,
                dynamics.cfg.latent_std,
            )
            action_with_time = action[:, None, ...]
            step_indices = jnp.full((1, 1), schedule.emax, dtype=jnp.int32)
            tau_indices = jnp.full((1, 1), schedule.k_max, dtype=jnp.int32)
            _, (_, new_dynamics_cache) = dynamics(
                action_with_time,
                step_indices,
                tau_indices,
                normalized,
                deterministic=True,
                caches=dynamics_cache,
            )
            _, decoder_cache = tokenizer.decode(
                latent,
                caches=tokenizer_cache.decoder,
                deterministic=True,
            )
            new_tokenizer_cache = tokenizer_caches_type(
                encoder=encoder_cache,
                decoder=decoder_cache,
            )
            return new_dynamics_cache, new_tokenizer_cache

        self._next_frame_jit = nnx.jit(
            compiled_next_frame,
            static_argnames=("latent_shape",),
        )
        self._observe_frame_jit = nnx.jit(compiled_observe_frame)

    def _warm_inference(self, config: _Config) -> None:
        """Compile the generation and conditioning paths before serving."""
        if config.warmup_steps == 0:
            return
        assert self._latent_shape is not None
        assert self._model_frame_shape is not None
        assert self._next_frame_jit is not None
        assert self._observe_frame_jit is not None
        jax = self._deps["jax"]
        jnp = self._deps["jnp"]
        rng = jax.random.PRNGKey(config.seed)
        dynamics_cache = self._empty_dynamics_cache
        tokenizer_cache = self._empty_tokenizer_cache
        noop = self._noop_action()
        for _ in range(config.warmup_steps):
            rng, step_rng = jax.random.split(rng)
            frame, dynamics_cache, tokenizer_cache, rng = self._next_frame_jit(
                self._tokenizer,
                self._dynamics,
                noop,
                self._latent_shape,
                dynamics_cache,
                tokenizer_cache,
                step_rng,
            )
            jax.block_until_ready((frame, dynamics_cache, tokenizer_cache, rng))
        zero_frame = jnp.zeros(self._model_frame_shape, dtype=jnp.uint8)
        observed = self._observe_frame_jit(
            self._tokenizer,
            self._dynamics,
            zero_frame,
            noop,
            self._empty_dynamics_cache,
            self._empty_tokenizer_cache,
        )
        jax.block_until_ready(observed)

    @connected
    def on_connect(self) -> None:
        """Start each connection from a randomly selected configured demo."""
        if self.state is None or self._config is None:
            return
        self.state._seed = self._config.seed
        self.state._reset_requested = True
        self._conditioning_source = self._random_demo_name()
        self._clear_controls()

    @disconnected
    def on_disconnect(self) -> None:
        """Release every held and transient control when a client leaves."""
        self._clear_controls()

    @event(name="set_key_state", description="Press or release one native VPT key.")
    def set_key_state(
        self,
        key: str = InputField(default="w", choices=_KEYS, description="Native VPT key name."),
        pressed: bool = InputField(default=True, description="True for key down."),
    ) -> ActionChanged:
        """Update one held keyboard key and return the resulting input state."""
        if not self.state.paused:
            if pressed:
                self.state._pressed_keys = self.state._pressed_keys.union((key,))
            else:
                self.state._pressed_keys = self.state._pressed_keys.difference((key,))
        return self._action_changed(control="set_key_state")

    @event(
        name="set_mouse_button_state",
        description="Press or release one native VPT mouse button.",
    )
    def set_mouse_button_state(
        self,
        button: str = InputField(
            default="left",
            choices=_MOUSE_BUTTONS,
            description="Native VPT mouse button.",
        ),
        pressed: bool = InputField(default=True, description="True for button down."),
    ) -> ActionChanged:
        """Update one held mouse button and return the resulting input state."""
        if not self.state.paused:
            if pressed:
                self.state._pressed_mouse_buttons = self.state._pressed_mouse_buttons.union(
                    (button,)
                )
            else:
                self.state._pressed_mouse_buttons = self.state._pressed_mouse_buttons.difference(
                    (button,)
                )
        return self._action_changed(control="set_mouse_button_state")

    @event(name="mouse_move", description="Accumulate raw VPT mouse movement for the next step.")
    def mouse_move(
        self,
        delta_x: float = InputField(
            default=0.0,
            ge=_CAMERA_DELTA_MIN,
            le=_CAMERA_DELTA_MAX,
            description="Horizontal raw mouse delta.",
        ),
        delta_y: float = InputField(
            default=0.0,
            ge=_CAMERA_DELTA_MIN,
            le=_CAMERA_DELTA_MAX,
            description="Vertical raw mouse delta.",
        ),
    ) -> ActionChanged:
        """Accumulate camera motion and return the received native input."""
        if not self.state.paused:
            self.state._delta_x = float(
                np.clip(self.state._delta_x + delta_x, _CAMERA_DELTA_MIN, _CAMERA_DELTA_MAX)
            )
            self.state._delta_y = float(
                np.clip(self.state._delta_y + delta_y, _CAMERA_DELTA_MIN, _CAMERA_DELTA_MAX)
            )
            return self._action_changed(
                control="mouse_move",
                delta_x=delta_x,
                delta_y=delta_y,
            )
        return self._action_changed(control="mouse_move")

    @event(name="mouse_wheel", description="Apply a native VPT scroll tick to the next step.")
    def mouse_wheel(
        self,
        delta: int = InputField(
            default=0,
            ge=-1,
            le=1,
            description="-1 scrolls down, 1 scrolls up, and 0 is neutral.",
        ),
    ) -> ActionChanged:
        """Accumulate a scroll tick and return the received native input."""
        if not self.state.paused:
            self.state._wheel_delta += delta
            return self._action_changed(control="mouse_wheel", wheel_delta=delta)
        return self._action_changed(control="mouse_wheel")

    @event(name="set_paused", description="Pause or resume world-model inference.")
    def set_paused(
        self,
        paused: bool = InputField(default=False, description="Pause world-model inference."),
    ) -> ActionChanged:
        """Set pause state and return the released native input state."""
        self.state.paused = paused
        self._clear_controls()
        return self._action_changed(control="set_paused")

    @event(name="reset", description="Reset caches from the active conditioning sequence.")
    def reset(
        self,
        seed: int = InputField(
            default=-1,
            ge=-1,
            le=2_147_483_647,
            description="New RNG seed; -1 reuses the current seed.",
        ),
    ) -> RolloutReset:
        """Request a reproducible reset and return the retained rollout state."""
        if seed >= 0:
            self.state._seed = seed
        self.state._reset_requested = True
        self._clear_controls()
        return RolloutReset(
            seed=self.state._seed,
            conditioning=self._conditioning_source,
        )

    @event(name="set_demo", description="Reset from a configured VPT demo window.")
    def set_demo(
        self,
        demo: str = InputField(
            default="demo_1",
            choices=_DEMO_CHOICES,
            description="Configured demo window.",
        ),
    ) -> ConditioningChanged:
        """Select a configured demo and return its identifier."""
        if demo not in self._demos:
            raise CommandError("demo_unavailable", f"{demo} is not configured.")
        self._conditioning_source = demo
        if self.state is not None:
            self.state._reset_requested = True
            self._clear_controls()
        return ConditioningChanged(source="demo", selection=demo)

    @event(name="random_demo", description="Reset from a randomly selected VPT demo window.")
    def random_demo(self) -> ConditioningChanged:
        """Select a random configured demo and return its identifier."""
        demo = self._random_demo_name()
        self._conditioning_source = demo
        if self.state is not None:
            self.state._reset_requested = True
            self._clear_controls()
        logger.info("selected random conditioning demo", demo=demo)
        return ConditioningChanged(source="demo", selection=demo)

    @event(
        name="set_conditioning_image",
        description="Reset from an uploaded Minecraft image with neutral action history.",
    )
    def set_conditioning_image(self, image: UploadedFile) -> ConditioningChanged:
        """Build a static neutral conditioning sequence from one uploaded image."""
        if self._model_frame_shape is None or self._config is None:
            raise CommandError("model_not_ready", "OpenDreamer is still loading.")
        if not image.mime_type.startswith("image/"):
            raise CommandError("unsupported_media", f"{image.name} must be an image.")
        try:
            frame = _decode_conditioning_image(image.data, self._model_frame_shape)
        except (ValueError, OSError) as error:
            raise CommandError("invalid_image", str(error)) from error
        self._uploaded_conditioning = _ConditioningSequence(
            frames=np.repeat(
                frame[None],
                self._config.conditioning_frames,
                axis=0,
            ).copy(),
            actions=self._repeated_noop_actions(self._config.conditioning_frames),
        )
        self._conditioning_source = "uploaded"
        if self.state is not None:
            self.state._reset_requested = True
            self._clear_controls()
        return ConditioningChanged(source="upload", selection=image.name)

    def inference(self) -> Iterator[OpenDreamerOutput | _IdleType]:
        """Generate frames from the latest held and transient controls."""
        if self._config is None or self._next_frame_jit is None or self._observe_frame_jit is None:
            raise RuntimeError("OpenDreamer was not loaded")
        assert self._latent_shape is not None
        jax = self._deps["jax"]
        jnp = self._deps["jnp"]

        rng = jax.random.PRNGKey(self.state._seed)
        dynamics_cache = self._empty_dynamics_cache
        tokenizer_cache = self._empty_tokenizer_cache
        conditioning: _ConditioningSequence | None = None
        observation_index = 0
        self.state._reset_requested = True

        with _mesh_context(jax, self._mesh):
            while True:
                if self.state._reset_requested:
                    self.state._reset_requested = False
                    rng = jax.random.PRNGKey(self.state._seed)
                    dynamics_cache = self._empty_dynamics_cache
                    tokenizer_cache = self._empty_tokenizer_cache
                    conditioning = self._select_conditioning()
                    observation_index = 0

                if conditioning is None:
                    yield Idle
                    continue

                if observation_index < conditioning.frames.shape[0]:
                    dynamics_cache, tokenizer_cache = self._observe_frame_jit(
                        self._tokenizer,
                        self._dynamics,
                        jnp.asarray(conditioning.frames[observation_index]),
                        self._action_at(conditioning.actions, observation_index),
                        dynamics_cache,
                        tokenizer_cache,
                    )
                    jax.block_until_ready((dynamics_cache, tokenizer_cache))
                    observation_index += 1
                    yield Idle
                    continue

                if self.state.paused:
                    yield Idle
                    continue

                action = self._build_action()
                rng, step_rng = jax.random.split(rng)
                frame, dynamics_cache, tokenizer_cache, rng = self._next_frame_jit(
                    self._tokenizer,
                    self._dynamics,
                    action,
                    self._latent_shape,
                    dynamics_cache,
                    tokenizer_cache,
                    step_rng,
                )
                jax.block_until_ready(frame)
                self._consume_transient_controls()
                output = np.asarray(frame[0, 0])
                if output.dtype != np.uint8:
                    output = np.clip(output, 0, 255).astype(np.uint8)
                yield OpenDreamerOutput(main_video=np.ascontiguousarray(output))

    def _select_conditioning(self) -> _ConditioningSequence | None:
        """Return the uploaded sequence or resolve the active configured demo."""
        if self._conditioning_source == "uploaded":
            return self._uploaded_conditioning
        if not self._demos:
            return None
        name = self._conditioning_source
        if name == "random":
            name = self._random_demo_name()
            self._conditioning_source = name
            logger.info("selected random conditioning demo", demo=name)
        return self._demos.get(name)

    def _random_demo_name(self) -> str:
        """Return one configured demo name from the session RNG."""
        if not self._demos:
            raise CommandError("demo_unavailable", "No conditioning demos are configured.")
        names = tuple(self._demos)
        return names[int(self._demo_rng.integers(len(names)))]

    def _action_changed(
        self,
        *,
        control: str,
        delta_x: float = 0.0,
        delta_y: float = 0.0,
        wheel_delta: int = 0,
    ) -> ActionChanged:
        """Describe the current native input state for an event response."""
        return ActionChanged(
            control=control,
            paused=self.state.paused,
            pressed_keys=[key for key in _KEYS if key in self.state._pressed_keys],
            pressed_mouse_buttons=[
                button for button in _MOUSE_BUTTONS if button in self.state._pressed_mouse_buttons
            ],
            delta_x=delta_x,
            delta_y=delta_y,
            wheel_delta=wheel_delta,
        )

    def _action_at(self, actions: Any, index: int) -> Any:
        """Remove the time dimension from one batched conditioning action."""
        action_type = self._deps["action_type"]

        def take(value: Any) -> Any:
            return None if value is None else value[:, index]

        return action_type(
            binary=take(actions.binary),
            categorical=take(actions.categorical),
            continuous=take(actions.continuous),
        )

    def _build_action(self) -> Any:
        """Build one upstream ``Actions`` value from the current Reactor state."""
        jnp = self._deps["jnp"]
        action_type = self._deps["action_type"]
        mouse_to_categorical = self._deps["mouse_to_categorical"]
        binary = np.zeros((1, len(self._key_to_index)), dtype=np.int32)
        for key in self.state._pressed_keys:
            binary[0, self._key_to_index[_KEY_TO_VPT_NAME[key]]] = 1
        for button in self.state._pressed_mouse_buttons:
            binary[0, self._key_to_index[_BUTTON_TO_VPT_NAME[button]]] = 1
        if self.state._wheel_delta < 0:
            binary[0, self._key_to_index["mouse.wheel_neg"]] = 1
        elif self.state._wheel_delta > 0:
            binary[0, self._key_to_index["mouse.wheel_pos"]] = 1
        categorical = mouse_to_categorical(
            np.asarray([self.state._delta_x], dtype=np.float32),
            np.asarray([self.state._delta_y], dtype=np.float32),
        )
        return action_type(
            binary=jnp.asarray(binary, dtype=jnp.int32),
            categorical=jnp.asarray(categorical, dtype=jnp.int32),
            continuous=None,
        )

    def _noop_action(self) -> Any:
        """Return one neutral upstream ``Actions`` value."""
        jnp = self._deps["jnp"]
        action_type = self._deps["action_type"]
        camera_classes = int(self._deps["camera_classes"])
        return action_type(
            binary=jnp.zeros((1, len(self._key_to_index) or 27), dtype=jnp.int32),
            categorical=jnp.full((1,), camera_classes // 2, dtype=jnp.int32),
            continuous=None,
        )

    def _repeated_noop_actions(self, frames: int) -> Any:
        """Return a batched neutral action history for static image conditioning."""
        jnp = self._deps["jnp"]
        action_type = self._deps["action_type"]
        noop = self._noop_action()

        def repeat(value: Any) -> Any:
            return None if value is None else jnp.repeat(value[:, None, ...], frames, axis=1)

        return action_type(
            binary=repeat(noop.binary),
            categorical=repeat(noop.categorical),
            continuous=repeat(noop.continuous),
        )

    def _validate_action_space(self) -> None:
        """Verify the loaded source and checkpoint use the expected VPT action space."""
        source_mapping = dict(self._deps["key_to_index"])
        if len(source_mapping) != int(self._deps["binary_actions"]):
            raise RuntimeError("OpenDreamer source has an inconsistent binary action space")
        missing = set(_KEY_TO_VPT_NAME.values()) | set(_BUTTON_TO_VPT_NAME.values())
        missing |= {"mouse.wheel_neg", "mouse.wheel_pos", "unknown"}
        if missing.difference(source_mapping):
            raise RuntimeError("OpenDreamer source is missing required VPT actions")
        if int(self._dynamics.cfg.num_binary_actions) != len(source_mapping):
            raise RuntimeError("checkpoint binary action count does not match the source")
        if int(self._dynamics.cfg.categorical_action_dim) != int(self._deps["camera_classes"]):
            raise RuntimeError("checkpoint camera action count does not match the source")
        self._key_to_index = source_mapping

    def _consume_transient_controls(self) -> None:
        """Consume camera and wheel deltas after one generated frame."""
        if self.state is None:
            return
        self.state._delta_x = 0.0
        self.state._delta_y = 0.0
        self.state._wheel_delta = 0

    def _clear_controls(self) -> None:
        """Release held controls and discard transient input."""
        if self.state is None:
            return
        self.state._pressed_keys = frozenset()
        self.state._pressed_mouse_buttons = frozenset()
        self._consume_transient_controls()


def _read_config(config_path: Path | None) -> _Config:
    """Read and validate the OpenDreamer model YAML."""
    if config_path is None:
        raise ValueError("OpenDreamer requires runtime.config in reactor.yaml")
    document = yaml.safe_load(config_path.read_text())
    if not isinstance(document, dict):
        raise ValueError(f"{config_path}: expected a YAML mapping")

    source = _mapping(document.get("source"), "source")
    checkpoint = _mapping(document.get("checkpoint"), "checkpoint")
    conditioning = _mapping(document.get("conditioning", {}), "conditioning")
    source_revision = str(source.get("revision", ""))
    checkpoint_revision = str(checkpoint.get("revision", ""))
    if not _REVISION_PATTERN.fullmatch(source_revision):
        raise ValueError("source.revision must be a full 40-character Git revision")
    if not _REVISION_PATTERN.fullmatch(checkpoint_revision):
        raise ValueError("checkpoint.revision must be a full 40-character revision")

    platform = str(document.get("platform", "cuda"))
    if platform not in {"cuda", "auto"}:
        raise ValueError("platform must be cuda or auto")
    num_steps = int(document.get("num_steps", 4))
    if num_steps <= 0 or num_steps & (num_steps - 1):
        raise ValueError("num_steps must be a positive power of two")
    tau_ctx_target = float(document.get("tau_ctx_target", 0.9))
    if not 0.0 < tau_ctx_target < 1.0:
        raise ValueError("tau_ctx_target must be between 0 and 1")
    conditioning_frames = int(conditioning.get("frames", 16))
    if conditioning_frames < 16:
        raise ValueError("conditioning.frames must be at least 16")
    warmup_steps = int(document.get("warmup_steps", 1))
    if warmup_steps < 0:
        raise ValueError("warmup_steps must be non-negative")
    memory_fraction = float(document.get("memory_fraction", 0.9))
    if not 0.0 < memory_fraction <= 1.0:
        raise ValueError("memory_fraction must be in (0, 1]")

    demos = _read_demos(conditioning.get("demos"))
    return _Config(
        source_revision=source_revision,
        checkpoint_repo_id=str(checkpoint["repo_id"]),
        checkpoint_revision=checkpoint_revision,
        platform=platform,
        seed=int(document.get("seed", 0)),
        num_steps=num_steps,
        tau_ctx_target=tau_ctx_target,
        conditioning_frames=conditioning_frames,
        demos=demos,
        warmup_steps=warmup_steps,
        memory_fraction=memory_fraction,
    )


def _mapping(value: Any, name: str) -> Mapping[str, Any]:
    """Return a YAML mapping or raise a precise configuration error."""
    if not isinstance(value, dict):
        raise ValueError(f"{name} must be a YAML mapping")
    return value


def _read_demos(value: Any) -> tuple[_DemoConfig, ...]:
    """Validate the three demos exposed by the runtime command schema."""
    if not isinstance(value, list) or len(value) != len(_DEMO_CHOICES):
        raise ValueError("conditioning.demos must define demo_1, demo_2, and demo_3")
    demos: list[_DemoConfig] = []
    for index, item in enumerate(value):
        demo = _mapping(item, f"conditioning.demos[{index}]")
        expected_name = _DEMO_CHOICES[index]
        name = str(demo.get("name", ""))
        if name != expected_name:
            raise ValueError(f"conditioning.demos[{index}].name must be {expected_name}")
        start_frame = int(demo.get("start_frame", 0))
        if start_frame < 0:
            raise ValueError(f"conditioning.demos[{index}].start_frame must be non-negative")
        video = _relative_upstream_asset(demo.get("video"), f"conditioning.demos[{index}].video")
        actions = _relative_upstream_asset(
            demo.get("actions"),
            f"conditioning.demos[{index}].actions",
        )
        demos.append(
            _DemoConfig(
                name=name,
                video=video,
                actions=actions,
                start_frame=start_frame,
            )
        )
    return tuple(demos)


def _relative_upstream_asset(value: Any, name: str) -> Path:
    """Return a safe path relative to the configured upstream checkout."""
    if not isinstance(value, str) or not value:
        raise ValueError(f"{name} must be a non-empty relative path")
    path = Path(value)
    if path.is_absolute() or ".." in path.parts:
        raise ValueError(f"{name} must stay within OPENDREAMER_PATH")
    return path


def _upstream_root() -> Path:
    """Return the external OpenDreamer checkout configured for this process.

    Raises:
        RuntimeError: If ``OPENDREAMER_PATH`` is unset or not an OpenDreamer checkout.
    """
    configured = os.environ.get(_UPSTREAM_ENV)
    if not configured:
        raise RuntimeError(
            f"Set {_UPSTREAM_ENV} to the OpenDreamer repository checkout before starting Reactor"
        )

    root = Path(configured).expanduser().resolve()
    required = (
        root / "dreamer/actions.py",
        root / "dreamer/checkpointing.py",
        root / "dreamer/generation.py",
        root / "dreamer/models.py",
    )
    missing = [str(path.relative_to(root)) for path in required if not path.is_file()]
    if missing:
        joined = ", ".join(missing)
        raise RuntimeError(
            f"{_UPSTREAM_ENV}={root} is not an OpenDreamer checkout; missing: {joined}"
        )
    return root


def _upstream_asset(upstream_root: Path, relative_path: Path) -> Path:
    """Resolve one validated demo asset inside the upstream checkout."""
    return (upstream_root / relative_path).resolve()


def _ensure_demo_assets(upstream_root: Path, demos: tuple[_DemoConfig, ...]) -> None:
    """Download the public default demo when its configured files are missing."""
    missing = {
        path
        for demo in demos
        for path in (
            _upstream_asset(upstream_root, demo.video),
            _upstream_asset(upstream_root, demo.actions),
        )
        if not path.is_file()
    }
    if not missing:
        return

    module_name = f"{__package__}.opendreamer_assets" if __package__ else "opendreamer_assets"
    assets = importlib.import_module(module_name)
    output_dir = upstream_root / "samples" / "vpt"
    default_paths = set(assets.demo_paths(output_dir))
    if not missing.issubset(default_paths):
        return
    logger.info("downloading missing OpenDreamer demo assets", directory=str(output_dir))
    assets.ensure_demo_assets(output_dir)


def _verify_source_revision(source_path: Path, expected: str) -> None:
    """Require the local upstream clone to match the pinned public revision."""
    if not (source_path / "dreamer").is_dir():
        raise FileNotFoundError(f"OpenDreamer source not found at {source_path}")
    if not (source_path / ".git").exists():
        raise RuntimeError(f"OpenDreamer source at {source_path} must be a Git checkout")
    result = subprocess.run(
        ["git", "-C", str(source_path), "rev-parse", "HEAD"],
        check=True,
        capture_output=True,
        text=True,
    )
    actual = result.stdout.strip()
    if actual != expected:
        raise RuntimeError(f"OpenDreamer source is {actual}; expected {expected}")


def _prepare_process_environment(config: _Config) -> None:
    """Set JAX process options before importing JAX."""
    if config.platform == "cuda":
        os.environ.setdefault("JAX_PLATFORMS", "cuda")
    os.environ.setdefault("XLA_PYTHON_CLIENT_MEM_FRACTION", str(config.memory_fraction))


def _load_dependencies(source_path: Path) -> dict[str, Any]:
    """Import optional OpenDreamer dependencies after config validation."""
    source = str(source_path)
    if source not in sys.path:
        sys.path.insert(0, source)
    jax = importlib.import_module("jax")
    actions = importlib.import_module("dreamer.actions")
    checkpointing = importlib.import_module("dreamer.checkpointing")
    generation = importlib.import_module("dreamer.generation")
    models = importlib.import_module("dreamer.models")
    parallel = importlib.import_module("dreamer.parallel")
    utils = importlib.import_module("dreamer.utils")
    return {
        "jax": jax,
        "jnp": importlib.import_module("jax.numpy"),
        "nnx": importlib.import_module("flax.nnx"),
        "snapshot_download": importlib.import_module("huggingface_hub").snapshot_download,
        "action_type": actions.Actions,
        "binary_actions": actions.NUM_BINARY_ACTIONS,
        "camera_classes": actions.NUM_CAMERA_CLASSES,
        "key_to_index": actions.key_to_index,
        "mouse_to_categorical": actions.mouse_movement_to_categorical,
        "parse_action_dicts": actions.parse_action_dicts,
        "shift_actions": actions.shift_actions,
        "bundle_type": checkpointing.DynamicsCheckpointBundle,
        "schedule_type": generation.DenoiseSchedule,
        "next_frame": generation.next_frame,
        "tokenizer_caches_type": models.TokenizerCaches,
        "build_parallel": parallel.build_parallel,
        "normalize_latents": utils.normalize_latents,
    }


def _mesh_context(jax: Any, mesh: Any) -> AbstractContextManager[Any]:
    """Return the mesh context supported by the installed JAX version."""
    if hasattr(jax, "set_mesh"):
        return jax.set_mesh(mesh)
    return mesh


def _read_conditioning_sequence(
    video_path: Path,
    actions_path: Path,
    target_shape: tuple[int, int, int],
    *,
    start_frame: int,
    required_frames: int,
    dependencies: Mapping[str, Any],
) -> _ConditioningSequence:
    """Read one configured video and action window from disk."""
    if not video_path.is_file():
        raise FileNotFoundError(f"conditioning video not found at {video_path}")
    if not actions_path.is_file():
        raise FileNotFoundError(f"conditioning actions not found at {actions_path}")
    frames = _decode_video_frames(
        video_path,
        target_shape,
        start_frame=start_frame,
        required_frames=required_frames,
    )
    action_dicts = _load_action_dicts(actions_path.read_text())
    actions = _prepare_conditioning_actions(
        action_dicts,
        start_frame=start_frame,
        required_frames=required_frames,
        dependencies=dependencies,
    )
    return _ConditioningSequence(frames=frames, actions=actions)


def _decode_conditioning_image(
    data: bytes,
    target_shape: tuple[int, int, int],
) -> np.ndarray:
    """Decode and center-crop one upload into an OpenDreamer RGB frame."""
    image_module = importlib.import_module("PIL.Image")
    image_ops = importlib.import_module("PIL.ImageOps")
    height, width, channels = target_shape
    if channels != 3:
        raise ValueError(f"OpenDreamer requires three RGB channels, got {channels}.")
    content_height = height - 8 if height > 8 else height
    try:
        with image_module.open(io.BytesIO(data)) as uploaded:
            rgb = image_ops.exif_transpose(uploaded).convert("RGB")
            fitted = image_ops.fit(
                rgb,
                (width, content_height),
                method=image_module.Resampling.LANCZOS,
            )
            frame = np.asarray(fitted, dtype=np.uint8)
    except (OSError, ValueError) as error:
        raise ValueError("Could not decode the uploaded conditioning image.") from error
    return _prepare_video_frame(frame, target_shape, index=0)


def _decode_video_frames(
    source: Path | io.BytesIO,
    target_shape: tuple[int, int, int],
    *,
    start_frame: int,
    required_frames: int,
) -> np.ndarray:
    """Decode consecutive exact-size RGB frames from an MP4 source."""
    av = importlib.import_module("av")
    frames: list[np.ndarray] = []
    try:
        with av.open(source, mode="r") as container:
            for index, frame in enumerate(container.decode(video=0)):
                if index < start_frame:
                    continue
                if len(frames) == required_frames:
                    break
                rgb = np.asarray(frame.to_ndarray(format="rgb24"), dtype=np.uint8)
                frames.append(_prepare_video_frame(rgb, target_shape, index=index))
    except (av.FFmpegError, IndexError) as error:
        raise ValueError("Could not decode the conditioning MP4.") from error
    if len(frames) < required_frames:
        raise ValueError(
            f"Conditioning video has {len(frames)} frames from offset {start_frame}; "
            f"expected at least {required_frames}."
        )
    return np.ascontiguousarray(np.stack(frames))


def _prepare_video_frame(
    frame: np.ndarray,
    target_shape: tuple[int, int, int],
    *,
    index: int,
) -> np.ndarray:
    """Validate one model frame and pad the native 640x360 VPT format."""
    height, width, channels = target_shape
    if frame.shape == target_shape:
        return np.ascontiguousarray(frame)
    if frame.shape == (height - 8, width, channels):
        return np.pad(frame, ((4, 4), (0, 0), (0, 0)), mode="constant")
    raise ValueError(
        f"Conditioning frame {index} has shape {frame.shape}; expected "
        f"{target_shape} or {(height - 8, width, channels)}."
    )


def _load_action_dicts(text: str) -> list[dict[str, Any]]:
    """Parse a JSON array or newline-delimited VPT action objects."""
    stripped = text.lstrip()
    if not stripped:
        raise ValueError("The conditioning action file is empty.")
    if stripped.startswith("["):
        document = json.loads(text)
        if not isinstance(document, list) or not all(isinstance(item, dict) for item in document):
            raise ValueError("The conditioning JSON must be an array of objects.")
        return document

    actions: list[dict[str, Any]] = []
    for line_number, line in enumerate(text.splitlines(), start=1):
        if not line.strip():
            continue
        try:
            item = json.loads(line)
        except json.JSONDecodeError as error:
            raise ValueError(f"Invalid conditioning JSONL at line {line_number}.") from error
        if not isinstance(item, dict):
            raise ValueError(f"Conditioning JSONL line {line_number} must be an object.")
        actions.append(item)
    if not actions:
        raise ValueError("The conditioning action file contains no actions.")
    return actions


def _prepare_conditioning_actions(
    action_dicts: list[dict[str, Any]],
    *,
    start_frame: int,
    required_frames: int,
    dependencies: Mapping[str, Any],
) -> Any:
    """Parse, batch, shift, and slice actions to match conditioning frames."""
    required_actions = start_frame + required_frames
    if len(action_dicts) < required_actions:
        raise ValueError(
            f"Conditioning actions contain {len(action_dicts)} entries; "
            f"expected at least {required_actions}."
        )
    jnp = dependencies["jnp"]
    action_type = dependencies["action_type"]
    parsed = dependencies["parse_action_dicts"](action_dicts[:required_actions])

    def add_batch(value: Any) -> Any:
        return None if value is None else jnp.asarray(value)[None]

    batched = action_type(
        binary=add_batch(parsed.binary),
        categorical=add_batch(parsed.categorical),
        continuous=add_batch(parsed.continuous),
    )
    shifted = dependencies["shift_actions"](
        batched,
        int(dependencies["camera_classes"]),
    )
    return shifted[:, start_frame:required_actions]
