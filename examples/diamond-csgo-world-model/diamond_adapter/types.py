"""Define the typed configuration and Reactor contract for the DIAMOND adapter."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from reactor_runtime import InputField, InputState, MessageField, ModelMessage, Output, Video

KEYS = ["w", "a", "s", "d", "space", "ctrl", "shift", "1", "2", "3", "r"]
MOUSE_BUTTONS = ["left", "right"]
CONTROLLERS = ["human", "replay"]
DELTA_X_MIN = -1000.0
DELTA_X_MAX = 1000.0
DELTA_Y_MIN = -200.0
DELTA_Y_MAX = 200.0


@dataclass(frozen=True)
class AdapterConfig:
    """Hold the adapter settings read from ``diamond.yaml``."""

    repo_id: str
    revision: str
    device: str
    profile: str
    seed: int


@dataclass(frozen=True)
class PreparedScene:
    """Hold one device-ready initial condition for the next reset."""

    obs: Any
    obs_full_res: Any
    act: Any
    next_act: Any | None


class DiamondOutput(Output):
    """Carry the generated Counter-Strike frame."""

    main_video: Video


class ActionChanged(ModelMessage):
    """Describe the native input state after a control command is accepted."""

    controller: str = MessageField(
        description="Active action source: human input or dataset replay."
    )
    pressed_keys: list[str] = MessageField(
        description="Native keyboard keys currently held for human control."
    )
    pressed_mouse_buttons: list[str] = MessageField(
        description="Native mouse buttons currently held for human control."
    )
    delta_x: float = MessageField(
        description="Horizontal relative mouse delta accepted by this command, otherwise zero."
    )
    delta_y: float = MessageField(
        description="Vertical relative mouse delta accepted by this command, otherwise zero."
    )


class SceneChanged(ModelMessage):
    """Describe the scene queued for the next world-model reset boundary."""

    source: str = MessageField(description="Queued scene source: upload or dataset.")
    scene: str = MessageField(description="Uploaded filename or queued dataset scene identifier.")


class DiamondState(InputState):
    """Hold per-connection controls while session-wide world state stays on the model."""

    controller: str = InputField(
        default="human",
        choices=CONTROLLERS,
        description="Use human commands or the current spawn's recorded action trajectory.",
    )
    paused: bool = InputField(
        default=False,
        description="Pause expensive world-model steps while preserving the shared scene.",
    )
    _pressed_keys: frozenset[str] = frozenset()
    _pressed_mouse_buttons: frozenset[str] = frozenset()
    _delta_x: float = 0.0
    _delta_y: float = 0.0
    _step_requested: bool = False
