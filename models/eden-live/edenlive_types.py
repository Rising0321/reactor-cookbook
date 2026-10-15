"""Client-facing EdenLive commands, shared state and synchronized video."""

from typing import Literal

import numpy as np
from reactor_runtime import (
    InputField,
    InputState,
    MessageField,
    ModelMessage,
    Output,
    Video,
)

Key = Literal[
    "w",
    "a",
    "s",
    "d",
    "arrowup",
    "arrowdown",
    "arrowleft",
    "arrowright",
    "f",
    " ",
    "shift",
    "e",
    "r",
    "g",
]


class EdenState(InputState):
    prompt: str = InputField(
        default="",
        max_length=4096,
        moderate=True,
        description="Scene description. Changes condition the next chunk while visual history stays.",
    )
    keys: list[Key] | None = InputField(
        default=None,
        max_length=14,
        description="Latest held native keys. Empty releases all; sampled before each uncaptured draft frame.",
    )
    paused: bool = InputField(
        default=True,
        description="Hold simulation and generation. Resume continues the same world after Start.",
    )
    _game: str = ""
    _image: np.ndarray | None = None
    _image_name: str = ""
    _seed: int = 2026092901
    _world_id: int = 0
    _applied_world_id: int | None = None
    _started: bool = False
    _chunks: int = 0
    _frames_wanted: int = 9
    _seconds: float | None = None
    _proxy: np.ndarray | None = None
    _packet: object | None = None
    _ended: bool = False
    _sampled_keys: tuple = ()


class EdenStatus(ModelMessage):
    """Complete shared state, broadcast after a transition or completed chunk."""

    game: str = MessageField(description="Selected game, empty before selection.")
    image_name: str = MessageField(
        description="Selected appearance reference filename."
    )
    prompt: str = MessageField(
        description="Active base prompt; native game events extend it per chunk."
    )
    seed: int = MessageField(description="Seed for the current world.")
    keys: list[str] = MessageField(
        description="Latest accepted held controls. Already captured frames retain their original input."
    )
    paused: bool = MessageField(description="Whether generation is paused.")
    started: bool = MessageField(description="Whether Start has created a world.")
    ended: bool = MessageField(
        description="Whether the game reached an outcome; Start begins a fresh world."
    )
    world_id: int = MessageField(description="Identity of the selected rollout.")
    chunks: int = MessageField(
        description="Successfully completed chunks in this world."
    )
    seconds: float | None = MessageField(
        description="Runtime-measured seconds for the last model chunk."
    )


class EdenOutput(Output):
    """One synchronized image per timestamp: native draft left, generated frame right."""

    main_video: Video
