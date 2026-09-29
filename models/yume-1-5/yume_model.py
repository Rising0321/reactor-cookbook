"""Runtime-independent YUME rolling-latent model and CPU step contract."""

from __future__ import annotations

import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import Literal

import numpy as np

Movement = Literal[
    "none",
    "forward",
    "backward",
    "left",
    "right",
    "forward_left",
    "forward_right",
    "backward_left",
    "backward_right",
]
View = Literal[
    "none",
    "pan_left",
    "pan_right",
    "tilt_up",
    "tilt_down",
    "tilt_up_left",
    "tilt_up_right",
    "tilt_down_left",
    "tilt_down_right",
]


@dataclass(frozen=True)
class YumeAnchor:
    mode: str
    media: bytes | None
    suffix: str
    seed: int


@dataclass(frozen=True)
class YumeInput:
    world_id: int
    anchor: YumeAnchor | None
    prompt: str
    movement: Movement
    view: View


@dataclass(frozen=True)
class YumeResult:
    world_id: int
    chunk_index: int
    frames: np.ndarray
    prompt: str
    conditioned_prompt: str
    movement: Movement
    view: View


class NoAnchor(Exception):
    """A fresh YUME world needs its selected scene."""


class YumeModel:
    def __init__(self) -> None:
        self.backend = None
        self.world_id: int | None = None
        self.chunk_index = 0

    def load(self, config_path: Path | None, weights_root: Path) -> None:
        from yume_assets import activate_source, prepare_assets, read_config

        self.config = read_config(config_path, weights_root)
        prepare_assets(self.config)
        activate_source(self.config)
        from yume_backend import YumeBackend

        self.backend = YumeBackend(self.config)

    def generate(self, input: YumeInput) -> YumeResult:
        if input.world_id != self.world_id:
            anchor = input.anchor
            if anchor is None:
                raise NoAnchor("a new world requires a scene anchor")
            kwargs = {
                "image": None,
                "video": None,
                "prompt": input.prompt,
                "seed": anchor.seed,
                "movement": "none",
                "view": "none",
            }
            if anchor.media is None:
                self.backend.reset(**kwargs)
            else:
                with tempfile.NamedTemporaryFile(
                    suffix=anchor.suffix, dir=self.config.runtime_dir
                ) as media:
                    media.write(anchor.media)
                    media.flush()
                    kwargs["image" if anchor.mode == "image_to_video" else "video"] = (
                        Path(media.name)
                    )
                    self.backend.reset(**kwargs)
            self.world_id = input.world_id
            self.chunk_index = 0
        frames, exact_prompt = self.backend.generate_chunk(
            prompt=input.prompt, movement=input.movement, view=input.view
        )
        self.chunk_index += 1
        return YumeResult(
            self.world_id,
            self.chunk_index,
            np.ascontiguousarray(frames),
            input.prompt,
            exact_prompt,
            input.movement,
            input.view,
        )

    def reset(self) -> None:
        if self.backend is not None:
            self.backend.end_session()
        self.world_id = None
        self.chunk_index = 0
