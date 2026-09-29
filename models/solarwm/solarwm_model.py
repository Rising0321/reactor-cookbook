"""SolarWM model ownership and immutable CPU step contract."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np


@dataclass(frozen=True)
class SolarWMAnchor:
    image: bytes
    prompt: str
    seed: int


@dataclass(frozen=True)
class SolarWMInput:
    world_id: int
    anchor: SolarWMAnchor | None
    poses: np.ndarray


@dataclass(frozen=True)
class SolarWMResult:
    world_id: int
    chunk_index: int
    frames: np.ndarray
    complete: bool
    max_chunks: int


class NoAnchor(Exception):
    """A fresh world requires its image and conditioning."""


class RolloutExhausted(Exception):
    """The configured native rollout limit has been reached."""


class SolarWMModel:
    def __init__(self) -> None:
        self.backend = None
        self.world_id: int | None = None
        self.chunk_index = 0
        self.max_chunks = 320

    def load(self, config_path: Path | None) -> None:
        from solarwm_backend import BackendSettings, SolarWMBackend
        from solarwm_config import prepare_runtime, read_config

        config = read_config(config_path)
        prepare_runtime(config)
        self.max_chunks = config.max_chunks
        self.backend = SolarWMBackend(
            BackendSettings(
                config.source_path,
                config.upstream_config,
                config.base_path,
                config.checkpoint_path,
                config.runtime_root,
            )
        )

    def generate(self, input: SolarWMInput) -> SolarWMResult:
        if input.world_id != self.world_id:
            if input.anchor is None:
                raise NoAnchor("a fresh world requires an anchor")
            self.backend.reset(
                input.anchor.seed, input.anchor.image, input.anchor.prompt
            )
            self.world_id = input.world_id
            self.chunk_index = 0
        if self.chunk_index >= self.max_chunks:
            raise RolloutExhausted("reset the world before continuing")
        frames = self.backend.generate_chunk(input.poses)
        self.chunk_index += 1
        return SolarWMResult(
            self.world_id,
            self.chunk_index,
            frames,
            self.chunk_index >= self.max_chunks,
            self.max_chunks,
        )

    def reset(self) -> None:
        if self.backend is not None:
            self.backend.end_session()
        self.world_id = None
        self.chunk_index = 0
