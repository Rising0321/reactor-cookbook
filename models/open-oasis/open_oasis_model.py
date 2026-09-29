"""Runtime-independent Open-Oasis rollout and step contract."""
from dataclasses import dataclass
from pathlib import Path

import numpy as np
from open_oasis_assets import read_config, prepare_source, download_checkpoints
from open_oasis_backend import OpenOasisBackend


@dataclass(frozen=True)
class OpenOasisInput:
    world_id: int
    conditioning: np.ndarray | None
    seed: int
    action: np.ndarray


@dataclass(frozen=True)
class OpenOasisResult:
    world_id: int
    frame: np.ndarray
    index: int


class NotSeeded(Exception):
    """A new world requires visual conditioning."""


class OpenOasisModel:
    def __init__(self) -> None:
        self.backend: OpenOasisBackend | None = None
        self.world_id: int | None = None
        self.index = 0

    def load(self, config_path: Path | None, weights_root: Path) -> None:
        config = read_config(config_path)
        prepare_source(config)
        model_path, vae_path = download_checkpoints(config, weights_root)
        self.backend = OpenOasisBackend(config, model_path, vae_path)

    def reset(self) -> None:
        self.world_id = None
        self.index = 0
        if self.backend is not None:
            self.backend.latents = None
            self.backend.actions = None
            self.backend.generator = None

    def generate(self, input: OpenOasisInput) -> OpenOasisResult:
        if self.backend is None:
            raise RuntimeError("Open-Oasis was not loaded")
        if input.world_id != self.world_id:
            if input.conditioning is None:
                raise NotSeeded("no conditioning for new world")
            self.backend.reset(input.conditioning, input.seed)
            self.world_id = input.world_id
            self.index = 0
            frame = input.conditioning[-1]
        else:
            frame = self.backend.generate_one(input.action)
            self.index += 1
        return OpenOasisResult(self.world_id, np.ascontiguousarray(frame), self.index)
