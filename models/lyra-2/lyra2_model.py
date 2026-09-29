"""Runtime-independent Lyra weights, native caches, and rollout bookkeeping."""

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import yaml

from lyra2_backend import Lyra2Backend


@dataclass(frozen=True)
class Lyra2Input:
    world_id: int
    anchor: np.ndarray | None
    prompt: str
    seed: int
    w2c: np.ndarray | None
    intrinsics: np.ndarray | None


@dataclass(frozen=True)
class Lyra2Result:
    world_id: int
    chunk: int
    prompt: str
    frames: np.ndarray | None
    corrected_c2w: np.ndarray | None
    intrinsics: np.ndarray | None


class NoAnchor(Exception):
    """A new world needs an image."""


class NoCamera(Exception):
    """A seeded world needs calibrated camera poses."""


class Lyra2Model:
    def __init__(self) -> None:
        self.backend: Lyra2Backend | None = None
        self.world_id: int | None = None
        self.chunk = 0

    def load(self, config_path: Path, weights_root: Path) -> None:
        config = yaml.safe_load(config_path.read_text())
        source = Path(config["source_path"]).expanduser()
        config["source_path"] = str(source if source.is_absolute() else config_path.parent / source)
        for key in ("cache_path", "output_path"):
            path = Path(config[key]).expanduser()
            config[key] = str(path if path.is_absolute() else weights_root / path)
        self.backend = Lyra2Backend(config, weights_root)

    def reset(self) -> None:
        if self.backend is not None:
            self.backend.clear()
        self.world_id = None
        self.chunk = 0

    def generate(self, input: Lyra2Input) -> Lyra2Result:
        if self.backend is None:
            raise RuntimeError("Lyra-2 not loaded")
        if input.world_id != self.world_id:
            if input.anchor is None:
                raise NoAnchor("A new Lyra world requires an anchor image")
            c2w, intrinsics = self.backend.reset(input.anchor, prompt=input.prompt, seed=input.seed)
            self.world_id = input.world_id
            self.chunk = 0
            return Lyra2Result(input.world_id, 0, input.prompt, None, c2w, intrinsics)
        if input.w2c is None or input.intrinsics is None:
            raise NoCamera("Camera poses must follow seed calibration")
        frames, corrected = self.backend.generate_chunk(
            input.w2c, input.intrinsics, prompt=input.prompt, chunk=self.chunk + 1,
        )
        self.chunk += 1
        return Lyra2Result(input.world_id, self.chunk, input.prompt, frames, corrected, None)
