"""Runtime-independent native Zing rollout and CPU step contract."""
from dataclasses import dataclass
from pathlib import Path
import tempfile

import numpy as np
from PIL import Image
from zing_assets import read_config, configure_environment, prepare_assets, activate_source


@dataclass(frozen=True)
class ZingInput:
    world_id: int
    image: np.ndarray | None
    prompt: str
    seed: int
    pressed_keys: frozenset[str]
    image_required: bool = False


@dataclass(frozen=True)
class ZingResult:
    world_id: int
    frames: np.ndarray
    index: int
    prompt: str
    pressed_keys: frozenset[str]
    cache_frames: int
    complete: bool


class RolloutComplete(Exception):
    """The configured native rollout horizon has been reached."""


class NotSeeded(Exception):
    """An image-conditioned world requires its anchor on its first step."""


class ZingModel:
    def __init__(self) -> None:
        self.backend = None
        self.config = None
        self.world_id: int | None = None
        self.index = 0

    def load(self, config_path: Path | None, weights_root: Path) -> None:
        self.config = read_config(config_path, weights_root)
        configure_environment(self.config)
        prepare_assets(self.config)
        activate_source(self.config)
        from zing_backend import ZingBackend
        self.backend = ZingBackend(self.config)

    def reset(self) -> None:
        if self.backend is not None:
            self.backend.end_session()
        self.world_id = None
        self.index = 0

    def generate(self, input: ZingInput) -> ZingResult:
        if self.backend is None or self.config is None:
            raise RuntimeError("Zing is not loaded")
        if input.world_id != self.world_id:
            if input.image_required and input.image is None:
                raise NotSeeded("no anchor for new image-conditioned world")
            if input.image is None:
                self.backend.reset(image=None, prompt=input.prompt, seed=input.seed)
            else:
                directory = self.config.asset_path / "uploads"
                directory.mkdir(parents=True, exist_ok=True)
                with tempfile.NamedTemporaryFile(suffix=".png", dir=directory) as image:
                    Image.fromarray(input.image).save(image.name)
                    self.backend.reset(image=Path(image.name), prompt=input.prompt, seed=input.seed)
            self.world_id = input.world_id
            self.index = 0
        if self.index >= self.config.max_chunks:
            raise RolloutComplete("rollout limit reached; reset required")
        frames = self.backend.generate_chunk(prompt=input.prompt, pressed_keys=input.pressed_keys)
        self.index += 1
        return ZingResult(
            self.world_id, frames, self.index, input.prompt, input.pressed_keys,
            self.backend.cache_frames(), self.index >= self.config.max_chunks,
        )
