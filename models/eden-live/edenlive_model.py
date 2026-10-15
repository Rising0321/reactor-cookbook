"""Runtime-independent, one-chunk EdenLive inference contract."""

from dataclasses import dataclass
from pathlib import Path

import numpy as np


@dataclass(frozen=True, kw_only=True)
class EdenAnchor:
    """Fresh-world reference: RGB uint8 [H,W,3], opening text and RNG seed.

    Pixels retain native precision until the backend's float bilinear resize.
    The application decodes once, with a 16-megapixel bound.
    """

    image: np.ndarray  # RGB uint8 [H,W,3], before the native float resize.
    prompt: str
    seed: int


@dataclass(frozen=True, kw_only=True)
class EdenInput:
    """One frozen step: requested world, current text and three proxy images.

    ``anchor`` crosses only until the result acknowledges ``world_id``.
    ``condition`` contains CPU RGB uint8 [3,704,1280,3], sampled at local
    positions 0, 4 and 8 of the matching simulation chunk.
    """

    world_id: int
    anchor: EdenAnchor | None
    prompt: str
    condition: np.ndarray  # RGB uint8 [3,704,1280,3], chunk-local frames 0,4,8.


@dataclass(frozen=True, kw_only=True)
class EdenResult:
    """Completed world id, one-based chunk count and CPU RGB uint8 pixels.

    ``frames`` is [9,704,1280,3] initially and [12,704,1280,3] thereafter.
    ``frames_wanted`` tells the application how many next proxy frames to capture.
    """

    world_id: int
    chunk_index: int
    frames: np.ndarray  # RGB uint8 [T,704,1280,3]; T=9, then 12.
    frames_wanted: int


class MissingAnchor(ValueError):
    """A new world needs its reference image, opening prompt and seed."""


class EdenModel:
    """Own model weights, native causal caches and the completed chunk count."""

    def __init__(self) -> None:
        self._backend = None
        self._world_id = None
        self._chunk_index = 0

    def load(
        self,
        *,
        source: Path,
        weights: Path,
        attention_backend: str = "native",
        cuda_graph: bool = False,
        compile_model: bool = False,
    ) -> None:
        """Load the authorized native model once, in the Runner's worker."""
        from edenlive_backend import EdenBackend

        self._backend = EdenBackend(
            source=source,
            weights=weights,
            attention_backend=attention_backend,
            cuda_graph=cuda_graph,
            compile_model=compile_model,
        )

    def reset(self) -> None:
        """Drop the current rollout and caches while retaining loaded weights."""
        if self._backend is not None:
            self._backend.reset()
        self._world_id = None
        self._chunk_index = 0

    def generate(self, input: EdenInput) -> EdenResult:
        """Return one chunk; raise MissingAnchor for an unknown world without one.

        Malformed geometry raises ValueError. Native compute failures propagate;
        they must end the serving session rather than claim a completed chunk.
        """
        if (
            input.condition.shape != (3, 704, 1280, 3)
            or input.condition.dtype != np.uint8
        ):
            raise ValueError("Expected three native 1280x704 RGB condition frames")
        if not input.prompt.strip():
            raise ValueError("The chunk prompt must be non-empty")
        if input.world_id != self._world_id:
            if input.anchor is None:
                raise MissingAnchor("A fresh world requires an anchor")
            self._backend.reset()
            self._backend.start(input.anchor)
            self._world_id = input.world_id
            self._chunk_index = 0
        frames = self._backend.step(self._chunk_index, input.condition, input.prompt)
        expected = 9 if self._chunk_index == 0 else 12
        if frames.shape != (expected, 704, 1280, 3) or frames.dtype != np.uint8:
            raise ValueError("Native output does not match the causal chunk geometry")
        self._chunk_index += 1
        return EdenResult(
            world_id=self._world_id,
            chunk_index=self._chunk_index,
            frames=frames,
            frames_wanted=12,
        )
