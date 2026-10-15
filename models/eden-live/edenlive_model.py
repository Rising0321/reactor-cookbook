# Copyright (c) 2026 Reactor Technologies, Inc. All rights reserved.
"""Runtime-independent, one-chunk EdenLive inference contract."""

import logging
import time
from dataclasses import dataclass
from pathlib import Path

import numpy as np

logger = logging.getLogger(__name__)

_WARMUP_PROMPT = "A stone corridor lit by torches."

# One condition frame per latent frame: chunk-local draft frames 0, 4, 8.
CONDITION_STRIDE = 4


def condition_frames(draft: np.ndarray) -> np.ndarray:
    """Select the draft frames a step conditions on, whichever source drew them."""
    return np.ascontiguousarray(draft[::CONDITION_STRIDE])


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
    """One frozen step: requested world, current text and three draft images.

    ``anchor`` crosses only until the result acknowledges ``world_id``.
    ``condition`` contains CPU RGB uint8 [3,704,1280,3]: the chunk's draft
    frames 0, 4 and 8, from ``condition_frames``. The model does not know
    which source drew them.
    """

    world_id: int
    anchor: EdenAnchor | None
    prompt: str
    condition: np.ndarray  # RGB uint8 [3,704,1280,3], chunk-local frames 0,4,8.


@dataclass(frozen=True, kw_only=True)
class EdenResult:
    """Completed world id, one-based chunk count and CPU RGB uint8 pixels.

    ``frames`` is [9,704,1280,3] initially and [12,704,1280,3] thereafter.
    ``frames_wanted`` tells the application how many draft frames the next step needs.
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
        warmup: bool = False,
    ) -> None:
        """Load the authorized native model once, in the Runner's worker.

        With ``warmup``, ``load()`` also generates throwaway worlds until every
        compiled shape and CUDA graph a session uses exists, so the first
        session's chunks run at steady-state speed.
        """
        from edenlive_backend import EdenBackend

        self._backend = EdenBackend(
            source=source,
            weights=weights,
            attention_backend=attention_backend,
            cuda_graph=cuda_graph,
            compile_model=compile_model,
        )
        if warmup:
            self._warm_up()

    def _warm_up(self) -> None:
        """Run every path a session takes once, then forget the throwaway worlds.

        A world's first chunks grow the cache, so each has its own shape; graphs
        are captured once the cache is full, captured again after a prompt
        change, and again in every new world. The first world runs past its
        capture and through a prompt change, and the second checks that a fresh
        world reuses what the first compiled.
        """
        backend = self._backend
        steady = backend.capture_index + backend.graph_warmup + 2
        recapture = backend.graph_warmup + 2
        rng = np.random.default_rng(0)
        started = time.perf_counter()
        for world, chunks in enumerate((steady + recapture, steady)):
            image = rng.integers(0, 256, (704, 1280, 3), np.uint8)
            backend.reset()
            backend.start(EdenAnchor(image=image, prompt=_WARMUP_PROMPT, seed=world))
            seconds = []
            for index in range(chunks):
                prompt = _WARMUP_PROMPT if index < steady else f"{_WARMUP_PROMPT} It rains."
                condition = rng.integers(0, 256, (3, 704, 1280, 3), np.uint8)
                begun = time.perf_counter()
                backend.step(index, condition, prompt)
                seconds.append(round(time.perf_counter() - begun, 3))
            logger.info("EdenLive warmup world %d chunk seconds: %s", world, seconds)
        backend.reset()
        logger.info("EdenLive warmup done in %.1f s", time.perf_counter() - started)

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
        if input.condition.shape != (3, 704, 1280, 3) or input.condition.dtype != np.uint8:
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
