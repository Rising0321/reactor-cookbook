# Copyright (c) 2026 Reactor Technologies, Inc. All rights reserved.
"""The draft chunk every source produces, and the client's video as a source.

A draft source is where the video EdenLive reskins comes from. The application
holds one source at a time, either the native game (``edenlive_game.EdenGame``)
or the client's ``source_video`` track (``ClientSource``), and drives both
through the same three calls:

- ``prepare(frames)`` asks for the next chunk of ``frames`` draft frames.
- ``await take()`` returns that chunk as a ``Draft``, or ``None`` while it is
  not ready yet.
- ``set_keys(held)``, ``set_paused(paused)`` and ``close()`` forward the
  session's controls; a source that has no use for one ignores it.

Past ``take()`` nothing knows which source made a chunk: the application
derives the model's condition frames, the prompt, and the left half of the
output from the ``Draft`` alone.
"""

import asyncio
from collections.abc import Sequence
from dataclasses import dataclass
from typing import Any, Protocol

import numpy as np
from PIL import Image
from reactor_runtime import BufferClosed, ReadMode

DRAFT_WIDTH = 1280
DRAFT_HEIGHT = 704


@dataclass(frozen=True, kw_only=True)
class Draft:
    """One chunk of draft video, ready to be reskinned.

    ``frames`` is RGB uint8 [T,704,1280,3]. ``metadata`` holds one dict per
    frame, published on `main_video` beside the frame. ``events`` are prompt
    sentences the source adds to this chunk only. ``ended`` says the source
    reached an outcome and cannot continue. ``capture_seconds`` is how long
    the source took to produce the chunk once asked, when it measures that.
    """

    frames: np.ndarray
    metadata: tuple[dict[str, Any], ...]
    events: tuple[str, ...] = ()
    ended: bool = False
    capture_seconds: float | None = None


class DraftSource(Protocol):
    """What the application needs from a source of draft video."""

    kind: str

    def prepare(self, frames: int) -> None: ...

    async def take(self) -> Draft | None: ...

    def set_keys(self, held: tuple[str, ...]) -> None: ...

    def set_paused(self, paused: bool) -> None: ...

    def close(self) -> None: ...


def fit_frames(frames: Sequence[np.ndarray]) -> np.ndarray:
    """Scale each RGB frame to cover 1280x704 and center-crop it there."""
    fitted = []
    for frame in frames:
        height, width = frame.shape[:2]
        if (width, height) != (DRAFT_WIDTH, DRAFT_HEIGHT):
            scale = max(DRAFT_WIDTH / width, DRAFT_HEIGHT / height)
            size = (
                max(DRAFT_WIDTH, round(width * scale)),
                max(DRAFT_HEIGHT, round(height * scale)),
            )
            left = (size[0] - DRAFT_WIDTH) // 2
            top = (size[1] - DRAFT_HEIGHT) // 2
            picture = Image.fromarray(np.ascontiguousarray(frame[..., :3]))
            picture = picture.resize(size, Image.Resampling.BILINEAR)
            frame = np.asarray(picture.crop((left, top, left + DRAFT_WIDTH, top + DRAFT_HEIGHT)))
        fitted.append(frame)
    return np.stack(fitted).astype(np.uint8, copy=False)


class ClientSource:
    """Draft frames read from the client's `source_video` track, newest first.

    Each chunk is the latest ``frames`` frames the client sent, so the draft
    follows the client's live video and never replays a backlog. The client
    has no keys or outcome; those calls do nothing.
    """

    kind = "client"

    def __init__(self, track) -> None:
        self._track = track
        self._wanted = 0

    def prepare(self, frames: int) -> None:
        self._wanted = frames

    async def take(self) -> Draft | None:
        if not self._wanted:
            return None
        try:
            frames = self._track.try_read(self._wanted, mode=ReadMode.LATEST)
        except BufferClosed:
            return None
        if frames is None:
            return None
        pixels = await asyncio.to_thread(fit_frames, [frame.data for frame in frames])
        metadata = tuple({"source": self.kind, "pts": frame.pts} for frame in frames)
        return Draft(frames=pixels, metadata=metadata)

    def set_keys(self, held: tuple[str, ...]) -> None:
        pass

    def set_paused(self, paused: bool) -> None:
        pass

    def close(self) -> None:
        pass
