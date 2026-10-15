# Copyright (c) 2026 Reactor Technologies, Inc. All rights reserved.
"""The native game as a draft source: simulation, capture, and event prompts."""

import asyncio
import time
from collections.abc import Mapping
from pathlib import Path
from threading import Condition

from edenlive_draft import Draft, fit_frames
from edenlive_model import CONDITION_STRIDE

# The shipped games report their HUD state here; anything else is an outcome.
_RUNNING_STATES = ("", "RUNNING", "READY")


def event_lines(chunk) -> tuple[str, ...]:
    """Match the native one-chunk event lifetime and sentence deduplication."""
    indexed = []
    for frame in chunk.frames:
        state = frame.ui_state
        events = state.get("events") if isinstance(state, Mapping) else None
        for event in events if isinstance(events, list) else ():
            if not isinstance(event, Mapping):
                continue
            kind, text, event_id = (
                event.get("type"),
                event.get("prompt"),
                event.get("id"),
            )
            if (
                isinstance(kind, str)
                and kind
                and isinstance(text, str)
                and text.strip()
                and isinstance(event_id, int)
                and not isinstance(event_id, bool)
            ):
                indexed.append((event_id, text.strip()))
    return tuple(dict.fromkeys(text for _, text in sorted(indexed, key=lambda pair: pair[0])))


def reached_outcome(ui_state) -> bool:
    """Whether a frame's HUD state says the game is over (won, lost, ...)."""
    return isinstance(ui_state, Mapping) and ui_state.get("state", "") not in _RUNNING_STATES


class EdenGame:
    """The native game as a draft source.

    One bounded capture is in flight at a time, with a latest-held-keys mailbox
    sampled once per captured frame.
    """

    kind = "game"

    def __init__(self, *, source: Path, slug: str, size: tuple[int, int] = (1280, 704)) -> None:
        from eden.apps.game2v.game_renderer import GameCaptureWorker
        from eden.apps.game2v.game_timeline import GameTimeline

        self.timeline = GameTimeline(slug)
        self.capture = GameCaptureWorker(
            slug, width=size[0], height=size[1], games_root=source / "realtime/games"
        )
        self._held = set()
        self._latest = frozenset()
        self._gate = Condition()
        self._paused = False
        self._closed = False
        self._pending = None
        self._sampled = []
        self._submitted_at = None
        self._captured_at = None

    def set_keys(self, held: tuple[str, ...]) -> None:
        with self._gate:
            self._latest = frozenset(held)

    def set_paused(self, paused: bool) -> None:
        with self._gate:
            self._paused = paused
            self._gate.notify_all()

    def _drain(self):
        # The capture thread reads plain application-owned values only. It
        # never touches Runtime state, handlers, the model or its caches.
        with self._gate:
            self._gate.wait_for(lambda: not self._paused or self._closed)
            if self._closed:
                raise CaptureClosed("Capture stopped")
            chosen = self._latest
        events = [{"action": key, "phase": "up"} for key in sorted(self._held - chosen)] + [
            {"action": key, "phase": "down"} for key in sorted(chosen - self._held)
        ]
        self._held = set(chosen)
        self._sampled.append(tuple(sorted(chosen)))
        return events

    def prepare(self, frames: int) -> None:
        """Keep at most one capture in flight, including a completed one."""
        if self._pending is None:
            self._sampled = []
            packet = self.timeline.begin_chunk(frames, 24)
            self._submitted_at = time.monotonic()
            self._captured_at = None
            self._pending = self.capture.submit(packet, drain_events=self._drain)
            self._pending.add_done_callback(self._stamp_capture)

    def _stamp_capture(self, _future) -> None:
        self._captured_at = time.monotonic()

    def ready(self) -> bool:
        return self._pending is not None and self._pending.done()

    async def take(self) -> Draft | None:
        """Return the captured chunk once it is ready; every frame keeps its keys."""
        if not self.ready():
            return None
        chunk = self._pending.result()
        self._pending = None
        sampled = tuple(self._sampled)
        if len(sampled) != len(chunk.frames):
            raise RuntimeError("Input samples do not match captured frames")
        conditioned = [i for i, frame in enumerate(chunk.frames) if frame.capture_condition]
        if conditioned != list(range(0, len(chunk.frames), CONDITION_STRIDE)):
            raise RuntimeError("Native condition frames are not every fourth draft frame")
        metadata = tuple(
            {
                "source": self.kind,
                "frame_id": frame.game_frame_id,
                "simulation_tick": frame.simulation_tick,
                "keys": list(keys),
            }
            for frame, keys in zip(chunk.frames, sampled, strict=True)
        )
        # The game may render below the draft size; scaling up runs off the loop.
        frames = await asyncio.to_thread(fit_frames, [frame.ui_frame for frame in chunk.frames])
        return Draft(
            frames=frames,
            metadata=metadata,
            events=event_lines(chunk),
            ended=reached_outcome(chunk.frames[-1].ui_state),
            capture_seconds=None
            if self._captured_at is None
            else self._captured_at - self._submitted_at,
        )

    def close(self):
        with self._gate:
            self._closed = True
            self._gate.notify_all()
        self.capture.close()
        self._pending = None


class CaptureClosed(Exception):
    """Wake an unfinished native capture when its world is closed."""
