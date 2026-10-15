"""Application-side native simulation, deterministic capture and event prompts."""

from collections.abc import Mapping
from pathlib import Path
from threading import Condition

import numpy as np


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
    return tuple(
        dict.fromkeys(text for _, text in sorted(indexed, key=lambda pair: pair[0]))
    )


class EdenGame:
    """One bounded native capture, with a latest-held-state mailbox per frame."""

    def __init__(self, *, source: Path, slug: str) -> None:
        from eden.apps.game2v.game_renderer import GameCaptureWorker
        from eden.apps.game2v.game_timeline import GameTimeline

        self.timeline = GameTimeline(slug)
        self.capture = GameCaptureWorker(
            slug, width=1280, height=704, games_root=source / "realtime/games"
        )
        self._held = set()
        self._latest = frozenset()
        self._gate = Condition()
        self._paused = False
        self._closed = False
        self._pending = None
        self._sampled = []

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
        events = [
            {"action": key, "phase": "up"} for key in sorted(self._held - chosen)
        ] + [{"action": key, "phase": "down"} for key in sorted(chosen - self._held)]
        self._held = set(chosen)
        self._sampled.append(tuple(sorted(chosen)))
        return events

    def prepare(self, frames: int) -> None:
        """Keep at most one capture in flight, including a completed one."""
        if self._pending is None:
            self._sampled = []
            packet = self.timeline.begin_chunk(frames, 24)
            self._pending = self.capture.submit(packet, drain_events=self._drain)

    def ready(self) -> bool:
        return self._pending is not None and self._pending.done()

    def collect(self):
        """Collect once; every captured frame remains paired with its inference."""
        if not self.ready():
            raise RuntimeError("Capture is not ready")
        chunk = self._pending.result()
        self._pending = None
        proxy = np.stack([frame.ui_frame for frame in chunk.frames])
        condition = np.stack(
            [frame.condition_frame for frame in chunk.frames if frame.capture_condition]
        )
        sampled = tuple(self._sampled)
        if len(sampled) != len(chunk.frames):
            raise RuntimeError("Input samples do not match captured frames")
        return proxy, condition, chunk, sampled

    def close(self):
        with self._gate:
            self._closed = True
            self._gate.notify_all()
        self.capture.close()
        self._pending = None


class CaptureClosed(Exception):
    """Wake an unfinished native capture when its world is closed."""
