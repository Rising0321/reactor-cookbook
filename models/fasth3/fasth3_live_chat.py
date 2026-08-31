"""Receive creative directions from a Bilibili live-room chat.

The listener only recognizes messages containing the configured command
prefix. It performs no model work itself: parsed requests are handed to the
model loop, which owns ordering, validation, and prompt generation.
"""

from __future__ import annotations

import re
from collections.abc import Callable
from dataclasses import dataclass


@dataclass(frozen=True, slots=True)
class LivePromptRequest:
    """One viewer request accepted from a live chat."""

    request: str
    viewer_name: str
    message_id: str


def parse_live_prompt(
    message: str,
    *,
    viewer_name: str,
    message_id: str,
    prefix: str,
    max_chars: int,
) -> LivePromptRequest | None:
    """Return a live prompt request when ``message`` uses ``prefix``.

    Matching is case-insensitive and accepts either an ASCII or full-width
    colon for the default ``!Prompt:`` command. The command may appear after
    provider-added text. Empty and oversized requests are ignored.

    Args:
        message: Raw live-chat message.
        viewer_name: Display name attached to the message.
        message_id: Provider identifier used for deduplication.
        prefix: Command prefix that opts a message into prompt generation.
        max_chars: Maximum accepted request length after the prefix.

    Returns:
        The parsed request, or ``None`` when the message is not eligible.
    """
    escaped = re.escape(prefix.rstrip(":："))
    match = re.search(rf"{escaped}\s*[:：]\s*(.*?)\s*$", message, re.IGNORECASE)
    if match is None:
        return None
    request = match.group(1).strip()
    if not request or len(request) > max_chars:
        return None
    return LivePromptRequest(
        request=request,
        viewer_name=viewer_name.strip() or "viewer",
        message_id=message_id,
    )


class BilibiliLiveChat:
    """Maintain one reconnecting Bilibili live-room WebSocket connection."""

    def __init__(
        self,
        *,
        room_id: int,
        prefix: str,
        max_request_chars: int,
        on_prompt: Callable[[LivePromptRequest], None],
        on_status: Callable[[bool, str | None], None],
    ) -> None:
        self.room_id = room_id
        self.prefix = prefix
        self.max_request_chars = max_request_chars
        self._on_prompt = on_prompt
        self._on_status = on_status

    async def run(self) -> None:
        """Listen until cancelled, reconnecting through ``blivedm`` as needed."""
        import blivedm
        import blivedm.models.web as web_models

        owner = self

        class Handler(blivedm.BaseHandler):
            """Forward only prompt commands into the Reactor model loop."""

            def _on_heartbeat(self, client, message) -> None:
                del client, message
                owner._on_status(True, None)

            def _on_danmaku(self, client, message: web_models.DanmakuMessage) -> None:
                del client
                request = parse_live_prompt(
                    message.msg,
                    viewer_name=message.uname,
                    message_id=f"{message.uid}:{message.rnd}:{message.timestamp}",
                    prefix=owner.prefix,
                    max_chars=owner.max_request_chars,
                )
                if request is not None:
                    owner._on_prompt(request)

            def on_client_stopped(self, client, exception) -> None:
                del client
                owner._on_status(False, str(exception) if exception else None)

        client = blivedm.BLiveClient(self.room_id, uid=0, heartbeat_interval=10)
        client.set_handler(Handler())
        client.start()
        try:
            await client.join()
        finally:
            await client.stop_and_close()
            self._on_status(False, None)


__all__ = ["BilibiliLiveChat", "LivePromptRequest", "parse_live_prompt"]
