"""Test LingBot-World v1 session and image-selection contracts."""

from __future__ import annotations

import asyncio
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pytest
from reactor_runtime import CommandError, UploadedFile

import lingbot_world_v1
from lingbot_world_v1 import LingBotWorldV1
from lingbot_world_v1_types import (
    CameraMotionChanged,
    ImageSelected,
    LingBotWorldState,
    StateUpdate,
)


def _world() -> tuple[Any, list[Any]]:
    sample = SimpleNamespace(
        image=Path("sample.jpg"),
        intrinsics=Path("intrinsics.npy"),
        prompt="A calm lakeside world",
    )
    config: Any = SimpleNamespace(seed=42, samples=(sample,), max_chunks=320)
    world = LingBotWorldV1()
    world.state = LingBotWorldState()
    world._config = config
    world._default_prompt = sample.prompt
    messages: list[Any] = []

    async def record(message: Any) -> None:
        messages.append(message)

    world.send = record
    return world, messages


def test_session_waits_for_an_explicit_image_selection() -> None:
    """Expose an empty idle world until upload or random selection succeeds."""
    world, _ = _world()

    world.on_session_started()
    state = world._state_update()

    assert world._selected_input is None
    assert world._selected_intrinsics is None
    assert state.prompt == ""
    assert state.image_source == "none"
    assert state.image_name == ""
    assert state.next_chunk is None
    assert state.next_chunk_frames is None


def test_first_upload_uses_the_default_public_calibration(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Allow upload to initialize a world before any built-in image is selected."""
    world, messages = _world()
    world.on_session_started()
    monkeypatch.setattr(
        lingbot_world_v1, "validate_uploaded_image", lambda _image: None
    )
    upload = UploadedFile(name="anchor.png", mime_type="image/png", data=b"image")

    reply = asyncio.run(world.set_image(upload, ""))

    assert isinstance(reply, ImageSelected)
    assert reply.source == "uploaded"
    assert reply.filename == "anchor.png"
    assert reply.prompt == "A calm lakeside world"
    assert world._selected_input is upload
    assert world._selected_intrinsics == Path("intrinsics.npy")
    state = messages[-1]
    assert isinstance(state, StateUpdate)
    assert state.image_source == "uploaded"
    assert state.image_name == "anchor.png"
    assert state.next_chunk == 1
    assert state.next_chunk_frames == 9


def test_camera_change_replies_and_broadcasts_complete_state(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Confirm a control change and broadcast the resulting durable state."""
    world, messages = _world()
    world.on_session_started()
    monkeypatch.setattr(
        lingbot_world_v1, "validate_uploaded_image", lambda _image: None
    )
    upload = UploadedFile(name="anchor.png", mime_type="image/png", data=b"image")
    asyncio.run(world.set_image(upload, ""))

    reply = asyncio.run(world.set_yaw(0.75))

    assert isinstance(reply, CameraMotionChanged)
    assert reply.yaw == 0.75
    assert reply.applies_to_chunk == 1
    assert isinstance(messages[-1], StateUpdate)
    assert messages[-1].yaw == 0.75


def test_camera_controls_require_an_image() -> None:
    """Reject motion that has no selected world to control."""
    world, _ = _world()
    world.on_session_started()

    with pytest.raises(CommandError):
        asyncio.run(world.set_forward(1.0))


@pytest.mark.parametrize(
    "axis", ["forward", "strafe", "vertical", "pitch", "yaw", "roll"]
)
def test_exhausted_world_accepts_only_neutral_camera_release(axis: str) -> None:
    """A final-chunk release is safe and does not invent a future output."""
    world, messages = _world()
    world.on_session_started()
    world._selected_input = Path("anchor.jpg")
    world._chunk_index = 320
    world.state._restart_requested = False
    world.state._limit_reached = True
    command = getattr(world, f"set_{axis}")
    response = asyncio.run(command(0.0))
    assert response.applies_to_chunk is None
    assert messages[-1].next_chunk is None
    assert messages[-1].completed_chunks == 320
    assert world.state._limit_reached is True
    assert world.state._restart_requested is False
    with pytest.raises(CommandError) as error:
        asyncio.run(command(0.05))
    assert error.value.code == "rollout_limit_reached"
