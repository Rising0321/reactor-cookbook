"""Test LingBot-World v1 session and image-selection contracts."""

from __future__ import annotations

import importlib
import sys
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pytest
from reactor_runtime import CommandError, UploadedFile

MODEL_DIR = Path(__file__).parents[1]
sys.path.insert(0, str(MODEL_DIR))

lingbot_schema = importlib.import_module("lingbot_schema")
lingbot_world_v1 = importlib.import_module("lingbot_world_v1")


def _world() -> Any:
    sample = SimpleNamespace(
        image=Path("sample.jpg"),
        intrinsics=Path("intrinsics.npy"),
        prompt="A calm lakeside world",
    )
    config: Any = SimpleNamespace(seed=42, samples=(sample,), max_chunks=320)
    world = lingbot_world_v1.LingBotWorldV1()
    world.state = lingbot_schema.LingBotWorldState()
    world._config = config
    world._default_prompt = sample.prompt
    return world


def test_session_waits_for_an_explicit_image_selection() -> None:
    """Expose an empty idle world until upload or random selection succeeds."""
    world = _world()

    world.on_session_started()
    state = world._state_update()

    assert world._selected_input is None
    assert world._selected_intrinsics is None
    assert state.prompt == ""
    assert state.image_source == "none"
    assert state.image_name == ""
    assert state.next_chunk is None
    assert state.next_chunk_frames is None
    assert state.paused is False
    assert state.step_queued is False


def test_first_upload_uses_the_default_public_calibration(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Allow upload to initialize a world before any built-in image is selected."""
    world = _world()
    world.on_session_started()
    monkeypatch.setattr(
        lingbot_world_v1, "validate_uploaded_image", lambda _image: None
    )
    upload = UploadedFile(name="anchor.png", mime_type="image/png", data=b"image")

    state = world.set_image(upload, "")

    assert world._selected_input is upload
    assert world._selected_intrinsics == Path("intrinsics.npy")
    assert state.prompt == "A calm lakeside world"
    assert state.image_source == "upload"
    assert state.image_name == "anchor.png"
    assert state.next_chunk == 1
    assert state.next_chunk_frames == 9


def test_camera_controls_require_an_image() -> None:
    """Reject motion that has no selected world to control."""
    world = _world()
    world.on_session_started()

    with pytest.raises(CommandError):
        world.set_forward(1.0)
