"""Test LingBot-World v1 session and image-selection contracts."""

from __future__ import annotations

import asyncio
from dataclasses import FrozenInstanceError
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import numpy as np
import pytest
from reactor_runtime import ApplicationError, CommandError, StepOutcome, UploadedFile

import lingbot_world_v1
from lingbot_world_v1 import LingBotWorldV1
from lingbot_world_v1_camera import CameraMotionPlanner, MotionConfig
from lingbot_world_v1_types import (
    CameraMotionChanged,
    ImageSelected,
    LingBotWorldState,
    RolloutLimitReached,
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


class FakeBackend:
    """Record model calls and return native first/subsequent chunk shapes."""

    def __init__(self) -> None:
        self.resets: list[tuple[Any, ...]] = []
        self.calls: list[tuple[np.ndarray, str]] = []
        self.index = 0
        self.ended = False

    def reset(self, *args: Any) -> None:
        self.resets.append(args)
        self.index = 0

    def generate_chunk(self, camera: np.ndarray, prompt: str) -> np.ndarray:
        self.calls.append((camera.copy(), prompt))
        frames = np.zeros((9 if self.index == 0 else 12, 8, 8, 3), dtype=np.uint8)
        self.index += 1
        return frames

    def end_session(self) -> None:
        self.ended = True


def _loaded_world() -> tuple[Any, FakeBackend, list[Any]]:
    world, messages = _world()
    backend = FakeBackend()
    world._backend = backend
    world._planner = CameraMotionPlanner(MotionConfig(1.0, 8.0))
    world.on_session_started()
    asyncio.run(world.random_image())
    return world, backend, messages


async def _step(world: Any) -> Any:
    input = await world.process_input()
    result = world.generate(input)
    return await world.process_output(StepOutcome(result=result, elapsed=0.1))


def test_refused_step_does_not_touch_model() -> None:
    world, _ = _world()
    world.on_session_started()
    with pytest.raises(ApplicationError):
        asyncio.run(world.process_input())


def test_generate_reads_only_the_input_snapshot() -> None:
    world, backend, _ = _loaded_world()
    input = asyncio.run(world.process_input())
    with pytest.raises(FrozenInstanceError):
        input.prompt = "mutated"
    world.state = None
    result = world.generate(input)
    assert result.frames.shape == (9, 8, 8, 3)
    assert backend.calls[0][1] == input.prompt
    assert backend.resets[0] == (
        input.seed,
        input.anchor_image,
        input.intrinsics,
        input.prompt,
    )


def test_ten_continuous_actions_preserve_rollout() -> None:
    world, backend, messages = _loaded_world()

    async def run() -> None:
        for index in range(10):
            reply = await world.set_yaw(0.25 if index % 2 == 0 else -0.25)
            assert reply.applies_to_chunk == index + 1
            output = await _step(world)
            assert output.main_video.shape[0] == (9 if index == 0 else 12)
            assert world._chunk_index == index + 1
        await world.set_prompt("The same lake at sunset")
        await _step(world)

    asyncio.run(run())
    assert len(backend.resets) == 1
    assert len(backend.calls) == 11
    assert backend.calls[-1][1] == "The same lake at sunset"
    assert messages[-1].next_chunk == 12
    assert "fps" not in vars(LingBotWorldV1)
    assert "inference" not in vars(LingBotWorldV1)


def test_reset_restarts_at_first_native_chunk() -> None:
    world, backend, _ = _loaded_world()
    asyncio.run(_step(world))
    asyncio.run(world.set_forward(1.0))
    asyncio.run(world.reset(123))
    output = asyncio.run(_step(world))
    assert output.main_video.shape[0] == 9
    assert len(backend.resets) == 2
    assert backend.resets[-1][0] == 123
    assert world._chunk_index == 1
    assert world.state.forward == 0
    np.testing.assert_array_equal(backend.calls[-1][0][0], np.eye(4))


def test_limit_emits_last_chunk_then_refuses_until_reset() -> None:
    world, backend, messages = _loaded_world()
    world._config.max_chunks = 2
    asyncio.run(_step(world))
    output = asyncio.run(_step(world))
    assert output.main_video.shape[0] == 12
    assert sum(isinstance(message, RolloutLimitReached) for message in messages) == 1
    with pytest.raises(ApplicationError):
        asyncio.run(world.process_input())
    with pytest.raises(CommandError):
        asyncio.run(world.set_prompt("new prompt"))
    assert len(backend.calls) == 2
    asyncio.run(world.reset(-1))
    assert asyncio.run(_step(world)).main_video.shape[0] == 9


def test_model_failure_propagates_without_claiming_a_completed_chunk() -> None:
    world, _, messages = _loaded_world()
    count = len(messages)
    error = RuntimeError("worker failed")
    with pytest.raises(RuntimeError, match="worker failed"):
        asyncio.run(world.process_output(StepOutcome(error=error, elapsed=0.1)))
    assert world._chunk_index == 0
    assert world.state._restart_requested
    assert len(messages) == count


def test_session_end_releases_rollout_and_clears_selection() -> None:
    world, backend, _ = _loaded_world()
    asyncio.run(_step(world))
    world.on_session_ended()
    assert backend.ended
    assert world._selected_input is None
    assert world._chunk_index == 0
