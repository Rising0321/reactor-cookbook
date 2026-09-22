"""Verify the native Runtime step boundary without loading model weights."""

import asyncio
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import AsyncMock, Mock

import numpy as np
import pytest
from reactor_runtime import ApplicationError, StepOutcome

from lingbot_world_v2 import LingBotWorldV2
from lingbot_world_v2_types import LingBotWorldV2State


def make_model():
    model = LingBotWorldV2()
    model.state = LingBotWorldV2State()
    model._config = SimpleNamespace(
        seed=42,
        max_chunks=20,
        chunk_latents=4,
        upload_intrinsics=(1, 1, 0.5, 0.5),
    )
    model.send = AsyncMock()
    model._selected_input = Path("anchor.png")
    model.state._reset_requested = False
    return model


def test_input_waits_for_image_and_rejects_rollout_limit():
    model = make_model()
    model._selected_input = None
    with pytest.raises(ApplicationError):
        asyncio.run(model.process_input())
    model._selected_input = Path("anchor.png")
    model._limit_reached = True
    with pytest.raises(ApplicationError):
        asyncio.run(model.process_input())


def test_failure_does_not_advance_progress():
    model = make_model()
    model.state._reset_requested = True
    with pytest.raises(RuntimeError, match="failed"):
        asyncio.run(model.process_output(StepOutcome(error=RuntimeError("failed"))))
    assert model._chunk_index == 0
    assert model.state._reset_requested


def test_controls_are_snapshotted_and_native_chunks_remain_continuous():
    model = make_model()
    backend = Mock()
    model._backend = backend
    model._planner = Mock()
    frames = np.zeros((13, 8, 8, 3), dtype=np.uint8)
    backend.generate_chunk.return_value = frames
    model.state._yaw = 0.25
    snapshot = asyncio.run(model.process_input())
    model.state._yaw = -0.25
    assert snapshot.yaw == 0.25
    result = model.generate(snapshot)
    asyncio.run(model.process_output(StepOutcome(result=result)))
    assert model._chunk_index == 1
    backend.reset.assert_not_called()
    backend.generate_chunk.assert_called_once()
    assert model.send.called
    backend.generate_chunk.return_value = np.zeros((16, 8, 8, 3), dtype=np.uint8)
    for _ in range(9):
        snapshot = asyncio.run(model.process_input())
        result = model.generate(snapshot)
        asyncio.run(model.process_output(StepOutcome(result=result)))
    assert model._chunk_index == 10
    assert backend.generate_chunk.call_count == 10
    backend.reset.assert_not_called()
