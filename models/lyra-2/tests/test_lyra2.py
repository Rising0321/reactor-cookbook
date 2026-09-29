"""Drive the step boundary and native continuity without model weights."""

import asyncio
from dataclasses import FrozenInstanceError
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import AsyncMock, Mock

import numpy as np
import pytest
from reactor_runtime import ApplicationError

from lyra2 import Lyra2
from lyra2_camera import Lyra2CameraPlanner
from lyra2_model import Lyra2Input, Lyra2Model, NoAnchor
from lyra2_schema import ChunkCompleted, Lyra2State


def app():
    value = Lyra2()
    value.state = Lyra2State()
    value.send = AsyncMock()
    value.planner = Lyra2CameraPlanner(translation_per_frame=.0021875, rotation_degrees_per_frame=.125)
    return value


def test_refused_step_never_reaches_model():
    value = app()
    value.engine = Mock()
    with pytest.raises(ApplicationError):
        asyncio.run(value.process_input())
    value.engine.generate.assert_not_called()


def test_generate_uses_only_frozen_input():
    value = app()
    value.engine = Mock()
    step = Lyra2Input(1, None, "scene", 1, None, None)
    value.state = None
    assert value.generate(step) is value.engine.generate.return_value
    value.engine.generate.assert_called_once_with(step)
    with pytest.raises(FrozenInstanceError):
        step.seed = 2


def test_anchor_ack_and_ten_continuous_native_steps():
    value = app()
    value.image = Path("anchor.png")
    value.anchor = np.zeros((8, 8, 3), np.uint8)
    value.state.prompt = "scene"
    backend = Mock()
    backend.reset.return_value = (np.eye(4), np.eye(3))
    backend.generate_chunk.return_value = (np.zeros((80, 8, 8, 3), np.uint8), None)
    value.engine.backend = backend

    async def run():
        first = await value.process_input()
        assert first.anchor is value.anchor
        assert (await value.process_input()).anchor is value.anchor
        result = value.generate(first)
        assert await value.process_output(SimpleNamespace(result=result, error=None, elapsed=.5)) is None
        assert value.send.call_args.args[0].generating is False
        assert value.send.call_args.args[0].completed_chunks == 0
        for index in range(1, 11):
            step = await value.process_input()
            assert step.anchor is None
            result = value.generate(step)
            output = await value.process_output(SimpleNamespace(result=result, error=None, elapsed=1.25))
            assert output is not None and value.chunk == index
        backend.reset.assert_called_once()
        assert [call.kwargs["chunk"] for call in backend.generate_chunk.call_args_list] == list(range(1, 11))
        messages = [call.args[0] for call in value.send.call_args_list if isinstance(call.args[0], ChunkCompleted)]
        assert len(messages) == 10
        assert all(message.generation_seconds == 1.25 for message in messages)
        assert all(call.args[0].generating is False for call in value.send.call_args_list
                   if hasattr(call.args[0], "generating"))
    asyncio.run(run())


def test_failed_model_step_reaches_output_without_counting():
    value = app()
    value.engine.backend = Mock()
    step = Lyra2Input(1, None, "scene", 1, None, None)
    with pytest.raises(NoAnchor) as caught:
        value.generate(step)
    with pytest.raises(NoAnchor):
        asyncio.run(value.process_output(SimpleNamespace(result=None, error=caught.value, elapsed=.1)))
    assert value.chunk == value.engine.chunk == 0
    value.send.assert_not_called()


def test_backend_failure_does_not_commit_chunk():
    engine = Lyra2Model()
    engine.backend = Mock()
    engine.world_id = 1
    engine.backend.generate_chunk.side_effect = ValueError("failed")
    with pytest.raises(ValueError):
        engine.generate(Lyra2Input(1, None, "scene", 1, np.zeros((80, 4, 4)), np.zeros((80, 3, 3))))
    assert engine.chunk == 0
