"""Verify native Runtime 3.5 chunk boundaries without loading model weights."""

import asyncio
from dataclasses import FrozenInstanceError
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
from reactor_runtime import ApplicationError, ReactorApp, StepOutcome

from alayaworld import AlayaStepState, AlayaWorld
from alayaworld_types import AlayaWorldState


def test_step_waits_for_image() -> None:
    model = AlayaWorld()
    model.state = AlayaWorldState()
    assert isinstance(model, ReactorApp)
    with pytest.raises(ApplicationError):
        asyncio.run(model.process_input())


def test_ten_steps_snapshot_controls_and_keep_rollout(monkeypatch) -> None:
    model = AlayaWorld()
    model.state = AlayaWorldState()
    model._selected_input = Path("anchor.png")
    model.state.prompt = "A stable landscape"
    model.state._reset_requested = True
    model._config = SimpleNamespace(max_chunks=512, max_chunks_per_rollout=512)
    resets = []
    controls = []

    async def discard(*_):
        pass

    def reset(*args):
        resets.append(args)
        model._ar_index = 0

    def generate(*args):
        controls.append(args)
        model._ar_index += 1
        return np.zeros((12, 8, 8, 3), dtype=np.uint8)

    monkeypatch.setattr(model, "_send_state_update", discard)
    monkeypatch.setattr(model, "send", discard)
    monkeypatch.setattr(model, "_reset_rollout", reset)
    monkeypatch.setattr(model, "_generate_chunk", generate)

    async def run():
        for index in range(10):
            model.state.prompt = f"Scene {index}"
            model.state.strafe = index / 10
            snapshot = await model.process_input()
            assert isinstance(snapshot, AlayaStepState)
            with pytest.raises(FrozenInstanceError):
                snapshot.prompt = "mutated"
            model.state.prompt = "later input"
            model.state.strafe = 0
            result = model.generate(snapshot)
            output = await model.process_output(StepOutcome(result=result))
            assert output.main_video.shape == (12, 8, 8, 3)
            assert controls[-1][0] == f"Scene {index}"
            assert controls[-1][1] == index / 10
        assert model._ar_index == 10

    asyncio.run(run())
    assert len(resets) == 1
    assert len(controls) == 10


def test_generation_error_releases_busy_flags() -> None:
    model = AlayaWorld()
    model.state = AlayaWorldState()
    model._reset_in_flight = True
    model._chunk_in_flight = True
    with pytest.raises(RuntimeError, match="native failure"):
        asyncio.run(
            model.process_output(StepOutcome(error=RuntimeError("native failure")))
        )
    assert not model._reset_in_flight
    assert not model._chunk_in_flight


def test_manual_reset_flushes_pending_media(monkeypatch) -> None:
    model = AlayaWorld()
    model.state = AlayaWorldState()
    model._selected_input = Path("anchor.png")
    flushes = []

    async def discard():
        pass

    monkeypatch.setattr(model, "_send_state_update", discard)
    monkeypatch.setattr(model.output, "flush", lambda: flushes.append(True))
    asyncio.run(model.reset(-1))
    assert flushes == [True]
    assert model.state._reset_requested
