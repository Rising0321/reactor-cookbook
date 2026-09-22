"""Test native step boundaries without loading GPU weights."""

import asyncio
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
from reactor_runtime import ApplicationError, StepOutcome

from sana_wm import SanaWM
from sana_wm_backend import TrajectoryCompleteError
from sana_wm_types import SanaWMState


def test_image_and_calibration_uploads_are_separate(monkeypatch):
    from reactor_runtime.interface.model.contract import ModelContract

    import sana_wm

    contract = ModelContract.of(SanaWM)
    assert "set_intrinsics" in contract.commands
    import inspect

    assert "intrinsics" not in inspect.signature(SanaWM.set_image).parameters
    model, _ = ready()
    monkeypatch.setattr(sana_wm, "_validate_image", lambda value: None)
    monkeypatch.setattr(sana_wm, "_validate_intrinsics", lambda value: None)
    calibration = object()
    image = SimpleNamespace(name="uploaded.png")

    async def run():
        await model.set_intrinsics(calibration)
        model.state.prompt = "Cars from the previous sample"
        await model.set_image(image, prompt="")
        assert model._intrinsics_input is calibration
        assert model._pending_intrinsics is None
        assert "Cars" not in model.state.prompt
        await model.set_image(image, prompt="A lake")
        assert model._intrinsics_input is None
        assert model.state.prompt == "A lake"

    asyncio.run(run())


class Backend:
    def __init__(self):
        self.chunk_index = 0
        self.trajectory_frames = None
        self.resets = []
        self.controls = []
        self.exhausted = False

    def reset(self, image, prompt, seed, **kwargs):
        self.resets.append((image, prompt, seed, kwargs))
        self.chunk_index = 0

    def generate_chunk(self, controls):
        if self.exhausted:
            raise TrajectoryCompleteError()
        self.controls.append(controls)
        self.chunk_index += 1
        return np.zeros((24, 8, 8, 3), dtype=np.uint8)


def ready():
    model = SanaWM()
    model.state = SanaWMState()
    model._config = SimpleNamespace(max_chunks=512)
    model._backend = Backend()
    model.on_session_started()
    messages = []

    async def send(message):
        messages.append(message)

    model.send = send
    return model, messages


def test_waits_for_explicit_image():
    model, _ = ready()
    with pytest.raises(ApplicationError):
        asyncio.run(model.process_input())


def test_snapshot_and_continuous_native_chunks():
    model, _ = ready()
    model._selected_image = Path("anchor.png")
    model.state.prompt = "A lake"
    model.state._reset_requested = True
    model.state._held_controls = {"forward"}

    async def run():
        for index in range(10):
            snapshot = await model.process_input()
            model.state._held_controls = {"yaw_left"}
            result = model.generate(snapshot)
            output = await model.process_output(StepOutcome(result=result))
            assert output.main_video.shape == (24, 8, 8, 3)
            assert model._chunk_index == index + 1
        assert len(model._backend.resets) == 1
        assert model._backend.controls[0] == {"forward"}
        assert model._backend.controls[1] == {"yaw_left"}

    asyncio.run(run())


def test_trajectory_completion_and_error_cleanup():
    model, messages = ready()
    model._selected_image = Path("anchor.png")
    model._backend.exhausted = True

    async def run():
        snapshot = await model.process_input()
        output = await model.process_output(
            StepOutcome(result=model.generate(snapshot))
        )
        assert output is None
        assert model.state._trajectory_exhausted
        assert not model._generating and not model._chunk_in_flight
        with pytest.raises(ApplicationError):
            await model.process_input()
        with pytest.raises(RuntimeError, match="failure"):
            await model.process_output(StepOutcome(error=RuntimeError("failure")))
        assert not model._generating and not model._chunk_in_flight

    asyncio.run(run())
    assert any(type(message).__name__ == "TrajectoryExhausted" for message in messages)


def test_automatic_restart_commits_effects_in_output(monkeypatch):
    model, messages = ready()
    model._selected_image = Path("anchor.png")
    model._chunk_index = 512
    model.state._held_controls = {"forward"}
    flushes = []
    monkeypatch.setattr(model.output, "flush", lambda: flushes.append(None))

    async def run():
        snapshot = await model.process_input()
        assert not messages and not flushes
        assert snapshot.restart and snapshot.automatic_restart
        assert not snapshot.controls
        await model.process_output(StepOutcome(result=model.generate(snapshot)))
        assert model._chunk_index == 1
        assert not model.state._held_controls
        assert flushes == [None]

    asyncio.run(run())
