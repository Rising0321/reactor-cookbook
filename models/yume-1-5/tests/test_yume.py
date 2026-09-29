"""Contract and rollout-boundary tests for the YUME-1.5 adapter."""

from __future__ import annotations

import asyncio
import io
from dataclasses import FrozenInstanceError
from pathlib import Path
from types import SimpleNamespace
from typing import Any, cast
from unittest.mock import AsyncMock, Mock

import numpy as np
import pytest
from PIL import Image
from reactor_runtime import ApplicationError, StepOutcome, UploadedFile
from reactor_runtime.interface.model.contract import ModelContract
from yume import Yume15
from yume_assets import read_config
from yume_backend import conditioned_prompt
from yume_images import validate_image
from yume_model import NoAnchor, YumeInput
from yume_types import YumeOutput, YumeState


def uploaded_image() -> UploadedFile:
    stream = io.BytesIO()
    Image.new("RGB", (32, 32), (20, 30, 40)).save(stream, format="PNG")
    return UploadedFile(name="start.png", mime_type="image/png", data=stream.getvalue())


def test_contract_has_only_real_upstream_controls() -> None:
    contract = ModelContract.of(Yume15)
    assert set(contract.commands) == {
        "release_controls",
        "reset",
        "set_key_state",
        "set_image",
        "set_prompt",
        "set_text_scene",
        "set_video_scene",
    }
    assert "fps" not in Yume15.__dict__
    assert Yume15.buffer_size == 29
    assert set(YumeOutput.__tracks__) == {"main_video"}


def test_native_fast_context_is_fixed() -> None:
    config = read_config(Path(__file__).parents[1] / "yume.yaml")
    assert config.frames_per_chunk == 32
    assert config.latent_frames_per_chunk == 8
    assert config.sample_steps == 4


def test_conditioning_explicitly_distinguishes_stationary_and_motion() -> None:
    stationary = conditioned_prompt("A city street", "none", "none")
    moving = conditioned_prompt("A city street", "forward", "pan_left")

    assert "Actual distance moved:0" in stationary
    assert "Angular change rate (turn speed):0" in stationary
    assert "View rotation speed:0" in stationary
    assert "pushes forward" not in stationary
    assert "Actual distance moved:4" in moving
    assert "Angular change rate (turn speed):4" in moving
    assert "View rotation speed:4" in moving


def test_upload_is_decodable() -> None:
    validate_image(uploaded_image())


def test_blank_image_prompt_uses_neutral_configured_prompt() -> None:
    model = Yume15()
    model.state = YumeState()
    model._config = read_config(Path(__file__).parents[1] / "yume.yaml")
    model.output = cast(Any, type("Output", (), {"flush": lambda self: None})())

    message = asyncio.run(model.set_image(uploaded_image(), "   ", 42))

    assert model.state.prompt == model._config.default_upload_prompt
    assert message.prompt == model._config.default_upload_prompt


def test_one_turn_is_one_chunk_and_prompt_can_change_without_reset(
    tmp_path: Path,
) -> None:
    class FakeBackend:
        resets = 0
        calls = 0

        def reset(self, **_: object) -> None:
            self.resets += 1

        def generate_chunk(
            self, *, prompt: str, movement: str, view: str
        ) -> tuple[np.ndarray, str]:
            self.calls += 1
            return np.zeros(
                (29, 8, 8, 3), dtype=np.uint8
            ), f"{movement}|{view}|{prompt}"

        def end_session(self) -> None:
            return

    model = Yume15()
    model.state = YumeState()
    model._config = read_config(Path(__file__).parents[1] / "yume.yaml")
    backend = FakeBackend()
    model._engine.backend = backend
    model._engine.config = SimpleNamespace(runtime_dir=tmp_path)
    model._seed = 42
    asyncio.run(model.set_image(uploaded_image(), "A forest trail", 42))

    async def generate() -> tuple[np.ndarray, np.ndarray]:
        first = await model.process_output(
            StepOutcome(
                result=model.generate(await model.process_input()), elapsed=1.25
            )
        )
        assert isinstance(first, YumeOutput)
        await model.set_prompt("Rain begins")
        second = await model.process_output(
            StepOutcome(
                result=model.generate(await model.process_input()), elapsed=1.25
            )
        )
        assert isinstance(second, YumeOutput)
        return cast(np.ndarray, first.main_video), cast(np.ndarray, second.main_video)

    first, second = asyncio.run(generate())
    assert first.shape == second.shape == (29, 8, 8, 3)
    assert backend.resets == 1
    assert backend.calls == 2


def test_reset_flushes_media() -> None:
    class FakeOutput:
        flushes = 0

        def flush(self) -> None:
            self.flushes += 1

    model = Yume15()
    model.state = YumeState()
    output = FakeOutput()
    model.output = cast(Any, output)
    model._request_reset()
    assert output.flushes == 1
    assert model._world_id != model._applied_world_id
    assert model.state._pressed_keys == frozenset()


def test_held_keys_combine_and_release_independently() -> None:
    model = Yume15()
    model.state = YumeState()
    model._mode = "text_to_video"
    asyncio.run(model.set_key_state("w", True))
    asyncio.run(model.set_key_state("a", True))
    asyncio.run(model.set_key_state("arrow_up", True))
    assert model._resolve_controls(model.state._pressed_keys) == (
        "forward_left",
        "tilt_up",
    )
    asyncio.run(model.set_key_state("a", False))
    assert model._resolve_controls(model.state._pressed_keys) == ("forward", "tilt_up")


def test_refusal_never_calls_model() -> None:
    app = Yume15()
    app.state = YumeState()
    app._engine = Mock()
    with pytest.raises(ApplicationError):
        asyncio.run(app.process_input())
    app._engine.generate.assert_not_called()


def test_generate_forwards_frozen_input_without_reading_state() -> None:
    app = Yume15()
    value = YumeInput(1, None, "forest", "none", "none")
    with pytest.raises(FrozenInstanceError):
        value.prompt = "changed"
    result = object()
    app._engine = SimpleNamespace(
        generate=lambda actual: result if actual is value else None
    )
    app.state = None
    assert app.generate(value) is result


def test_ten_continuous_hook_steps_send_anchor_once() -> None:
    app = Yume15()
    app.state = YumeState()
    app.send = AsyncMock()
    anchors = []

    class Backend:
        def reset(self, **kwargs):
            anchors.append(kwargs)

        def generate_chunk(self, **kwargs):
            return np.zeros((29, 8, 8, 3), np.uint8), kwargs["prompt"]

        def end_session(self):
            pass

    app._engine.backend = Backend()

    async def drive():
        await app.set_text_scene("forest", 42)
        for index in range(1, 11):
            value = await app.process_input()
            assert (value.anchor is not None) == (index == 1)
            result = app.generate(value)
            assert result.chunk_index == index
            await app.process_output(StepOutcome(result=result, elapsed=0.5))
        assert len(anchors) == 1
        states = [c.args[0] for c in app.send.call_args_list if type(c.args[0]).__name__ == "StateUpdate"]
        assert all(not state.generating for state in states)
        await app.reset(43)
        assert (await app.process_input()).anchor is not None

    asyncio.run(drive())


def test_model_error_reaches_output_without_counting_step() -> None:
    app = Yume15()
    app.state = YumeState()
    with pytest.raises(NoAnchor) as caught:
        app.generate(YumeInput(1, None, "forest", "none", "none"))
    with pytest.raises(NoAnchor):
        asyncio.run(app.process_output(StepOutcome(error=caught.value)))
    assert app._chunk_index == app._engine.chunk_index == 0
