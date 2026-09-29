"""Contract and upstream-fidelity tests for the Zing adapter."""

from __future__ import annotations

import asyncio
import os
from dataclasses import FrozenInstanceError, replace
from unittest.mock import Mock
from pathlib import Path
from typing import Any, cast

import numpy as np
import pytest
from types import SimpleNamespace
from reactor_runtime import ApplicationError
from PIL import Image
from reactor_runtime import UploadedFile
from reactor_runtime.interface.model.contract import ModelContract

from zing import Zing
from zing_assets import activate_source, read_config
from zing_images import validate_image
from zing_types import ZingOutput, ZingState


def test_contract_covers_text_image_and_all_native_controls() -> None:
    contract = ModelContract.of(Zing)
    assert set(contract.commands) == {
        "example_image", "release_controls", "reset", "set_image", "set_key", "set_prompt"
    }
    assert "fps" not in Zing.__dict__
    assert Zing.buffer_size == 16
    assert set(ZingOutput.__tracks__) == {"main_video"}


def test_released_cache_and_chunk_geometry_are_preserved() -> None:
    config = read_config(Path(__file__).parents[1] / "zing.yaml", Path("/tmp/zing-test"))
    source_override = os.environ.get("ZING_TEST_SOURCE_PATH")
    if source_override:
        config = replace(config, source_path=Path(source_override))
    activate_source(config)
    load_config = pytest.importorskip("zing_v0_5.config").load_config
    upstream = load_config(config.source_path / "config" / "zing.yaml")
    assert upstream.generator.local_attn_size == config.local_attn_size == 97
    assert upstream.generator.sink_size == config.sink_size == 9
    assert upstream.inference.frames_per_block == 4
    assert upstream.vae.temporal_scale == 4


def test_all_eight_keys_can_be_held_and_released() -> None:
    model = Zing()
    model.state = ZingState()
    model.state.prompt = "world"
    model._completed_chunks = 2
    async def mutate() -> None:
        for key in ("w", "a", "s", "d", "i", "j", "k", "l"):
            await model.set_key(cast(Any, key), True)
        assert model.state._pressed_keys == frozenset("wasdijkl")
        result = await model.release_controls()
        assert result.released_keys == sorted("wasdijkl")
    asyncio.run(mutate())


def test_prompt_switch_does_not_reset_an_active_rollout() -> None:
    model = Zing()
    model.state = ZingState()
    model.state.prompt = "first"
    model._applied_world_id = model._world_epoch
    model._completed_chunks = 3
    model._active_prompt = "first"
    message = asyncio.run(model.set_prompt("second"))
    assert message.applies_to_chunk == 4
    assert message.resets_rollout is False
    assert model._applied_world_id == model._world_epoch


def test_image_upload_is_real_and_moderated(tmp_path: Path) -> None:
    path = tmp_path / "frame.png"
    Image.new("RGB", (64, 32), (20, 40, 60)).save(path)
    validate_image(UploadedFile(name=path.name, mime_type="image/png", data=path.read_bytes()))


def test_new_session_waits_for_user_input() -> None:
    class IdleBackend:
        def reset(self, **_: object) -> None:
            raise AssertionError("idle session must not reset the backend")
        def generate_chunk(self, **_: object) -> np.ndarray:
            raise AssertionError("idle session must not generate")
        def cache_frames(self) -> int: return 0
        def end_session(self) -> None: pass
    model = Zing()
    model.state = ZingState()
    model._config = read_config(Path(__file__).parents[1] / "zing.yaml", Path("/tmp/zing-test"))
    model.engine.backend = IdleBackend()
    model.on_session_started()
    assert model.state.prompt == ""
    assert not model._state_update().reset_queued
    assert model._conditioning == "none"
    with pytest.raises(ApplicationError):
        asyncio.run(model.process_input())


def test_one_backend_call_maps_to_one_reactor_output() -> None:
    class FakeBackend:
        def __init__(self) -> None:
            self.calls = 0
        def reset(self, **_: object) -> None: pass
        def generate_chunk(self, **_: object) -> np.ndarray:
            self.calls += 1
            return np.zeros((16, 704, 1248, 3), dtype=np.uint8)
        def cache_frames(self) -> int: return 4 * self.calls
        def end_session(self) -> None: pass
    model = Zing()
    model.state = ZingState()
    model.state.prompt = "A traversable courtyard"
    model._conditioning = "text"
    model._config = read_config(Path(__file__).parents[1] / "zing.yaml", Path("/tmp/zing-test"))
    backend = FakeBackend()
    model.engine.backend = backend
    model.engine.config = model._config
    async def run() -> None:
        result = model.generate(await model.process_input())
        output = await model.process_output(SimpleNamespace(result=result, error=None, elapsed=.25))
        assert cast(np.ndarray, output.main_video).shape == (16, 704, 1248, 3)
    asyncio.run(run())
    assert backend.calls == 1


def test_ten_chunks_preserve_world_anchor_and_native_cache(tmp_path):
    class Backend:
        calls = 0
        resets = 0
        def reset(self, **kwargs):
            assert kwargs["image"] is not None
            self.resets += 1
        def generate_chunk(self, **kwargs):
            self.calls += 1
            return np.zeros((16, 8, 8, 3), dtype=np.uint8)
        def cache_frames(self):
            return self.calls * 4
    model = Zing()
    model.state = ZingState()
    model._conditioning = "uploaded"
    model._image = np.zeros((8, 8, 3), dtype=np.uint8)
    model._config = read_config(Path(__file__).parents[1] / "zing.yaml", tmp_path)
    model.engine.config = model._config
    model.engine.backend = backend = Backend()
    messages = []
    async def send(message):
        messages.append(message)
    model.send = send
    async def run():
        for index in range(1, 11):
            input = await model.process_input()
            assert (input.image is not None) == (index == 1)
            result = model.generate(input)
            assert (result.index, result.cache_frames) == (index, index * 4)
            await model.process_output(SimpleNamespace(result=result, error=None, elapsed=.25))
    asyncio.run(run())
    assert (backend.resets, backend.calls) == (1, 10)
    assert model._completed_chunks == 10
    assert all(m.generation_seconds == .25 for m in messages if type(m).__name__ == "ChunkCompleted")
    states = [m for m in messages if type(m).__name__ == "StateUpdate"]
    assert len(states) == 10
    assert all(not m.generating for m in states)
    assert [m.completed_chunks for m in states] == list(range(1, 11))


def test_generate_reads_only_input_and_failure_never_completes():
    model = Zing()
    marker = object()
    model.engine = SimpleNamespace(generate=lambda input: input)
    model.state = None
    assert model.generate(marker) is marker
    model.state = ZingState()
    def fail(input):
        raise ValueError("native failure")
    model.engine = SimpleNamespace(generate=fail)
    with pytest.raises(ValueError) as failure:
        model.generate(marker)
    with pytest.raises(ValueError, match="native failure"):
        asyncio.run(model.process_output(SimpleNamespace(error=failure.value)))
    assert model._completed_chunks == 0
    assert model._applied_world_id is None


def test_refusal_and_frozen_unacknowledged_anchor():
    model = Zing()
    model.state = ZingState()
    model.engine = Mock()
    with pytest.raises(ApplicationError):
        asyncio.run(model.process_input())
    model.engine.generate.assert_not_called()
    model._conditioning = "uploaded"
    model._image = np.zeros((8, 8, 3), np.uint8)
    step = asyncio.run(model.process_input())
    with pytest.raises(FrozenInstanceError):
        step.world_id = 99
    assert asyncio.run(model.process_input()).image is step.image


def test_native_failure_preserves_completed_index():
    model = Zing()
    model.state = ZingState()
    model._conditioning = "text"
    model.engine.config = SimpleNamespace(max_chunks=32)
    model.engine.backend = backend = Mock()
    backend.generate_chunk.return_value = np.zeros((16, 8, 8, 3), np.uint8)
    backend.cache_frames.return_value = 5
    result = model.generate(asyncio.run(model.process_input()))
    asyncio.run(model.process_output(SimpleNamespace(result=result, error=None, elapsed=.1)))
    assert model._completed_chunks == model.engine.index == 1
    backend.generate_chunk.side_effect = RuntimeError("native failure")
    model.send = Mock()
    with pytest.raises(RuntimeError) as error:
        model.generate(asyncio.run(model.process_input()))
    with pytest.raises(RuntimeError, match="native failure"):
        asyncio.run(model.process_output(SimpleNamespace(error=error.value)))
    assert model._completed_chunks == model.engine.index == 1
    model.send.assert_not_called()
