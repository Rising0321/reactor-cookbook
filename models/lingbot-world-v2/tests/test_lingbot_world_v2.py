"""Test image-gated inference and safe camera release after rollout exhaustion."""

import asyncio
from types import SimpleNamespace

import pytest
from lingbot_world_v2 import LingBotWorldV2
from lingbot_world_v2_types import LingBotWorldV2State
from reactor_runtime import CommandError, UploadedFile
from reactor_runtime.interface.model.contract import ModelContract


def _model():
    """Construct a weight-free loaded model with captured state broadcasts."""
    model = LingBotWorldV2()
    model.state = LingBotWorldV2State()
    model._config = SimpleNamespace(seed=42, max_chunks=256)
    messages = []

    async def send(message):
        messages.append(message)

    model.send = send
    model.on_session_started()
    return model, messages


def _select(model):
    """Set an explicit test anchor without invoking image decoding or GPU work."""
    model._selected_input = UploadedFile(
        name="anchor.jpg", mime_type="image/jpeg", data=b"test"
    )
    model._image_source = "uploaded"


def test_empty_session_has_no_next_output():
    """No selected image means no schedulable chunk or declared frame count."""
    model, _ = _model()
    state = model._state_update()
    assert state.next_chunk is None
    assert state.next_chunk_frames is None


@pytest.mark.parametrize("command", ["release_camera", "set_camera"])
def test_exhausted_world_accepts_neutral_release(command):
    """Release is idempotent and cannot imply a nonexistent next chunk."""
    model, messages = _model()
    _select(model)
    model._limit_reached = True
    model._chunk_index = 256
    kwargs = dict(forward=0.0, strafe=0.0, vertical=0.0, pitch=0.0, yaw=0.0, roll=0.0)
    response = asyncio.run(
        getattr(model, command)(**(kwargs if command == "set_camera" else {}))
    )
    assert response.applies_to_chunk is None
    assert model._limit_reached is True
    assert model.state._reset_requested is False
    assert messages[-1].next_chunk is None
    assert messages[-1].completed_chunks == 256
    for axis in kwargs:
        with pytest.raises(CommandError) as error:
            asyncio.run(model.set_camera(**(kwargs | {axis: 0.05})))
        assert error.value.code == "rollout_limit_reached"


def test_public_schema_explains_direction_only_translation():
    """Schema-driven clients must not mistake normalized translation for speed."""
    contract = ModelContract.of(LingBotWorldV2)
    command = contract.commands["set_camera"]
    assert "Translation is direction-only" in command.description
    assert "Pitch/yaw/roll remain magnitude-sensitive" in command.description
    schema = contract.render_schema().to_openapi()
    operation = schema["paths"]["/events/set_camera"]["post"]
    fields = operation["requestBody"]["content"]["application/json"]["schema"][
        "properties"
    ]
    for axis in ("forward", "strafe", "vertical"):
        assert "does not change displacement scale" in fields[axis]["description"]
