"""Check Matrix's upload-gated session startup contract."""

from __future__ import annotations

import asyncio
import io
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import AsyncMock, Mock

import numpy as np
import pytest
from PIL import Image
from reactor_runtime import ApplicationError, CommandError, StepOutcome, UploadedFile
from reactor_runtime.interface.model.contract import ModelContract

from matrix_game_3_5 import MatrixGame35
from matrix_game_3_5_types import MatrixGame35State

MODEL_DIR = Path(__file__).parents[1]

GENERIC_PROMPT = (
    "An immersive first-person view that faithfully continues the input scene, "
    "preserving its existing environment, objects, geometry, materials, lighting, "
    "and visual style as the camera moves naturally through it."
)


def _upload() -> UploadedFile:
    """Return a small valid anchor upload."""
    payload = io.BytesIO()
    Image.new("RGB", (8, 8), color=(20, 40, 60)).save(payload, format="PNG")
    return UploadedFile(
        name="anchor.png",
        mime_type="image/png",
        data=payload.getvalue(),
    )


def _model() -> MatrixGame35:
    """Return a loaded-enough model for lifecycle and command checks."""
    model = MatrixGame35()
    model.state = MatrixGame35State()
    model._config = SimpleNamespace(seed=3407, max_chunks=512)
    model._default_prompt = GENERIC_PROMPT
    return model


def test_session_waits_for_an_image_selection() -> None:
    """Wait for the viewer's anchor instead of generating from the demo image."""
    model = _model()

    model.on_session_started()
    state = model._state_update()

    assert model._selected_input is None
    assert state.image_source == "none"
    assert state.image_name == ""
    assert state.completed_chunks == 0
    assert state.next_chunk is None
    assert set(ModelContract.of(MatrixGame35).commands) == {
        "reset",
        "set_forward",
        "set_image",
        "set_pitch",
        "set_prompt",
        "set_roll",
        "set_strafe",
        "set_vertical",
        "set_yaw",
    }


def test_first_upload_starts_continuous_generation() -> None:
    """Start continuous generation from the uploaded anchor."""
    model = _model()
    model.on_session_started()

    state = model.set_image(_upload(), "")

    assert state.image_source == "uploaded"
    assert state.image_name == "anchor.png"
    assert state.completed_chunks == 0
    assert state.next_chunk == 1
    assert state.prompt == GENERIC_PROMPT
    assert model.state._restart_requested is True


def test_generation_controls_require_an_uploaded_image() -> None:
    """Reject controls that cannot produce a chunk before anchor selection."""
    model = _model()
    model.on_session_started()

    with pytest.raises(CommandError) as error:
        model.set_forward(1.0)

    assert error.value.code == "image_required"


def make_model():
    model = MatrixGame35()
    model.state = MatrixGame35State()
    model._config = SimpleNamespace(
        seed=42,
        max_chunks=20,
        chunk_latents=4,
        upload_intrinsics=(1, 1, 0.5, 0.5),
    )
    model.send = AsyncMock()
    model._selected_input = Path("anchor.png")
    model.state._restart_requested = False
    return model


def test_input_waits_for_image_and_rejects_rollout_limit():
    model = make_model()
    model._selected_input = None
    with pytest.raises(ApplicationError):
        asyncio.run(model.process_input())
    model._selected_input = Path("anchor.png")
    model.state._limit_reached = True
    with pytest.raises(ApplicationError):
        asyncio.run(model.process_input())


def test_failure_does_not_advance_progress():
    model = make_model()
    model.state._restart_requested = True
    with pytest.raises(RuntimeError, match="failed"):
        asyncio.run(model.process_output(StepOutcome(error=RuntimeError("failed"))))
    assert model._chunk_index == 0
    assert model.state._restart_requested


def test_controls_are_snapshotted_and_native_chunks_remain_continuous():
    model = make_model()
    backend = Mock()
    model._backend = backend
    model._planner = Mock()
    frames = np.zeros((12, 8, 8, 3), dtype=np.uint8)
    backend.generate_chunk.return_value = frames
    model.state.yaw = 0.25
    snapshot = asyncio.run(model.process_input())
    model.state.yaw = -0.25
    assert snapshot.yaw == 0.25
    result = model.generate(snapshot)
    asyncio.run(model.process_output(StepOutcome(result=result)))
    assert model._chunk_index == 1
    backend.reset.assert_not_called()
    backend.generate_chunk.assert_called_once()
    assert model.send.called
    backend.generate_chunk.return_value = np.zeros((12, 8, 8, 3), dtype=np.uint8)
    for _ in range(9):
        snapshot = asyncio.run(model.process_input())
        result = model.generate(snapshot)
        asyncio.run(model.process_output(StepOutcome(result=result)))
    assert model._chunk_index == 10
    assert backend.generate_chunk.call_count == 10
    backend.reset.assert_not_called()
