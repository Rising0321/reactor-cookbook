"""Check Matrix's upload-gated session startup contract."""

from __future__ import annotations

import io
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest
from PIL import Image
from reactor_runtime import CommandError, UploadedFile
from reactor_runtime.interface.model.contract import ModelContract

MODEL_DIR = Path(__file__).parents[1]
sys.path.insert(0, str(MODEL_DIR))

GENERIC_PROMPT = (
    "An immersive first-person view that faithfully continues the input scene, "
    "preserving its existing environment, objects, geometry, materials, lighting, "
    "and visual style as the camera moves naturally through it."
)

from matrix_game_3_5 import MatrixGame35
from matrix_schema import MatrixGame35State


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


def test_session_starts_paused_without_selecting_an_image() -> None:
    """Wait for the viewer's anchor instead of generating from the demo image."""
    model = _model()

    model.on_session_started()
    state = model._state_update()

    assert model._selected_input is None
    assert state.image_source == "none"
    assert state.image_name == ""
    assert state.paused is True
    assert state.step_queued is False
    assert state.completed_chunks == 0
    assert state.next_chunk is None
    paused_field = (
        ModelContract.of(MatrixGame35)
        .commands["set_paused"]
        .command.__command_fields__["paused"]
    )
    assert paused_field.info.default is True


def test_first_upload_queues_one_chunk_and_keeps_generation_paused() -> None:
    """Provide visual upload confirmation without enabling continuous rollout."""
    model = _model()
    model.on_session_started()

    state = model.set_image(_upload(), "")

    assert state.image_source == "upload"
    assert state.image_name == "anchor.png"
    assert state.paused is True
    assert state.step_queued is True
    assert state.completed_chunks == 0
    assert state.next_chunk == 1
    assert state.prompt == GENERIC_PROMPT
    assert model.state._restart_requested is True


@pytest.mark.parametrize(
    ("command", "arguments"),
    [
        ("set_forward", (1.0,)),
        ("set_paused", (False,)),
        ("step", ()),
    ],
)
def test_generation_controls_require_an_uploaded_image(
    command: str,
    arguments: tuple[object, ...],
) -> None:
    """Reject controls that cannot produce a chunk before anchor selection."""
    model = _model()
    model.on_session_started()

    with pytest.raises(CommandError) as error:
        getattr(model, command)(*arguments)

    assert error.value.code == "image_required"
