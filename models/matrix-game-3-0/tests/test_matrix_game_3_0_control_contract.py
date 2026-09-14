"""Atomic neutral and schema-defined keyboard meanings."""

import asyncio
from types import SimpleNamespace

from matrix_game_3_0 import MatrixGame30
from matrix_game_3_0_types import MatrixGame30State
from reactor_runtime.interface.model.contract import ModelContract


def test_atomic_neutral_is_valid_at_limit_and_before_image_selection():
    """One release command neutralizes all six channels even at the limit."""
    model = MatrixGame30()
    model.state = MatrixGame30State()
    model._config = SimpleNamespace(max_chunks=64)
    model.state._pressed_keys = frozenset("wasd")
    model.state.pitch = 0.75
    model.state.yaw = -0.25
    model.state._limit_reached = True

    async def run():
        result = await model.release_controls()
        assert not result.pressed_keys
        assert result.pitch == result.yaw == 0
        assert result.applies_to_chunk is None
        for key in "wasd":
            await model.set_key_state(key, False)
        await model.set_pitch(0.0)
        await model.set_yaw(0.0)

    asyncio.run(run())


def test_schema_supplies_directions_and_atomic_release():
    """A schema-only client can identify directions and the neutral command."""
    contract = ModelContract.of(MatrixGame30)
    assert "release_controls" in contract.commands
    field = contract.commands["set_key_state"].command.__command_fields__["key"]
    for meaning in ("forward", "backward", "left", "right"):
        assert meaning in field.info.description
