"""No unsolicited demo generation or implicit world reset."""

import asyncio
from types import SimpleNamespace

from test_evoke import _ready_model


def test_session_waits_for_conditioning_without_reset_or_generation():
    """No backend reset, generation, or default demo precedes user selection."""
    model, backend, _ = _ready_model()
    model._config = SimpleNamespace(seed=42, max_chunks=2)
    model.on_session_started()
    assert model._media is None
    assert model._state_update().next_chunk is None
    assert model._state_update().input_source == "none"
    assert not model.state._restart_requested

    async def run():
        output = model.inference()
        for _ in range(3):
            assert await anext(output) is None
        await output.aclose()

    asyncio.run(run())
    assert not backend.reset_calls
    assert not backend.generate_calls


def test_capacity_freezes_world_and_explicit_reset_resumes():
    """Reaching the native horizon retains the world and permits neutral release."""
    model, backend, messages = _ready_model()
    model._config = SimpleNamespace(max_chunks=2)

    async def run():
        output = model.inference()
        assert await anext(output) is not None
        assert await anext(output) is not None
        assert model.state._limit_reached
        for _ in range(3):
            assert await anext(output) is None
        assert len(backend.reset_calls) == 1
        assert len(backend.generate_calls) == 2
        assert model._state_update().next_chunk is None
        await model.set_pitch(0.0)
        await model.release_controls()
        assert model.state.pitch == model.state.yaw == model.state.forward == 0
        await model.reset(42)
        assert await anext(output) is not None
        assert len(backend.reset_calls) == 2
        assert (
            sum(type(message).__name__ == "RolloutLimitReached" for message in messages)
            == 1
        )
        await output.aclose()

    asyncio.run(run())
