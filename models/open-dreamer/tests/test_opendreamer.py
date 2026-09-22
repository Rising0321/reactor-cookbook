"""Verify native step-loop conditioning, caches and transient controls without a GPU."""

from __future__ import annotations

import asyncio
from contextlib import nullcontext
from types import SimpleNamespace

import numpy as np
import pytest
from reactor_runtime import ApplicationError, StepOutcome

import opendreamer
from opendreamer import OpenDreamer
from opendreamer_types import OpenDreamerState, RolloutConditioning


def ready_model(monkeypatch):
    """Create a two-frame conditioning sequence and deterministic mock JAX calls."""
    model = OpenDreamer()
    model.state = OpenDreamerState()
    model._config = SimpleNamespace(seed=42)
    model._latent_shape = (1, 1, 1, 1)
    model._empty_dynamics_cache = 0
    model._empty_tokenizer_cache = 0
    model._deps = {
        "jax": SimpleNamespace(
            random=SimpleNamespace(
                PRNGKey=lambda seed: seed, split=lambda seed: (seed + 1, seed + 2)
            ),
            block_until_ready=lambda value: value,
        ),
        "jnp": np,
        "action_type": SimpleNamespace,
        "mouse_to_categorical": lambda x, y: x + y,
    }
    model._key_to_index = {
        "key.keyboard.w": 0,
        "mouse.0": 1,
        "mouse.wheel_neg": 2,
        "mouse.wheel_pos": 3,
    }
    conditioning = RolloutConditioning(
        np.zeros((2, 8, 8, 3), dtype=np.uint8),
        SimpleNamespace(
            binary=np.zeros((1, 2, 4)), categorical=np.zeros((1, 2)), continuous=None
        ),
    )
    model._demos = {"demo_1": conditioning}
    model._conditioning_source = "demo_1"
    calls = []

    def observe(tokenizer, dynamics, frame, action, dc, tc):
        calls.append(("observe", dc, tc))
        return dc + 1, tc + 1

    def generate(tokenizer, dynamics, action, shape, dc, tc, rng):
        calls.append(("generate", dc, tc, action))
        return np.zeros((1, 1, 8, 8, 3), dtype=np.uint8), dc + 1, tc + 1, rng

    model._observe_frame_jit = observe
    model._next_frame_jit = generate
    monkeypatch.setattr(opendreamer, "mesh_context", lambda *args: nullcontext())
    model.state._reset_requested = True
    return model, calls


def step(model):
    """Drive the three public step hooks with no live state visible to generation."""
    input = asyncio.run(model.process_input())
    state = model.state
    model.state = None
    result = model.generate(input)
    model.state = state
    return asyncio.run(model.process_output(StepOutcome(result=result)))


def test_conditioning_then_ten_actions_preserve_cache(monkeypatch):
    model, calls = ready_model(monkeypatch)
    model.state._delta_x = 5
    assert step(model) is None
    assert step(model) is None
    assert model.state._delta_x == 5
    for index in range(10):
        model.state._pressed_keys = frozenset({"w"})
        model.state._delta_x = index + 1
        model.state._wheel_delta = 1
        output = step(model)
        assert output.main_video.shape == (8, 8, 3)
        assert model.state._delta_x == model.state._wheel_delta == 0
        assert model.state._pressed_keys == frozenset({"w"})
        np.testing.assert_array_equal(calls[-1][3].binary, [[1, 0, 0, 1]])
    assert [call[0] for call in calls] == ["observe"] * 2 + ["generate"] * 10
    assert model._dynamics_cache == model._tokenizer_cache == 12
    assert "fps" not in vars(OpenDreamer)


def test_reset_reobserves_conditioning_and_session_end_releases_cache(monkeypatch):
    model, calls = ready_model(monkeypatch)
    for _ in range(4):
        step(model)
    model.state._reset_requested = True
    assert step(model) is None
    assert calls[-1] == ("observe", 0, 0)
    model.on_session_ended()
    assert model._dynamics_cache is None
    assert model._tokenizer_cache is None
    assert model._conditioning is None


def test_missing_conditioning_refuses_and_error_does_not_consume_input(monkeypatch):
    model, _ = ready_model(monkeypatch)
    model._demos = {}
    with pytest.raises(ApplicationError):
        asyncio.run(model.process_input())
    model.state._delta_x = 9
    with pytest.raises(RuntimeError, match="failed"):
        asyncio.run(model.process_output(StepOutcome(error=RuntimeError("failed"))))
    assert model.state._delta_x == 9
    assert model.state._reset_requested
