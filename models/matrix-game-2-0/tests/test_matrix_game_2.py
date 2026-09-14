"""Weight-free checks of the universal checkpoint's control contract."""

import unittest
from pathlib import Path
from unittest.mock import AsyncMock

from matrix_game_2 import MatrixGame2
from matrix_game_2_backend import _keyboard_vector, _mouse_vector
from matrix_game_2_types import MatrixGame2State
from reactor_runtime import CommandError


class NativeVectorTests(unittest.TestCase):
    """Lock channel identities to the published universal checkpoint."""

    def test_keyboard_order_matches_upstream_universal_checkpoint(self):
        """Keep each semantic key in its native channel."""
        expected = {
            "w": (1.0, 0.0, 0.0, 0.0),
            "s": (0.0, 1.0, 0.0, 0.0),
            "a": (0.0, 0.0, 1.0, 0.0),
            "d": (0.0, 0.0, 0.0, 1.0),
        }
        for key, vector in expected.items():
            with self.subTest(key=key):
                self.assertEqual(_keyboard_vector((key,)), vector)

    def test_neutral_and_combination_are_not_channel_permutations(self):
        """Preserve neutral and multi-hot representations."""
        self.assertEqual(_keyboard_vector(()), (0.0, 0.0, 0.0, 0.0))
        self.assertEqual(_keyboard_vector(("a", "w")), (1.0, 0.0, 1.0, 0.0))
        with self.assertRaises(ValueError):
            _keyboard_vector(("left",))

    def test_mouse_order_and_native_scale(self):
        """Keep pitch before yaw at the native scale."""
        self.assertEqual(_mouse_vector(1.0, -1.0), (0.1, -0.1))
        self.assertEqual(_mouse_vector(-1.0, 1.0), (-0.1, 0.1))
        self.assertEqual(_mouse_vector(0.0, 0.0), (0.0, 0.0))
        with self.assertRaises(ValueError):
            _mouse_vector(1.1, 0.0)


class CommandContractTests(unittest.IsolatedAsyncioTestCase):
    """Check pulse lifecycle independently of GPU generation quality."""

    def setUp(self):
        """Prepare a selected world with a mocked message transport."""
        self.model = MatrixGame2()
        self.model.state = MatrixGame2State()
        self.model._selected_input = Path("image.png")
        self.model.send = AsyncMock()

    async def test_a_and_d_hold_release_independently(self):
        """Releasing A must not clear held D or exchange its channel."""
        await self.model.set_key_state(key="a", pressed=True)
        self.assertEqual(self.model.state._pressed_keys, frozenset(("a",)))
        await self.model.set_key_state(key="d", pressed=True)
        await self.model.set_key_state(key="a", pressed=False)
        self.assertEqual(self.model.state._pressed_keys, frozenset(("d",)))
        self.model.release_controls()
        self.assertEqual(self.model.state._pressed_keys, frozenset())

    async def test_neutral_does_not_reset_world(self):
        """A release must preserve the current rollout's causal history."""
        self.model._chunk_index = 17
        self.model.state.pitch = 0.5
        self.model.state.yaw = -0.5
        await self.model.set_key_state(key="a", pressed=True)
        state = self.model.release_controls()
        self.assertEqual(state.completed_chunks, 17)
        self.assertFalse(state.reset_queued)
        self.assertEqual((state.pressed_keys, state.pitch, state.yaw), ([], 0.0, 0.0))

    async def test_exhaustion_requires_explicit_reset(self):
        """Capacity exhaustion must reject control instead of restarting silently."""
        self.model._chunk_index = 120
        self.model.state._limit_reached = True
        with self.assertRaises(CommandError):
            await self.model.set_key_state(key="a", pressed=True)
        self.assertEqual(self.model._chunk_index, 120)
        self.assertFalse(self.model.state._restart_requested)

    async def test_all_neutral_commands_are_valid_after_exhaustion(self):
        """Preserve the last valid output while clearing controls at the horizon."""
        self.model._chunk_index = 120
        self.model.state._limit_reached = True
        self.model.state._pressed_keys = frozenset(("a", "d"))
        self.model.state.pitch = 0.5
        self.model.state.yaw = -0.5
        for key in ("a", "d"):
            response = await self.model.set_key_state(key=key, pressed=False)
            self.assertIsNone(response.applies_to_chunk)
        for axis in ("pitch", "yaw"):
            response = await getattr(self.model, f"set_{axis}")(**{axis: 0.0})
            self.assertIsNone(response.applies_to_chunk)
        state = self.model.release_controls()
        self.assertEqual((state.pressed_keys, state.pitch, state.yaw), ([], 0.0, 0.0))
        self.assertEqual(state.completed_chunks, 120)
        self.assertTrue(state.limit_reached)
        self.assertFalse(state.reset_queued)
        self.assertIsNone(state.next_chunk)

    async def test_nonzero_camera_commands_are_rejected_after_exhaustion(self):
        """Reject every nonneutral direction without changing the exhausted world."""
        self.model.state._limit_reached = True
        for axis in ("pitch", "yaw"):
            for value in (-0.5, 0.5):
                with (
                    self.subTest(axis=axis, value=value),
                    self.assertRaises(CommandError),
                ):
                    await getattr(self.model, f"set_{axis}")(**{axis: value})
        self.assertEqual((self.model.state.pitch, self.model.state.yaw), (0.0, 0.0))

    async def test_neutral_before_image_does_not_claim_a_future_chunk(self):
        """Allow safe release without inventing a playable world."""
        self.model._selected_input = None
        key = await self.model.set_key_state(key="a", pressed=False)
        pitch = await self.model.set_pitch(pitch=0.0)
        yaw = await self.model.set_yaw(yaw=0.0)
        for response in (key, pitch, yaw):
            self.assertIsNone(response.applies_to_chunk)


if __name__ == "__main__":
    unittest.main()
