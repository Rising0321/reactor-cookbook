"""Verify published camera directions in OpenCV camera-to-world coordinates."""

import numpy as np
import pytest

from evoke_camera import CameraMotionPlanner, MotionConfig


@pytest.mark.parametrize("pitch", [-1.0, 1.0])
def test_pitch_matches_published_upward_sign(pitch):
    """Positive pitch looks up; a fixed forward landmark moves down on screen."""
    planner = CameraMotionPlanner(MotionConfig(24, 1, 30))
    poses = planner.plan_chunk(
        strafe=0,
        vertical=0,
        forward=0,
        pitch=pitch,
        yaw=0,
        roll=0,
        frame_count=13,
    )
    rotation = poses[-1, :3, :3]
    optical_axis = rotation @ np.array([0.0, 0.0, 1.0])
    assert optical_axis[1] * pitch < 0
    landmark_in_camera = rotation.T @ np.array([0.0, 0.0, 10.0])
    assert landmark_in_camera[1] * pitch > 0


def test_opposite_pitch_chunks_restore_orientation():
    """Equal opposite continuation chunks restore the preceding orientation."""
    planner = CameraMotionPlanner(MotionConfig(24, 1, 30))
    inputs = dict(strafe=0, vertical=0, forward=0, yaw=0, roll=0)
    planner.plan_chunk(**inputs, pitch=0, frame_count=13)
    planner.plan_chunk(**inputs, pitch=1, frame_count=12)
    final = planner.plan_chunk(**inputs, pitch=-1, frame_count=12)
    np.testing.assert_allclose(final[-1, :3, :3], np.eye(3), atol=1e-6)
