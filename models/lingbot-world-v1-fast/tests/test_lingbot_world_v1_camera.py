"""Verify public camera directions in OpenCV coordinates without GPU weights."""

import numpy as np
import pytest
from lingbot_world_v1_camera import CameraMotionPlanner, MotionConfig


@pytest.mark.parametrize("value", [-1.0, -0.05, 0.05, 1.0])
def test_pitch_moves_optical_axis_up_and_landmark_down(value):
    """Positive pitch looks up; a fixed front landmark moves down in the image."""
    planner = CameraMotionPlanner(MotionConfig(1.0, 8.0))
    controls = dict(
        strafe=0.0, vertical=0.0, forward=0.0, pitch=value, yaw=0.0, roll=0.0
    )
    pose = planner.plan_chunk(**controls)[1]
    optical_axis = pose[:3, :3] @ np.array([0.0, 0.0, 1.0])
    landmark = np.linalg.inv(pose) @ np.array([0.0, 0.0, 5.0, 1.0])
    assert optical_axis[1] * value < 0
    assert landmark[1] / landmark[2] * value > 0
    np.testing.assert_allclose(pose[:3, :3].T @ pose[:3, :3], np.eye(3), atol=1e-6)


@pytest.mark.parametrize("axis,component,sign", [("yaw", 0, 1), ("vertical", 1, -1)])
def test_other_axis_directions_remain_consistent(axis, component, sign):
    """Positive yaw looks right; positive vertical translates upward."""
    planner = CameraMotionPlanner(MotionConfig(1.0, 8.0))
    controls = dict(strafe=0.0, vertical=0.0, forward=0.0, pitch=0.0, yaw=0.0, roll=0.0)
    controls[axis] = 1.0
    pose = planner.plan_chunk(**controls)[1]
    vector = pose[:3, 3] if axis == "vertical" else pose[:3, 2]
    assert vector[component] * sign > 0
