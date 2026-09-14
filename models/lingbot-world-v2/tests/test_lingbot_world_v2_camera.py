"""Verify public camera directions in OpenCV coordinates without GPU weights."""

import numpy as np
import pytest
from lingbot_world_v2_camera import CameraMotionPlanner


@pytest.mark.parametrize("value", [-1.0, -0.05, 0.05, 1.0])
def test_pitch_moves_optical_axis_up_and_landmark_down(value):
    """Positive pitch looks up; a fixed front landmark moves down in the image."""
    planner = CameraMotionPlanner(16.0, 45.0)
    controls = dict(
        strafe=0.0, vertical=0.0, forward=0.0, pitch=value, yaw=0.0, roll=0.0
    )
    pose = planner.plan_chunk(**controls, latent_frames=4, temporal_stride=4)[1]
    optical_axis = pose[:3, :3] @ np.array([0.0, 0.0, 1.0])
    landmark = np.linalg.inv(pose) @ np.array([0.0, 0.0, 5.0, 1.0])
    assert optical_axis[1] * value < 0
    assert landmark[1] / landmark[2] * value > 0
    np.testing.assert_allclose(pose[:3, :3].T @ pose[:3, :3], np.eye(3), atol=1e-6)


@pytest.mark.parametrize("axis,component,sign", [("yaw", 0, 1), ("vertical", 1, -1)])
def test_other_axis_directions_remain_consistent(axis, component, sign):
    """Positive yaw looks right; positive vertical translates upward."""
    planner = CameraMotionPlanner(16.0, 45.0)
    controls = dict(strafe=0.0, vertical=0.0, forward=0.0, pitch=0.0, yaw=0.0, roll=0.0)
    controls[axis] = 1.0
    pose = planner.plan_chunk(**controls, latent_frames=4, temporal_stride=4)[1]
    vector = pose[:3, 3] if axis == "vertical" else pose[:3, 2]
    assert vector[component] * sign > 0


@pytest.mark.parametrize("axis", ["forward", "strafe", "vertical"])
def test_translation_is_direction_only_and_preserves_sign(axis):
    """Nonzero single-axis amplitude changes cannot increase native translation."""
    poses = []
    for magnitude in (0.05, 0.1, -0.05):
        planner = CameraMotionPlanner(16.0, 45.0)
        controls = dict(
            strafe=0.0, vertical=0.0, forward=0.0, pitch=0.0, yaw=0.0, roll=0.0
        )
        controls[axis] = magnitude
        poses.append(planner.plan_chunk(**controls, latent_frames=4, temporal_stride=4))
    np.testing.assert_allclose(poses[0], poses[1], atol=1e-6)
    np.testing.assert_allclose(poses[0][:, :3, 3], -poses[2][:, :3, 3], atol=1e-6)


def test_translation_ratios_and_angular_magnitude_remain_effective():
    """Translation normalization preserves axis ratios and does not normalize pitch."""
    rotations = []
    for pitch in (0.05, 0.1):
        planner = CameraMotionPlanner(16.0, 45.0)
        pose = planner.plan_chunk(
            strafe=0.1,
            vertical=0.0,
            forward=0.05,
            pitch=pitch,
            yaw=0.0,
            roll=0.0,
            latent_frames=4,
            temporal_stride=4,
        )[1]
        assert pose[0, 3] / pose[2, 3] == pytest.approx(2.0)
        rotations.append(pose[:3, :3])
    assert not np.allclose(rotations[0], rotations[1])
