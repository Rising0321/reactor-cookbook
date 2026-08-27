"""Translate frontend controls into HY-World 1.5 native camera conditions."""

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np

_TRANSLATION_PER_LATENT = 0.08
_ROTATION_RADIANS_PER_LATENT = math.radians(3.0)
_ACTION_LABELS = {
    (0, 0, 0, 0): 0,
    (1, 0, 0, 0): 1,
    (0, 1, 0, 0): 2,
    (0, 0, 1, 0): 3,
    (0, 0, 0, 1): 4,
    (1, 0, 1, 0): 5,
    (1, 0, 0, 1): 6,
    (0, 1, 1, 0): 7,
    (0, 1, 0, 1): 8,
}


@dataclass(frozen=True)
class CameraControl:
    """Hold one atomic camera command sampled at a chunk boundary."""

    forward: float
    strafe: float
    pitch: float
    yaw: float


@dataclass(frozen=True)
class CameraChunk:
    """Carry continuous camera poses and matching discrete action labels."""

    viewmats: np.ndarray
    intrinsics: np.ndarray
    actions: np.ndarray


class NativeCameraPlanner:
    """Advance HY-World's four trained camera axes one latent at a time."""

    def __init__(self) -> None:
        self._pose = np.eye(4, dtype=np.float64)
        self._first_chunk = True

    def reset(self) -> None:
        """Restore the canonical first-person camera pose."""
        self._pose = np.eye(4, dtype=np.float64)
        self._first_chunk = True

    def plan(self, control: CameraControl, latent_count: int = 4) -> CameraChunk:
        """Return one chunk in the exact native pose and action representation.

        The first causal latent is an unmoved image anchor. Later latents apply
        at most 0.08 translation units and three degrees of rotation, matching
        the public HY-World trajectory generator. Continuous matrices and the
        corresponding 81-way action labels are returned together because the
        model was trained with both conditions.

        Args:
            control: Normalized native camera axes sampled for the chunk.
            latent_count: Number of latent camera positions in the chunk.

        Returns:
            Four world-to-camera matrices, normalized intrinsics, and labels.

        Raises:
            ValueError: If a control is outside [-1, 1] or count is not positive.
        """
        _validate_control(control)
        if latent_count <= 0:
            raise ValueError("latent_count must be positive")

        poses: list[np.ndarray] = []
        actions: list[int] = []
        if self._first_chunk:
            poses.append(self._pose.copy())
            actions.append(0)

        while len(poses) < latent_count:
            previous_pose = self._pose
            self._pose = _advance_pose(previous_pose, control)
            poses.append(self._pose.copy())
            actions.append(_action_label(previous_pose, self._pose))

        self._first_chunk = False
        c2w = np.stack(poses).astype(np.float32)
        viewmats = np.linalg.inv(c2w).astype(np.float32)
        intrinsic = np.asarray(
            [[0.5050505, 0.0, 0.5], [0.0, 0.89786756, 0.5], [0.0, 0.0, 1.0]],
            dtype=np.float32,
        )
        intrinsics = np.repeat(intrinsic[None], latent_count, axis=0)
        return CameraChunk(
            viewmats=np.ascontiguousarray(viewmats),
            intrinsics=np.ascontiguousarray(intrinsics),
            actions=np.asarray(actions, dtype=np.int64),
        )


def _advance_pose(pose: np.ndarray, control: CameraControl) -> np.ndarray:
    """Apply one native latent-rate motion in local camera coordinates."""
    next_pose = pose.copy()
    rotation_axes = _bounded_vector(control.pitch, control.yaw)
    pitch = rotation_axes[0] * _ROTATION_RADIANS_PER_LATENT
    yaw = rotation_axes[1] * _ROTATION_RADIANS_PER_LATENT
    next_pose[:3, :3] = next_pose[:3, :3] @ _rotation_y(yaw) @ _rotation_x(pitch)

    translation_axes = _bounded_vector(control.strafe, control.forward)
    local_translation = np.asarray(
        [translation_axes[0], 0.0, translation_axes[1]], dtype=np.float64
    )
    next_pose[:3, 3] += next_pose[:3, :3] @ local_translation * _TRANSLATION_PER_LATENT
    return next_pose


def _action_label(previous_pose: np.ndarray, current_pose: np.ndarray) -> int:
    """Classify the relative pose with the native movement and rotation thresholds."""
    relative = np.linalg.inv(previous_pose) @ current_pose
    movement = relative[:3, 3]
    norm = float(np.linalg.norm(movement))
    translation = [0, 0, 0, 0]
    if norm > 0.0001:
        direction = movement / norm
        if direction[2] > 0.5:
            translation[0] = 1
        elif direction[2] < -0.5:
            translation[1] = 1
        if direction[0] > 0.5:
            translation[2] = 1
        elif direction[0] < -0.5:
            translation[3] = 1

    rotation_matrix = relative[:3, :3]
    yaw = math.asin(float(np.clip(-rotation_matrix[2, 0], -1.0, 1.0)))
    pitch = math.atan2(rotation_matrix[2, 1], rotation_matrix[2, 2])
    pitch, yaw = math.degrees(pitch), math.degrees(yaw)
    rotation = [0, 0, 0, 0]
    if yaw > 0.05:
        rotation[0] = 1
    elif yaw < -0.05:
        rotation[1] = 1
    if pitch > 0.05:
        rotation[2] = 1
    elif pitch < -0.05:
        rotation[3] = 1
    return _ACTION_LABELS[tuple(translation)] * 9 + _ACTION_LABELS[tuple(rotation)]


def _bounded_vector(first: float, second: float) -> np.ndarray:
    """Return a two-axis vector with magnitude at most one."""
    value = np.asarray([first, second], dtype=np.float64)
    norm = float(np.linalg.norm(value))
    if norm > 1.0:
        value /= norm
    return value


def _rotation_x(angle: float) -> np.ndarray:
    """Return a right-handed pitch matrix."""
    cosine, sine = math.cos(angle), math.sin(angle)
    return np.asarray(
        [[1.0, 0.0, 0.0], [0.0, cosine, -sine], [0.0, sine, cosine]],
        dtype=np.float64,
    )


def _rotation_y(angle: float) -> np.ndarray:
    """Return a right-handed yaw matrix."""
    cosine, sine = math.cos(angle), math.sin(angle)
    return np.asarray(
        [[cosine, 0.0, sine], [0.0, 1.0, 0.0], [-sine, 0.0, cosine]],
        dtype=np.float64,
    )


def _validate_control(control: CameraControl) -> None:
    """Require every public camera axis to be normalized."""
    for name in ("forward", "strafe", "pitch", "yaw"):
        value = getattr(control, name)
        if not -1.0 <= value <= 1.0:
            raise ValueError(f"{name} must be between -1 and 1")
