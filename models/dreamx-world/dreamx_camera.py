"""Integrate DreamX-World's native keyboard camera trajectory one chunk at a time."""

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np

_FIRST_CHUNK_PIXEL_FRAMES = 9
FRAMES_PER_CHUNK = 12
_FIRST_LATENT_INDICES = (0, 1, 5)
_LATER_LATENT_INDICES = (0, 4, 8)


@dataclass(frozen=True)
class CameraChunk:
    """Carry three current latent poses and the previous chunk reference pose."""

    poses: np.ndarray
    reference_pose: np.ndarray | None


class DreamXCameraController:
    """Preserve upstream camera position and rotation across native chunks."""

    def __init__(self, speed: float) -> None:
        if speed <= 0:
            raise ValueError("DreamX camera speed must be positive")
        self._speed = speed
        self.reset()

    def reset(self) -> None:
        """Return the camera to the upstream identity starting pose."""
        self._position = np.zeros(3, dtype=np.float64)
        self._rotation = np.eye(3, dtype=np.float64)
        self._frame_index = 0
        self._previous_latent_pose: np.ndarray | None = None

    def plan_chunk(
        self, pressed_keys: frozenset[str], *, first_chunk: bool
    ) -> CameraChunk:
        """Return latent-aligned poses for one native three-latent-frame chunk."""
        frame_count = (
            _FIRST_CHUNK_PIXEL_FRAMES if first_chunk else FRAMES_PER_CHUNK
        )
        selected_indices = (
            _FIRST_LATENT_INDICES if first_chunk else _LATER_LATENT_INDICES
        )
        pixel_poses = np.stack(
            [self._advance_one_frame(pressed_keys) for _ in range(frame_count)],
            axis=0,
        )
        latent_poses = np.ascontiguousarray(
            pixel_poses[list(selected_indices)], dtype=np.float32
        )
        reference = self._previous_latent_pose
        self._previous_latent_pose = latent_poses[-1].copy()
        return CameraChunk(poses=latent_poses, reference_pose=reference)

    def _advance_one_frame(self, pressed_keys: frozenset[str]) -> np.ndarray:
        """Apply one upstream pixel-frame camera integration step."""
        move_step = self._speed * 0.05
        rotate_step = self._speed * math.pi / 180.0

        pitch_delta = rotate_step * (
            int("i" in pressed_keys) - int("k" in pressed_keys)
        )
        yaw_delta = rotate_step * (int("l" in pressed_keys) - int("j" in pressed_keys))
        if pitch_delta != 0.0 or yaw_delta != 0.0:
            self._rotation = self._rotation @ _euler_to_rotation_matrix(
                pitch_delta,
                yaw_delta,
                0.0,
            )

        local_movement = np.zeros(3, dtype=np.float64)
        local_movement[2] = move_step * (
            int("w" in pressed_keys) - int("s" in pressed_keys)
        )
        local_movement[0] = move_step * (
            int("d" in pressed_keys) - int("a" in pressed_keys)
        )
        self._position += self._rotation @ local_movement

        world_to_camera_rotation = self._rotation.T
        world_to_camera_translation = -world_to_camera_rotation @ self._position
        world_to_camera = np.hstack(
            (world_to_camera_rotation, world_to_camera_translation.reshape(3, 1))
        )
        pose = np.asarray(
            [
                self._frame_index,
                0.8,
                0.8,
                0.5,
                0.5,
                0.0,
                0.0,
                *world_to_camera.flatten(),
            ],
            dtype=np.float32,
        )
        self._frame_index += 1
        return pose


def _euler_to_rotation_matrix(roll: float, pitch: float, yaw: float) -> np.ndarray:
    """Return the rotation matrix used by upstream trajectory generation."""
    rotation_x = np.asarray(
        [
            [1.0, 0.0, 0.0],
            [0.0, math.cos(roll), -math.sin(roll)],
            [0.0, math.sin(roll), math.cos(roll)],
        ]
    )
    rotation_y = np.asarray(
        [
            [math.cos(pitch), 0.0, math.sin(pitch)],
            [0.0, 1.0, 0.0],
            [-math.sin(pitch), 0.0, math.cos(pitch)],
        ]
    )
    rotation_z = np.asarray(
        [
            [math.cos(yaw), -math.sin(yaw), 0.0],
            [math.sin(yaw), math.cos(yaw), 0.0],
            [0.0, 0.0, 1.0],
        ]
    )
    return rotation_z @ rotation_y @ rotation_x
