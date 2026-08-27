"""Integrate Echo-WM's native pure-camera motion across interactive chunks."""

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class MotionConfig:
    """Hold the native camera integration constants."""

    fps: float
    translation_speed: float
    rotation_speed_degrees: float
    pitch_speed_degrees: float
    pitch_limit_degrees: float


@dataclass(frozen=True)
class CameraChunk:
    """Carry the three camera poses aligned with one causal latent block."""

    latent_poses: np.ndarray


def _rotation_x(angle: float) -> np.ndarray:
    cosine, sine = math.cos(angle), math.sin(angle)
    return np.asarray(
        [[1.0, 0.0, 0.0], [0.0, cosine, -sine], [0.0, sine, cosine]],
        dtype=np.float64,
    )


def _rotation_y(angle: float) -> np.ndarray:
    cosine, sine = math.cos(angle), math.sin(angle)
    return np.asarray(
        [[cosine, 0.0, sine], [0.0, 1.0, 0.0], [-sine, 0.0, cosine]],
        dtype=np.float64,
    )


class EchoCameraPlanner:
    """Preserve Echo-WM camera pose, inertia, and pitch across chunks."""

    def __init__(self, config: MotionConfig) -> None:
        if config.fps <= 0 or config.translation_speed <= 0:
            raise ValueError("Echo-WM camera rates must be positive")
        self._config = config
        self.reset()

    def reset(self) -> None:
        """Return camera position, rotation, and inertia to upstream defaults."""
        self._pose = np.eye(4, dtype=np.float64)
        self._pitch = 0.0
        self._velocity = np.zeros(4, dtype=np.float64)
        self._active_axes: frozenset[str] = frozenset()

    def plan_chunk(
        self,
        *,
        forward: float,
        strafe: float,
        pitch: float,
        yaw: float,
        frame_count: int,
    ) -> CameraChunk:
        """Advance native camera integration and return latent-aligned poses."""
        if frame_count <= 0 or frame_count % 8:
            raise ValueError(
                "Echo-WM camera chunks must be a positive multiple of 8 frames"
            )
        pixel_poses = [
            self._advance_one_frame(
                forward=forward,
                strafe=strafe,
                pitch=pitch,
                yaw=yaw,
            )
            for _ in range(frame_count)
        ]
        indices = np.arange(7, frame_count, 8)
        return CameraChunk(
            latent_poses=np.ascontiguousarray(
                np.stack(pixel_poses)[indices], dtype=np.float32
            )
        )

    def _advance_one_frame(
        self,
        *,
        forward: float,
        strafe: float,
        pitch: float,
        yaw: float,
    ) -> np.ndarray:
        config = self._config
        target = np.asarray(
            [
                forward * config.translation_speed,
                strafe * config.translation_speed,
                yaw * math.radians(config.rotation_speed_degrees),
                pitch * math.radians(config.pitch_speed_degrees),
            ],
            dtype=np.float64,
        )
        active = frozenset(
            name
            for name, value in zip(
                ("forward", "strafe", "yaw", "pitch"),
                (forward, strafe, yaw, pitch),
                strict=True,
            )
            if value != 0.0
        )
        sign_changed = any(
            np.sign(target[index]) != np.sign(self._velocity[index])
            and target[index] != 0.0
            for index in range(4)
        )
        if active - self._active_axes or sign_changed:
            self._velocity = target
        else:
            time_constant = 0.45 if np.any(target) else 1.0
            blend = 1.0 - math.exp(-(1.0 / config.fps) / time_constant)
            self._velocity += (target - self._velocity) * blend
        self._active_axes = active

        next_pitch = float(
            np.clip(
                self._pitch + self._velocity[3],
                -math.radians(config.pitch_limit_degrees),
                math.radians(config.pitch_limit_degrees),
            )
        )
        pitch_step = next_pitch - self._pitch
        self._pitch = next_pitch
        rotation = (
            _rotation_y(self._velocity[2])
            @ self._pose[:3, :3]
            @ _rotation_x(pitch_step)
        )
        forward_axis = rotation[:, 2].copy()
        right_axis = rotation[:, 0].copy()
        forward_axis[1] = 0.0
        right_axis[1] = 0.0
        forward_axis /= max(float(np.linalg.norm(forward_axis)), 1e-6)
        right_axis /= max(float(np.linalg.norm(right_axis)), 1e-6)
        pose = np.eye(4, dtype=np.float64)
        pose[:3, :3] = rotation
        pose[:3, 3] = (
            self._pose[:3, 3]
            + forward_axis * self._velocity[0]
            + right_axis * self._velocity[1]
        )
        self._pose = pose
        return pose.copy()
