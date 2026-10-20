"""LingBot-specific WBench camera/calibration hooks; no stepping implementation.

The core Runtime drives and captures this application exactly like any other
ReactorApp. Native model generation and KV-cache handling are inherited.
"""

from dataclasses import replace
import json

import numpy as np
from lingbot_world_v2 import LingBotWorldV2
from reactor_runtime import event, InputField, ModelMessage, MessageField, CommandError


class BenchGeneration(ModelMessage):
    intrinsics: list[float] = MessageField(description="Shared fx, fy, cx, cy.")
    noise_chunks: int | None = MessageField(
        default=None,
        description="Initial noise allocation horizon; null preserves native allocation.",
    )


class BenchCamera(ModelMessage):
    index: int = MessageField(
        description="Capture index that will consume these poses."
    )


class WBenchLingBot(LingBotWorldV2):
    @event(
        name="bench_camera",
        description="Stage one LingBot v2 camera block; trusted adapter only.",
    )
    async def bench_camera(
        self,
        index: int = InputField(ge=0, description="Next capture index."),
        poses: list[list[list[float]]] = InputField(
            description="Four normalized framewise 4x4 transforms."
        ),
    ) -> BenchCamera:
        if (
            self._bench.poisoned
            or index != self._bench.index
            or getattr(self, "_bench_camera", None) is not None
            or getattr(self, "_camera_path", None) is not None
            or (index != 0 and not getattr(self, "_bench_camera_mode", False))
        ):
            raise CommandError(
                "invalid_camera",
                "Camera blocks require a fresh exclusive live-camera session",
            )
        try:
            value = np.asarray(poses, dtype=np.float32)
        except (ValueError, TypeError, OverflowError) as error:
            raise CommandError("invalid_camera", "Invalid camera array") from error
        if (
            value.shape != (self._config.chunk_latents, 4, 4)
            or not np.isfinite(value).all()
            or not np.allclose(value[:, 3], [0, 0, 0, 1], atol=1e-5)
        ):
            raise CommandError(
                "invalid_camera", "Expected four finite homogeneous transforms"
            )
        rotation = value[:, :3, :3]
        if (
            not np.allclose(
                rotation.transpose(0, 2, 1) @ rotation, np.eye(3), atol=1e-4
            )
            or not np.allclose(np.linalg.det(rotation), 1, atol=1e-4)
            or (
                index == 0 and not np.array_equal(value[0], np.eye(4, dtype=np.float32))
            )
        ):
            raise CommandError(
                "invalid_camera", "Invalid rotations or nonidentity initial anchor"
            )
        self._bench_camera = np.ascontiguousarray(value)
        self._bench_camera_mode = True
        return BenchCamera(index=index)

    @event(
        name="bench_configure",
        description="Idle-only WBench camera calibration and optional noise allocation.",
    )
    async def bench_configure(
        self,
        intrinsics: list[float] = InputField(
            description="Four finite fx, fy, cx, cy values."
        ),
        noise_chunks: int | None = InputField(
            default=None,
            ge=1,
            le=256,
            description="Initial noise allocation horizon; null preserves native allocation.",
        ),
    ) -> BenchGeneration:
        if self._bench.poisoned or self._bench.index:
            raise CommandError(
                "invalid_setup", "Configure an idle, unused LingBot v2 session"
            )
        if (
            len(intrinsics) != 4
            or not np.isfinite(intrinsics).all()
            or min(intrinsics[:2]) <= 0
            or (
                noise_chunks is not None
                and (type(noise_chunks) is not int or not 1 <= noise_chunks <= self._config.max_chunks)
            )
        ):
            raise CommandError("invalid_setup", "Invalid intrinsics or noise horizon")
        self._bench_settings = BenchGeneration(
            intrinsics=list(intrinsics), noise_chunks=noise_chunks
        )
        return self._bench_settings

    async def process_input(self):
        index = self._bench.index
        camera = getattr(self, "_bench_camera", None)
        if getattr(self, "_bench_camera_mode", False) and camera is None:
            raise CommandError(
                "missing_camera", "Stage one camera block before each step"
            )
        consumed = await super().process_input()
        if camera is not None:
            if (
                getattr(self, "_camera_path", None) is not None
                or (consumed.anchor is not None) != (index == 0)
                or (index and consumed.world_id != self._bench_camera_world)
            ):
                raise CommandError(
                    "camera_reset",
                    "Camera streaming cannot reset/reanchor or use a finite path",
                )
            consumed = replace(consumed, poses=camera)
            self._bench_camera_world = consumed.world_id
            self._bench_camera = None
        settings = getattr(self, "_bench_settings", None)
        if settings is not None:
            if settings.noise_chunks is not None and index >= settings.noise_chunks:
                raise CommandError(
                    "noise_exhausted", "Configured noise allocation exhausted"
                )
            if consumed.anchor is not None:
                consumed = replace(
                    consumed,
                    anchor=replace(
                        consumed.anchor,
                        intrinsics=np.asarray(settings.intrinsics, dtype=np.float32),
                        noise_chunks=(
                            settings.noise_chunks if settings.noise_chunks is not None
                            else consumed.anchor.noise_chunks
                        ),
                    ),
                )
        self._wbench_input = consumed
        self._wbench_streamed = camera is not None
        return consumed

    async def process_output(self, outcome):
        if outcome.error is None:
            consumed, result = self._wbench_input, outcome.result
            root, index = self._bench.root, self._bench.index
            if isinstance(consumed.poses, np.ndarray):
                with (root / f"camera-{index:04}.json").open("x") as output:
                    json.dump(
                        dict(
                            index=index,
                            world_id=consumed.world_id,
                            anchored=consumed.anchor is not None,
                            native_chunk=result.chunk_index,
                            mode="streamed" if self._wbench_streamed else "native",
                            poses=consumed.poses.tolist(),
                        ),
                        output,
                    )
            if getattr(self, "_bench_settings", None) is not None and index == 0:
                noise = self._engine._backend._base_noise
                (root / "generation.json").write_text(
                    json.dumps(
                        dict(
                            intrinsics=consumed.anchor.intrinsics.tolist(),
                            noise_chunks=(
                                consumed.anchor.noise_chunks if consumed.anchor.noise_chunks is not None
                                else self._config.max_chunks
                            ),
                            requested_noise_chunks=self._bench_settings.noise_chunks,
                            seed=consumed.anchor.seed,
                            prompt=consumed.prompt,
                            noise_shape=list(noise.shape),
                        ),
                        indent=2,
                    )
                )
        return await super().process_output(outcome)
