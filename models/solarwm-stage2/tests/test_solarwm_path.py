import asyncio
import json
from types import SimpleNamespace

import numpy as np
import pytest
from reactor_runtime import ApplicationError, CommandError, StepOutcome, UploadedFile

from solarwm_stage2 import SolarWM
from solarwm_stage2_camera import CameraMotionPlanner, MotionConfig, load_camera_path
from solarwm_stage2_types import SolarWMState


def path(frames=117):
    poses = np.repeat(np.eye(4)[None], frames, axis=0)
    poses[:, 2, 3] = np.arange(frames) * 0.03
    return json.dumps(dict(poses=poses.tolist())).encode()


def test_causal_indices_and_finite_budget():
    value = load_camera_path(path(), 320)
    assert value.frames == 117 and value.chunks == 10
    np.testing.assert_allclose(
        value.poses[:, 2, 3], np.array([0, *range(1, 117, 4)]) * 0.03, atol=1e-6
    )
    with pytest.raises(ValueError):
        value.chunk(10)


@pytest.mark.parametrize("frames", [0, 8, 10, 116, 120])
def test_reject_partial_chunks(frames):
    with pytest.raises(ValueError):
        load_camera_path(path(frames), 320)


def test_bad_pose_atomic_and_ten_chunks():
    app = SolarWM()
    app.state = SolarWMState()
    app._config = SimpleNamespace(max_chunks=320)
    app.state._selected_image = np.zeros((8, 8, 3), np.uint8)
    app.state._planner = CameraMotionPlanner(MotionConfig(1, 8))

    async def run():
        uploaded = UploadedFile(
            name="path.json", mime_type="application/json", data=path()
        )
        selected = await app.set_camera_path(uploaded)
        assert selected.chunks == 10
        original = app._camera_path
        with pytest.raises(CommandError):
            await app.set_camera_path(
                UploadedFile(name="bad.json", mime_type="application/json", data=b"{}")
            )
        assert app._camera_path is original
        with pytest.raises(CommandError):
            await app.set_forward(1.0)
        for i in range(10):
            value = await app.process_input()
            again = await app.process_input()
            np.testing.assert_array_equal(value.poses, again.poses)
            np.testing.assert_array_equal(value.poses, original.chunk(i))
            assert (value.anchor is not None) == (i == 0)
            result = SimpleNamespace(
                world_id=value.world_id,
                chunk_index=i + 1,
                complete=False,
                max_chunks=320,
                frames=np.zeros((9 if i == 0 else 12, 8, 8, 3), np.uint8),
            )
            await app.process_output(StepOutcome(result=result, elapsed=0.1))
        with pytest.raises(ApplicationError):
            await app.process_input()
        await app.reset(42)
        np.testing.assert_array_equal(
            (await app.process_input()).poses, original.chunk(0)
        )

    asyncio.run(run())
