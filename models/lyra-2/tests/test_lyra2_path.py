import asyncio
import json
from unittest.mock import AsyncMock, Mock

import numpy as np
import pytest
from lyra2 import Lyra2
from lyra2_camera import Lyra2CameraPlanner, camera_path_chunk, load_camera_path
from lyra2_model import Lyra2Result
from lyra2_types import Lyra2State
from reactor_runtime import ApplicationError, CommandError, StepOutcome, UploadedFile


def payload(frames=801):
    poses = np.repeat(np.eye(4)[None], frames, axis=0)
    poses[:, 0, 3] = 7
    poses[:, 2, 3] = np.arange(frames) * 0.01
    return json.dumps({"poses": poses.tolist()}).encode()


def test_exact_native80_pose_slices_and_calibration():
    poses = load_camera_path(payload())
    seed = np.eye(4)
    seed[:3, 3] = [1, 2, 3]
    calibration = seed @ np.linalg.inv(poses[0])
    for index in range(10):
        chunk = camera_path_chunk(poses, calibration, index, np.diag([100, 200, 1]))
        expected = np.repeat(np.eye(4)[None], 80, axis=0)
        expected[:, 0, 3] = -1
        expected[:, 1, 3] = -2
        expected[:, 2, 3] = -3 - np.arange(index * 80 + 1, index * 80 + 81) * 0.01
        np.testing.assert_allclose(chunk.w2c, expected, atol=1e-6)
        assert chunk.intrinsics.shape == (80, 3, 3)


@pytest.mark.parametrize("frames", [0, 1, 80, 82, 800, 802])
def test_invalid_count(frames):
    with pytest.raises(ValueError):
        load_camera_path(payload(frames))


def test_seed_then10chunks_do_not_rebase_authored_future_path():
    async def run():
        app = Lyra2()
        app.state = Lyra2State()
        app.send = AsyncMock()
        app.output.flush = Mock()
        app.state._anchor = np.zeros((4, 4, 3), np.uint8)
        app.state.prompt = "scene"
        app.state._planner = Lyra2CameraPlanner(
            translation_per_frame=0.01, rotation_degrees_per_frame=0.1
        )
        await app.set_camera_path(
            UploadedFile(name="path.json", mime_type="application/json", data=payload())
        )
        inp = await app.process_input()
        assert inp.anchor is not None and inp.w2c is None
        seed = np.eye(4)
        seed[:3, 3] = [1, 2, 3]
        setup = Lyra2Result(inp.world_id, 0, None, seed, np.eye(3))
        assert await app.process_output(StepOutcome(result=setup, elapsed=0.1)) is None
        calibration = app._path_calibration.copy()
        with pytest.raises(CommandError):
            await app.release_camera()
        for index in range(10):
            a = await app.process_input()
            b = await app.process_input()
            np.testing.assert_array_equal(a.w2c, b.w2c)
            np.testing.assert_array_equal(app._path_calibration, calibration)
            corrected = np.eye(4)
            corrected[0, 3] = 100 + index
            out = Lyra2Result(
                inp.world_id,
                index + 1,
                np.zeros((80, 2, 2, 3), np.uint8),
                corrected,
                None,
            )
            await app.process_output(StepOutcome(result=out, elapsed=0.1))
        assert app._state().camera_path_complete
        with pytest.raises(ApplicationError):
            await app.process_input()
        await app.reset(42)
        assert app._path_calibration is None
        assert (await app.process_input()).anchor is not None

    asyncio.run(run())
