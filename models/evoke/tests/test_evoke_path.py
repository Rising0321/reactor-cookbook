"""Native 36-pose windows preserve finite uploaded trajectories."""

import asyncio
import json

import numpy as np
import pytest
from evoke_camera import load_camera_path
from reactor_runtime import ApplicationError, CommandError, StepOutcome, UploadedFile
from test_evoke import _ready_model


def upload(chunks=10):
    poses = np.tile(np.eye(4), (36 * chunks, 1, 1))
    poses[:, 0, 3] = np.arange(len(poses)) * 0.02 + 3
    return UploadedFile(
        name="path.json",
        mime_type="application/json",
        data=json.dumps({"poses": poses.tolist()}).encode(),
    )


def test_native_pose_windows():
    file = upload()
    poses = np.asarray(json.loads(file.data)["poses"], dtype=np.float32)
    path = load_camera_path(file.data, 10)
    np.testing.assert_array_equal(
        path.chunks.reshape(-1, 4, 4), np.linalg.inv(poses[0]) @ poses
    )


def test_ten_chunks_stop_reset_and_controls():
    async def run():
        app, backend, _ = _ready_model()
        reply = await app.set_camera_path(upload())
        assert (reply.chunks, reply.frames) == (10, 360)
        with pytest.raises(CommandError):
            await app.set_yaw(1)
        expected = load_camera_path(upload().data, 10).chunks
        for index in range(10):
            value = await app.process_input()
            np.testing.assert_array_equal(value.trajectory, expected[index])
            np.testing.assert_array_equal(
                (await app.process_input()).trajectory, value.trajectory
            )
            result = app.generate(value)
            out = await app.process_output(StepOutcome(result=result))
            assert len(out.main_video) == 36
        assert len(backend.generate_calls) == 10
        with pytest.raises(ApplicationError, match="complete"):
            await app.process_input()
        await app.reset(42)
        np.testing.assert_array_equal(
            (await app.process_input()).trajectory, expected[0]
        )
        await app.start_text(prompt="Scene", seed=42)
        assert app._camera_path is None
        with pytest.raises(CommandError):
            await app.set_camera_path(upload())

    asyncio.run(run())


@pytest.mark.parametrize("count", [0, 1, 35, 37, 73])
def test_invalid_lengths(count):
    with pytest.raises(ValueError):
        load_camera_path(
            json.dumps({"poses": np.tile(np.eye(4), (count, 1, 1)).tolist()}).encode(),
            10,
        )
