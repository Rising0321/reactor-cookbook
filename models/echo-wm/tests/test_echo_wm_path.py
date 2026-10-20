"""Finite camera paths preserve native sampling and causal progress."""

import asyncio
import json

import numpy as np
import pytest
from echo_wm_camera import load_camera_path
from reactor_runtime import ApplicationError, CommandError, UploadedFile
from test_echo_wm import _selected_app, _step


def upload(chunks=12):
    poses = np.tile(np.eye(4), (24 * chunks + 1, 1, 1))
    poses[:, 0, 3] = np.arange(len(poses)) * 0.05
    return UploadedFile(
        name="path.json",
        mime_type="application/json",
        data=json.dumps({"poses": poses.tolist()}).encode(),
    )


def test_native_sampling_and_anchor():
    trajectory = upload()
    data = json.loads(trajectory.data)
    poses = np.asarray(data["poses"], dtype=np.float32)
    poses[:, 2, 3] += 2
    parsed = load_camera_path(json.dumps({"poses": poses.tolist()}).encode(), 12)
    expected = np.linalg.inv(poses[0]) @ poses[::8]
    np.testing.assert_allclose(parsed.chunks.reshape(-1, 4, 4), expected[1:])
    assert parsed.frames == 289


def test_finite_path_steps_reset_and_controls():
    async def run():
        app, engine, messages = _selected_app(12)
        result = await app.set_camera_path(upload())
        assert (result.chunks, result.frames) == (12, 289)
        with pytest.raises(CommandError):
            await app.set_camera_motion(forward=1, strafe=0, yaw=0, pitch=0)
        expected = load_camera_path(upload().data, 12).chunks
        for index in range(12):
            inp = await app.process_input()
            assert inp.rollout_chunks == 12
            np.testing.assert_array_equal(inp.poses, expected[index])
            # Preparing an input does not consume the uploaded trajectory.
            np.testing.assert_array_equal((await app.process_input()).poses, inp.poses)
            out = await _step(app)
            assert len(out.main_video) == (25 if index == 0 else 24)
        assert len(engine.inputs) == 12
        assert not any(type(x).__name__ == "AutomaticResetQueued" for x in messages)
        with pytest.raises(ApplicationError, match="complete"):
            await app.process_input()
        await app.reset(seed=42)
        np.testing.assert_array_equal((await app.process_input()).poses, expected[0])

    asyncio.run(run())


@pytest.mark.parametrize("count", [0, 1, 24, 26, 50])
def test_invalid_lengths(count):
    poses = np.tile(np.eye(4), (count, 1, 1))
    with pytest.raises(ValueError):
        load_camera_path(json.dumps({"poses": poses.tolist()}).encode(), 12)


def test_invalid_rotation_does_not_change_active_path():
    async def run():
        app, _, _ = _selected_app()
        await app.set_camera_path(upload())
        active = app._camera_path
        bad = json.loads(upload().data)
        bad["poses"][1][0][0] = 2
        with pytest.raises(CommandError):
            await app.set_camera_path(
                UploadedFile(
                    name="bad.json",
                    mime_type="application/json",
                    data=json.dumps(bad).encode(),
                )
            )
        assert app._camera_path is active

    asyncio.run(run())
