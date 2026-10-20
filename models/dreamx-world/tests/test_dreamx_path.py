import asyncio
import json
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import AsyncMock, Mock

import numpy as np
import pytest
from dreamx_camera import DreamXCameraController, load_camera_path
from dreamx_types import DreamXWorldState
from dreamx_world import DreamXWorld
from dreamx_world_model import DreamXResult
from reactor_runtime import ApplicationError, CommandError, StepOutcome, UploadedFile


def payload(frames=117):
    poses = np.repeat(np.eye(4)[None], frames, axis=0)
    poses[:, 2, 3] = np.arange(frames) * 0.05
    return json.dumps({"poses": poses.tolist()}).encode()


def test_exact_native_latent_sampling_and_references():
    path = load_camera_path(payload(), 10)
    assert len(path.chunks) == 10 and path.frames == 117
    actual = np.concatenate([c.poses for c in path.chunks])
    expected = [0] + list(range(1, 114, 4))
    np.testing.assert_array_equal(actual[:, 0], expected)
    np.testing.assert_allclose(actual[:, 18], -np.array(expected) * 0.05, atol=1e-6)
    assert path.chunks[0].reference_pose is None
    for previous, current in zip(path.chunks, path.chunks[1:]):
        np.testing.assert_array_equal(current.reference_pose, previous.poses[-1])


@pytest.mark.parametrize("frames", [0, 8, 10, 12, 116, 118, 129])
def test_invalid_lengths(frames):
    with pytest.raises(ValueError):
        load_camera_path(payload(frames), 10)


def test_invalid_rotation():
    data = json.loads(payload(9))
    data["poses"][0][0][0] = 2
    with pytest.raises(ValueError, match="orthonormal"):
        load_camera_path(json.dumps(data).encode(), 10)


def app():
    result = DreamXWorld()
    result.state = DreamXWorldState()
    result._config = SimpleNamespace(max_chunks_per_rollout=10)
    result._camera = DreamXCameraController(1.5)
    result._selected_input = Path("uploaded.jpg")
    result.state.prompt = "A world"
    result.send = AsyncMock()
    result.output.flush = Mock()
    return result


def test_finite_path_retry_reset_and_key_rejection():
    async def run():
        model = app()
        await model.set_camera_path(
            UploadedFile(name="path.json", mime_type="application/json", data=payload())
        )
        with pytest.raises(CommandError):
            await model.set_key_state("w", True)
        for index in range(10):
            a = await model.process_input()
            b = await model.process_input()
            np.testing.assert_array_equal(a.poses, b.poses)
            assert (a.anchor is not None) == (index == 0)
            result = DreamXResult(
                np.zeros((9 if index == 0 else 12, 2, 2, 3), np.uint8),
                a.world_id,
                index + 1,
                a.prompt,
                index == 9,
            )
            await model.process_output(StepOutcome(result=result, elapsed=0.01))
        assert model._state_update().camera_path_complete
        with pytest.raises(ApplicationError):
            await model.process_input()
        await model.reset(42)
        restarted = await model.process_input()
        np.testing.assert_array_equal(
            restarted.poses, model._camera_path.chunks[0].poses
        )
        assert restarted.anchor is not None
        model._select_image(Path("new.jpg"), "uploaded", "new")
        assert model._camera_path is None

    asyncio.run(run())


def test_rejected_upload_leaves_active_path_and_world_unchanged():
    async def run():
        model = app()
        await model.set_camera_path(
            UploadedFile(name="ok.json", mime_type="application/json", data=payload())
        )
        selected = model._camera_path
        world = model.state._world_id
        with pytest.raises(CommandError):
            await model.set_camera_path(
                UploadedFile(
                    name="bad.json", mime_type="application/json", data=b"{broken"
                )
            )
        assert model._camera_path is selected
        assert model.state._world_id == world

    asyncio.run(run())


def test_prope_rounds_relative_pose_before_inverse():
    torch = pytest.importorskip("torch")
    from dreamx_backend import DreamXBackend

    path = load_camera_path(payload(), 10)
    backend = DreamXBackend.__new__(DreamXBackend)
    backend._device = torch.device("cpu")
    backend._dtype = torch.bfloat16
    for chunk in path.chunks:
        actual = backend._camera_condition(chunk)
        rows = (
            chunk.poses
            if chunk.reference_pose is None
            else np.concatenate([chunk.reference_pose[None], chunk.poses])
        )
        matrices = np.repeat(np.eye(4)[None], len(rows), axis=0)
        matrices[:, :3] = rows[:, 7:].reshape(-1, 3, 4)
        relative = np.stack(
            [np.eye(4)] + [matrices[0] @ np.linalg.inv(m) for m in matrices[1:]]
        ).astype(np.float32)
        if chunk.reference_pose is not None:
            relative = relative[1:]
        rounded = torch.tensor(relative, dtype=torch.bfloat16)
        rotation = rounded[:, :3, :3].transpose(-1, -2)
        expected = torch.zeros_like(rounded)
        expected[:, :3, :3] = rotation
        expected[:, :3, 3] = -torch.einsum("tij,tj->ti", rotation, rounded[:, :3, 3])
        expected[:, 3, 3] = 1
        assert torch.equal(actual["viewmats"][0, ::880], expected)


def test_keyboard_limit_preserves_world_without_automatic_restart():
    app = DreamXWorld()
    app.state = DreamXWorldState()
    app._config = SimpleNamespace(max_chunks_per_rollout=10)
    app._camera = DreamXCameraController(1.5)
    app._selected_input = Path("image.jpg")
    app.state.prompt = "A city"
    app.state._world_id = app.state._applied_world_id = 1
    app.send = AsyncMock()
    out = DreamXResult(
        world_id=1,
        chunk_index=10,
        frames=np.zeros((12, 2, 2, 3), np.uint8),
        prompt="A city",
        complete=True,
    )
    asyncio.run(app.process_output(StepOutcome(result=out, elapsed=1)))
    assert app.state._world_id == 1
    assert app._state_update().next_chunk is None
    assert not app._state_update().reset_queued
    with pytest.raises(ApplicationError, match="Rollout complete"):
        asyncio.run(app.process_input())
    assert not any(
        type(call.args[0]).__name__ == "RolloutResetQueued"
        for call in app.send.call_args_list
    )
