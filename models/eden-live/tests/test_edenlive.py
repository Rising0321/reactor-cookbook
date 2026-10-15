# Copyright (c) 2026 Reactor Technologies, Inc. All rights reserved.
"""Exercise real Runtime hooks with fake compute and the model's bookkeeping."""

import asyncio
import io
import sys
from concurrent.futures import Future, ThreadPoolExecutor
from pathlib import Path
from threading import Condition, Event
from types import ModuleType, SimpleNamespace

import edenlive_app as edenlive
import numpy as np
import pytest
from edenlive_app import EdenLive
from edenlive_draft import ClientSource, Draft, fit_frames
from edenlive_game import CaptureClosed, EdenGame, event_lines, reached_outcome
from edenlive_model import (
    CONDITION_STRIDE,
    EdenInput,
    EdenModel,
    MissingAnchor,
    condition_frames,
)
from edenlive_types import EdenState
from PIL import Image
from reactor_runtime import ApplicationError, CommandError, ReadMode, StepOutcome, UploadedFile
from reactor_runtime.core.model import SessionStarted
from reactor_runtime.interface.model.contract import ModelContract

pytestmark = pytest.mark.asyncio


class FakeBackend:
    def __init__(self):
        self.resets = 0
        self.steps = []
        self.anchors = []
        self.error = None

    def reset(self):
        self.resets += 1

    def start(self, anchor):
        self.anchors.append(anchor)

    def step(self, index, condition, prompt):
        self.steps.append((index, condition, prompt))
        if self.error:
            raise self.error
        return np.zeros((9 if index == 0 else 12, 704, 1280, 3), np.uint8)


class FakeGame:
    kind = "game"

    def __init__(self, **kwargs):
        self.slug = kwargs.get("slug")
        self.index = 0
        self.keys = []
        self.closed = False
        self.held = ()
        self.pending = None
        self.paused = False
        self.outcome = False

    def set_keys(self, keys):
        self.held = keys

    def set_paused(self, paused):
        self.paused = paused

    def prepare(self, frames):
        if self.pending is None:
            self.pending = frames

    def ready(self):
        return self.pending is not None

    async def take(self):
        if not self.ready():
            return None
        frames, self.pending = self.pending, None
        self.keys.append(self.held)
        metadata = tuple(
            {
                "source": "game",
                "frame_id": self.index + i + 1,
                "simulation_tick": (self.index + i + 1) * 2,
                "keys": list(self.held),
            }
            for i in range(frames)
        )
        self.index += frames
        return Draft(
            frames=np.full((frames, 704, 1280, 3), 7, np.uint8),
            metadata=metadata,
            ended=self.outcome,
        )

    def close(self):
        self.closed = True


class FakeTrack:
    """A client `source_video` track holding frames the test pushes."""

    def __init__(self):
        self.frames = []
        self.reads = []

    def push(self, count, value=5, shape=(720, 1280, 3)):
        for _ in range(count):
            self.frames.append(
                SimpleNamespace(data=np.full(shape, value, np.uint8), pts=len(self.frames) / 24)
            )

    def try_read(self, n, mode=None):
        self.reads.append((n, mode))
        if len(self.frames) < n:
            return None
        latest, self.frames = self.frames[-n:], []
        return latest


async def app(monkeypatch):
    monkeypatch.setattr(edenlive, "EdenGame", FakeGame)
    instance = EdenLive()
    model = EdenModel()
    model._backend = FakeBackend()
    instance._engine = model
    instance.media = SimpleNamespace(source_video=FakeTrack())
    instance._on_loop_ready()
    sent = []
    instance.bind_output(
        broadcast=sent.append, addressed=lambda *args: None, media=lambda chunk: None
    )
    await instance._dispatch_reactor_event(SessionStarted("test"))
    return instance, model, sent


async def ready(monkeypatch):
    instance, model, sent = await app(monkeypatch)
    instance.state._game = "freeze"
    instance.state._image = np.zeros((64, 64, 3), np.uint8)
    instance.state.prompt = "A cavern."
    await instance.start(seed=7)
    return instance, model, sent


async def test_schema_and_runtime_free_model_boundary():
    contract = ModelContract.of(EdenLive)
    assert set(contract.commands) == {
        "set_source",
        "select_game",
        "set_image",
        "start",
        "set_prompt",
        "set_keys",
        "set_paused",
    }
    for name in ("edenlive_model.py", "edenlive_backend.py", "edenlive_attention.py"):
        assert "reactor_runtime" not in (Path(__file__).parents[1] / name).read_text()


async def test_model_passes_execution_settings_without_loading_runtime(monkeypatch):
    calls = []
    stub = ModuleType("edenlive_backend")
    stub.EdenBackend = lambda **kwargs: calls.append(kwargs) or FakeBackend()
    monkeypatch.setitem(sys.modules, "edenlive_backend", stub)
    model = EdenModel()
    model.load(
        source=Path("/source"),
        weights=Path("/weights"),
        attention_backend="fa4",
        cuda_graph=True,
    )
    assert calls == [
        {
            "source": Path("/source"),
            "weights": Path("/weights"),
            "attention_backend": "fa4",
            "cuda_graph": True,
            "compile_model": False,
        }
    ]
    assert model._chunk_index == 0 and model._world_id is None


async def test_invalid_execution_options_fail_before_asset_loading(tmp_path):
    from edenlive_assets import resolve_assets

    config = tmp_path / "edenlive.yaml"
    config.write_text("attention_backend: sparse\n")
    with pytest.raises(ValueError, match="attention_backend"):
        resolve_assets(config, tmp_path)
    config.write_text('cuda_graph: "yes"\n')
    with pytest.raises(ValueError, match="boolean"):
        resolve_assets(config, tmp_path)
    config.write_text("game_capture: [832]\n")
    with pytest.raises(ValueError, match="game_capture"):
        resolve_assets(config, tmp_path)


async def test_fa4_dense_layout_is_local_and_rejects_multiple_gpus(monkeypatch):
    from edenlive_attention import install_flash_attention4

    calls = []

    class Tensor:
        def __init__(self, name, transposes=()):
            self.name, self.transposes = name, transposes

        def transpose(self, a, b):
            return Tensor(self.name, self.transposes + ((a, b),))

    class Attention:
        def __init__(self, qkv_format="bshd"):
            self.qkv_format = qkv_format
            self.device_mesh = None

        def is_context_parallel_enabled(self):
            return self.device_mesh is not None

    class MultiHead:
        def __init__(self):
            self.attn_op = Attention()

    def flash(query, key, value, **kwargs):
        calls.append((query, key, value, kwargs))
        return Tensor("output"), None

    for name, attributes in (
        ("flash_attn.cute", {"flash_attn_func": flash}),
        ("flashdreams.core.attention.cp", {"ContextParallelAttention": Attention}),
        (
            "flashdreams.recipes.wan.transformer.impl.modules",
            {"MultiHeadAttention": MultiHead},
        ),
    ):
        module = ModuleType(name)
        module.__dict__.update(attributes)
        monkeypatch.setitem(sys.modules, name, module)
    first, second = MultiHead(), MultiHead()
    untouched = second.attn_op
    install_flash_attention4(SimpleNamespace(modules=lambda: (first,)))
    assert second.attn_op is untouched
    output = first.attn_op._impl(Tensor("q"), Tensor("k"), Tensor("v"))
    assert all(t.transposes == ((1, 2),) for t in calls[0][:3])
    assert calls[0][3] == {"causal": False}
    assert output.transposes == ((1, 2),)
    first.attn_op.device_mesh = object()
    with pytest.raises(RuntimeError, match="single-GPU"):
        first.attn_op._impl(Tensor("q"), Tensor("k"), Tensor("v"))


async def test_refused_step_never_calls_model(monkeypatch):
    instance, model, _ = await app(monkeypatch)
    with pytest.raises(ApplicationError):
        await instance.process_input()
    assert not model._backend.steps


async def test_anchor_ack_prompt_and_frame_pairing(monkeypatch):
    instance, model, _ = await ready(monkeypatch)
    first = await instance.process_input()
    assert first.anchor.seed == 7
    result = instance.generate(first)
    output = await instance.process_output(StepOutcome(result=result, elapsed=0.5))
    assert output.main_video.shape == output.draft_video.shape == (9, 704, 1280, 3)
    assert np.all(output.draft_video == 7)
    assert np.all(output.main_video == 0)
    assert output.__metadata__["main_video"][0]["frame_id"] == 1
    assert output.__metadata__["draft_video"] == output.__metadata__["main_video"]
    await instance.set_prompt(prompt="The cavern glows blue.")
    second = await instance.process_input()
    assert second.anchor is None and second.world_id == first.world_id
    assert second.prompt == "The cavern glows blue."
    instance.state.prompt = "A later command cannot mutate a captured step."
    instance.generate(second)
    assert model._backend.steps[-1][2] == second.prompt
    assert len(model._backend.anchors) == 1


async def test_failures_do_not_complete_or_reset(monkeypatch):
    instance, model, _ = await ready(monkeypatch)
    input = await instance.process_input()
    model._backend.error = RuntimeError("test compute failure")
    with pytest.raises(RuntimeError) as error:
        instance.generate(input)
    with pytest.raises(RuntimeError, match="test compute failure"):
        await instance.process_output(StepOutcome(error=error.value))
    assert instance.state._chunks == model._chunk_index == 0
    assert instance.state._applied_world_id is None
    assert model._backend.resets == 1


async def test_pause_update_resume_and_disconnect(monkeypatch):
    instance, model, _ = await ready(monkeypatch)
    await instance.set_keys(keys=["w", "f"])
    await instance.set_paused(paused=True)
    await instance.set_prompt(prompt="Blue ice.")
    with pytest.raises(ApplicationError):
        await instance.process_input()
    await instance.set_paused(paused=False)
    await instance.set_keys(keys=["w"], client=SimpleNamespace(id="player"))
    await instance.on_disconnected(SimpleNamespace(id="viewer"))
    assert instance.state.keys == ["w"]
    await instance.on_disconnected(SimpleNamespace(id="player"))
    assert instance.state._started and instance.state._world_id == 1
    assert not instance.state.keys and model._backend.resets == 0
    await instance.process_input()
    assert instance._draft_source.keys == [()]


async def test_restart_and_session_seed_isolation(monkeypatch):
    instance, model, _ = await ready(monkeypatch)
    first_game = instance._draft_source
    await instance.start(seed=11)
    assert first_game.closed
    assert instance.state._world_id == 2
    assert (await instance.process_input()).anchor.seed == 11
    instance.on_session_ended()
    assert model._backend.resets == 1 and instance._draft_source is None
    assert EdenState()._seed == 2026092901


async def test_refused_commands_send_nothing(monkeypatch):
    instance, _, sent = await app(monkeypatch)
    for method, kwargs, code in (
        (instance.start, {"seed": 0}, "game_required"),
        (instance.set_prompt, {"prompt": " "}, "prompt_required"),
        (instance.set_paused, {"paused": False}, "start_required"),
    ):
        before = len(sent)
        with pytest.raises(CommandError) as error:
            await method(**kwargs)
        assert error.value.code == code and len(sent) == before


async def test_upload_stages_without_generation(monkeypatch):
    instance, model, _ = await app(monkeypatch)
    instance.state._game = "freeze"
    data = io.BytesIO()
    Image.new("RGB", (64, 64)).save(data, format="PNG")
    await instance.set_image(
        UploadedFile(name="cover.png", mime_type="image/png", data=data.getvalue())
    )
    assert instance.state._image.shape == (64, 64, 3)
    assert not instance.state._started and not model._backend.steps
    for payload in (b"", b"broken", b"a" * (25 * 1024 * 1024 + 1)):
        with pytest.raises(CommandError) as error:
            await instance.set_image(
                UploadedFile(name="bad.png", mime_type="image/png", data=payload)
            )
        assert error.value.code == "invalid_image"


async def test_model_missing_anchor():
    model = EdenModel()
    model._backend = FakeBackend()
    with pytest.raises(MissingAnchor):
        model.generate(
            EdenInput(
                world_id=1,
                anchor=None,
                prompt="Scene",
                condition=np.zeros((3, 704, 1280, 3), np.uint8),
            )
        )
    assert model._backend.resets == 0


async def test_event_prompt_matches_native_dedup():
    chunk = SimpleNamespace(
        frames=[
            SimpleNamespace(
                ui_state={
                    "events": [
                        {"id": 2, "type": "shot", "prompt": " Blue ice. "},
                        {"id": 1, "type": "move", "prompt": "Moving."},
                        {"id": 3, "type": "shot", "prompt": "Blue ice."},
                    ]
                }
            )
        ]
    )
    assert event_lines(chunk) == ("Moving.", "Blue ice.")


async def test_actual_runtime_loop(monkeypatch):
    instance, model, _ = await ready(monkeypatch)
    emitted = []

    async def emit(output, **kwargs):
        emitted.append(output)
        instance._live.clear()

    monkeypatch.setattr(instance, "emit", emit)
    monkeypatch.setattr(instance, "_report_step", lambda *args: None)
    instance._live.set()
    task = asyncio.create_task(instance.run())
    for _ in range(100):
        if emitted:
            break
        await asyncio.sleep(0.01)
    task.cancel()
    with pytest.raises(asyncio.CancelledError):
        await task
    assert len(emitted) == 1 and len(model._backend.steps) == 1
    assert instance.state._chunks == 1


async def test_latest_keys_replace_unconsumed_transitions(monkeypatch):
    instance, _, _ = await ready(monkeypatch)
    await instance.set_keys(keys=["f"])
    await instance.set_keys(keys=[])
    snapshot = await instance.process_input()
    assert instance._draft_source.keys == [()]
    assert [frame["keys"] for frame in instance.state._draft.metadata] == [[]] * 9
    output = await instance.process_output(StepOutcome(result=instance.generate(snapshot)))
    assert output.__metadata__["main_video"][0]["keys"] == []
    assert instance.state._draft is None


async def test_terminal_game_pauses_until_fresh_start(monkeypatch):
    instance, _, _ = await ready(monkeypatch)
    instance._draft_source.outcome = True
    snapshot = await instance.process_input()
    await instance.process_output(StepOutcome(result=instance.generate(snapshot)))
    assert instance.state.paused and instance.state._ended
    with pytest.raises(ApplicationError):
        await instance.process_input()
    with pytest.raises(CommandError) as error:
        await instance.set_paused(paused=False)
    assert error.value.code == "game_ended"
    await instance.start(seed=7)
    assert not instance.state._ended


async def test_actual_runtime_refusal_loop(monkeypatch):
    instance, model, _ = await app(monkeypatch)
    attempts = []
    original = instance.process_input

    async def observe():
        attempts.append(True)
        return await original()

    monkeypatch.setattr(instance, "process_input", observe)
    monkeypatch.setattr(instance, "_report_step", lambda *args: None)
    instance._live.set()
    task = asyncio.create_task(instance.run())
    for _ in range(100):
        if attempts:
            break
        await asyncio.sleep(0.01)
    task.cancel()
    with pytest.raises(asyncio.CancelledError):
        await task
    assert attempts and not model._backend.steps


async def test_actual_runtime_failure_loop(monkeypatch):
    instance, model, _ = await ready(monkeypatch)
    model._backend.error = RuntimeError("inference failed")
    monkeypatch.setattr(instance, "_report_step", lambda *args: None)
    instance._live.set()
    with pytest.raises(RuntimeError, match="inference failed"):
        await asyncio.wait_for(instance.run(), timeout=5)
    assert instance.state._chunks == 0
    assert instance.state._applied_world_id is None


def bare_game():
    game = EdenGame.__new__(EdenGame)
    game._held, game._latest, game._sampled = set(), frozenset(), []
    game._gate, game._paused, game._closed = Condition(), False, False
    game._pending = None
    game._submitted_at = game._captured_at = None
    game.timeline = SimpleNamespace(begin_chunk=lambda frames, fps: frames)
    return game


async def test_capture_samples_latest_state_per_frame_without_replaying_taps():
    game = bare_game()
    delivered = []

    def submit(count, drain_events):
        packets = []
        for i in range(count):
            if i == 1:
                game.set_keys(("w",))
                game.set_keys(("a",))
                game.set_keys(())
            if i == 2:
                game.set_keys(("f",))
            delivered.append(drain_events())
            pixels = np.zeros((2, 2, 3), np.uint8)
            packets.append(
                SimpleNamespace(
                    ui_frame=pixels,
                    condition_frame=pixels,
                    capture_condition=i in (0, 4, 8),
                    ui_state={"state": "WON"} if i == count - 1 else {},
                    game_frame_id=i + 1,
                    simulation_tick=2 * (i + 1),
                )
            )
        future = Future()
        future.set_result(SimpleNamespace(frames=packets))
        return future

    game.capture = SimpleNamespace(submit=submit)
    game.prepare(9)
    game.prepare(12)  # Completed captures also occupy the single slot.
    assert len(delivered) == 9
    draft = await game.take()
    assert [frame["keys"] for frame in draft.metadata] == [[], []] + [["f"]] * 7
    assert draft.frames.shape == (9, 704, 1280, 3) and draft.ended
    assert delivered[0] == []
    assert delivered[1] == []
    assert delivered[2] == [{"action": "f", "phase": "down"}]
    assert game._held == {"f"}

    game._held = {"w"}
    game.set_keys(("f",))
    assert game._drain() == [
        {"action": "w", "phase": "up"},
        {"action": "f", "phase": "down"},
    ]
    assert game._held == {"f"}


async def test_capture_wait_refuses_without_holding_runtime_lock(monkeypatch):
    instance, model, _ = await ready(monkeypatch)
    monkeypatch.setattr(instance._draft_source, "ready", lambda: False)
    with pytest.raises(ApplicationError, match="draft frames"):
        await instance.process_input()
    await instance.set_keys(keys=["w"])
    await instance.set_keys(keys=["d"])
    assert instance._draft_source.held == ("d",)
    assert model._backend.steps == []


async def test_capture_waits_for_the_step_and_keeps_pairing(monkeypatch):
    instance, model, _ = await ready(monkeypatch)
    first = await instance.process_input()
    assert instance._draft_source.pending is None
    await instance.process_output(StepOutcome(result=instance.generate(first)))
    for chunk in range(2, 13):
        snapshot = await instance.process_input()
        # Nothing is captured ahead, so keys set between steps reach the next chunk.
        assert instance._draft_source.pending is None
        result = instance.generate(snapshot)
        output = await instance.process_output(StepOutcome(result=result))
        metadata = output.__metadata__["main_video"]
        assert metadata[0]["frame_id"] == 10 + (chunk - 2) * 12
        assert metadata[-1]["frame_id"] == 9 + (chunk - 1) * 12
        assert output.main_video.shape[0] == 12
    assert len(model._backend.steps) == 12 and len(model._backend.anchors) == 1


async def test_pause_holds_capture_and_close_wakes_it():
    game = bare_game()
    game.set_paused(True)
    entered = Event()

    def drain():
        entered.set()
        return game._drain()

    with ThreadPoolExecutor(max_workers=1) as pool:
        pending = pool.submit(drain)
        assert entered.wait(1) and not pending.done()
        game.set_keys(("w",))
        game.set_paused(False)
        assert pending.result(1) == [{"action": "w", "phase": "down"}]
        game.set_paused(True)
        entered.clear()
        pending = pool.submit(drain)
        assert entered.wait(1) and not pending.done()
        game.capture = SimpleNamespace(close=lambda: None)
        game.close()
        with pytest.raises(CaptureClosed):
            pending.result(1)


async def test_capture_failure_is_not_hidden():
    game = bare_game()
    game._pending = Future()
    game._pending.set_exception(RuntimeError("capture failed"))
    with pytest.raises(RuntimeError, match="capture failed"):
        await game.take()


async def test_game_refuses_a_condition_layout_the_shared_slice_would_miss():
    game = bare_game()
    game._sampled = [()] * 9
    future = Future()
    frame = SimpleNamespace(
        ui_frame=np.zeros((2, 2, 3), np.uint8), ui_state={}, game_frame_id=1, simulation_tick=1
    )
    frames = [SimpleNamespace(**vars(frame), capture_condition=i in (1, 5, 9)) for i in range(9)]
    future.set_result(SimpleNamespace(frames=frames))
    game._pending = future
    with pytest.raises(RuntimeError, match="every fourth"):
        await game.take()


async def test_hud_outcome_detection():
    assert not reached_outcome({"state": "RUNNING"})
    assert not reached_outcome({"roundId": 0})
    assert not reached_outcome(None)
    assert reached_outcome({"state": "LOST"})


async def test_condition_frames_are_the_shared_stride():
    draft = np.arange(12)[:, None, None, None] * np.ones((12, 2, 2, 3), np.uint8)
    assert CONDITION_STRIDE == 4
    assert [int(f[0, 0, 0]) for f in condition_frames(draft)] == [0, 4, 8]
    assert len(condition_frames(draft[:9])) == 3


async def test_fit_frames_cover_crops_any_client_geometry():
    tall = np.zeros((1080, 720, 3), np.uint8)
    tall[540, 360] = 255
    wide = np.full((480, 1920, 3), 9, np.uint8)
    exact = np.full((704, 1280, 3), 3, np.uint8)
    fitted = fit_frames([tall, wide, exact])
    assert fitted.shape == (3, 704, 1280, 3) and fitted.dtype == np.uint8
    assert np.all(fitted[1] == 9) and np.all(fitted[2] == 3)


async def test_client_source_reads_latest_frames_and_never_ends():
    track = FakeTrack()
    source = ClientSource(track)
    assert await source.take() is None
    source.prepare(9)
    track.push(5)
    assert await source.take() is None
    track.push(20, value=8, shape=(480, 640, 3))
    draft = await source.take()
    assert draft.frames.shape == (9, 704, 1280, 3) and np.all(draft.frames == 8)
    assert draft.metadata[0]["source"] == "client" and not draft.ended and draft.events == ()
    assert track.reads[-1][1] is ReadMode.LATEST


async def client_ready(monkeypatch):
    instance, model, sent = await app(monkeypatch)
    await instance.set_source(source="client")
    data = io.BytesIO()
    Image.new("RGB", (64, 64)).save(data, format="PNG")
    await instance.set_image(
        UploadedFile(name="look.png", mime_type="image/png", data=data.getvalue())
    )
    await instance.set_prompt(prompt="A neon city.")
    await instance.start(seed=3)
    return instance, model, sent


async def test_client_source_goes_through_the_same_step(monkeypatch):
    instance, model, sent = await client_ready(monkeypatch)
    track = instance.media.source_video
    with pytest.raises(ApplicationError, match="from the client"):
        await instance.process_input()
    track.push(9, value=40)
    first = await instance.process_input()
    assert first.anchor.seed == 3 and first.prompt == "A neon city."
    assert np.all(first.condition == 40) and first.condition.shape == (3, 704, 1280, 3)
    output = await instance.process_output(StepOutcome(result=instance.generate(first)))
    assert np.all(output.draft_video == 40)
    assert output.__metadata__["main_video"][0]["source"] == "client"
    assert sent[-1].source == "client" and sent[-1].chunks == 1
    track.push(12, value=41)
    second = await instance.process_input()
    assert second.anchor is None and np.all(second.condition == 41)


async def test_switching_source_mid_world_keeps_the_world(monkeypatch):
    instance, model, _ = await ready(monkeypatch)
    first = await instance.process_input()
    await instance.process_output(StepOutcome(result=instance.generate(first)))
    game = instance._draft_source
    await instance.set_source(source="client")
    assert game.closed and isinstance(instance._draft_source, ClientSource)
    instance.media.source_video.push(12, value=50)
    step = await instance.process_input()
    assert step.anchor is None and step.world_id == first.world_id
    assert np.all(step.condition == 50)
    await instance.process_output(StepOutcome(result=instance.generate(step)))
    await instance.set_source(source="game")
    assert isinstance(instance._draft_source, FakeGame) and instance._draft_source is not game
    step = await instance.process_input()
    assert step.anchor is None and np.all(step.condition == 7)
    assert len(model._backend.anchors) == 1 and model._backend.resets == 1


async def test_switching_away_from_an_ended_game_allows_resume(monkeypatch):
    instance, _, _ = await ready(monkeypatch)
    instance._draft_source.outcome = True
    snapshot = await instance.process_input()
    await instance.process_output(StepOutcome(result=instance.generate(snapshot)))
    assert instance.state._ended
    await instance.set_source(source="client")
    await instance.set_paused(paused=False)
    assert not instance.state._ended and not instance.state.paused


async def test_source_refusals(monkeypatch):
    instance, _, sent = await app(monkeypatch)
    await instance.set_source(source="client")
    with pytest.raises(CommandError) as error:
        await instance.start(seed=0)
    assert error.value.code == "image_required"
    instance, _, _ = await client_ready(monkeypatch)
    with pytest.raises(CommandError) as error:
        await instance.set_source(source="game")
    assert error.value.code == "game_required"
    assert instance.state._source == "client"


async def test_warmup_covers_capture_recapture_and_a_fresh_world(monkeypatch):
    backend = FakeBackend()
    backend.capture_index, backend.graph_warmup = 6, 2
    stub = ModuleType("edenlive_backend")
    stub.EdenBackend = lambda **kwargs: backend
    monkeypatch.setitem(sys.modules, "edenlive_backend", stub)
    model = EdenModel()
    model.load(source=Path("/source"), weights=Path("/weights"), warmup=True)
    steady, recapture = 6 + 2 + 2, 2 + 2
    assert [index for index, _, _ in backend.steps] == [
        *range(steady + recapture),
        *range(steady),
    ]
    prompts = [prompt for _, _, prompt in backend.steps[: steady + recapture]]
    assert len(set(prompts[:steady])) == 1 and prompts[steady] != prompts[0]
    assert [anchor.seed for anchor in backend.anchors] == [0, 1]
    assert backend.resets == 3
    assert model._world_id is None and model._chunk_index == 0


async def test_load_skips_warmup_unless_asked(monkeypatch):
    backend = FakeBackend()
    stub = ModuleType("edenlive_backend")
    stub.EdenBackend = lambda **kwargs: backend
    monkeypatch.setitem(sys.modules, "edenlive_backend", stub)
    EdenModel().load(source=Path("/source"), weights=Path("/weights"))
    assert backend.steps == []


async def test_chunk_timer_splits_the_cycle_and_summarizes():
    from edenlive_timing import SUMMARY_EVERY, ChunkTimer

    timer = ChunkTimer()
    assert (
        timer.finished(1.0, frames=9, generate_s=0.3, capture_s=None, prompt_changed=True) is None
    )
    at = 1.0
    for _ in range(SUMMARY_EVERY):
        timer.attempt(at + 0.5)  # emit blocked 0.5 s
        timer.attempt(at + 0.6)  # a refused attempt does not move the start
        timer.drafted(at + 0.7)  # draft arrived 0.2 s after the first attempt
        at += 1.0
        sample = timer.finished(at, frames=12, generate_s=0.25, capture_s=0.6, prompt_changed=False)
        assert sample["cycle_s"] == pytest.approx(1.0)
        assert sample["emit_s"] == pytest.approx(0.5)
        assert sample["draft_wait_s"] == pytest.approx(0.2)
    summary = timer.summary()
    assert summary["new_fps"] == pytest.approx(12.0) and summary["chunks"] == SUMMARY_EVERY
    assert summary["generate_s"] == 0.25 and summary["capture_s"] == 0.6
    assert timer.summary() is None
    timer.restart()
    assert (
        timer.finished(9.0, frames=12, generate_s=0.2, capture_s=0.5, prompt_changed=False) is None
    )


async def test_chunk_timer_summary_covers_one_run():
    from edenlive_timing import SUMMARY_EVERY, ChunkTimer

    def run(timer, at, chunks, capture_s):
        timer.finished(at, frames=12, generate_s=0.2, capture_s=capture_s, prompt_changed=False)
        for _ in range(chunks):
            timer.attempt(at + 0.1)
            timer.drafted(at + 0.2)
            at += 0.5
            timer.finished(at, frames=12, generate_s=0.2, capture_s=capture_s, prompt_changed=False)
        return at

    timer = ChunkTimer()
    at = run(timer, 0.0, SUMMARY_EVERY - 1, capture_s=0.6)
    timer.restart()  # e.g. `set_source` to `client`
    at = run(timer, at + 5.0, 1, capture_s=None)
    assert timer.summary() is None
    run(timer, at, SUMMARY_EVERY - 1, capture_s=None)
    summary = timer.summary()
    assert summary["chunks"] == SUMMARY_EVERY and summary["capture_s"] is None
