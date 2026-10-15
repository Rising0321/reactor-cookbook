"""Exercise real Runtime hooks with fake compute and the model's bookkeeping."""

import asyncio
import io
import sys
from concurrent.futures import Future, ThreadPoolExecutor
from pathlib import Path
from threading import Condition, Event
from types import ModuleType, SimpleNamespace

import numpy as np
import pytest
from PIL import Image
from reactor_runtime import ApplicationError, CommandError, StepOutcome, UploadedFile
from reactor_runtime.core.model import SessionStarted
from reactor_runtime.interface.model.contract import ModelContract

import edenlive_app as edenlive
from edenlive_app import EdenLive
from edenlive_game import CaptureClosed, EdenGame, event_lines
from edenlive_model import EdenInput, EdenModel, EdenResult, MissingAnchor
from edenlive_types import EdenState

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
    def __init__(self, **kwargs):
        self.index = 0
        self.keys = []
        self.closed = False
        self.held = ()
        self.pending = None
        self.paused = False

    def set_keys(self, keys):
        self.held = keys

    def set_paused(self, paused):
        self.paused = paused

    def prepare(self, frames):
        if self.pending is None:
            self.pending = frames

    def ready(self):
        return self.pending is not None

    def collect(self):
        frames, self.pending = self.pending, None
        self.keys.append(self.held)
        packet = SimpleNamespace(
            frames=tuple(
                SimpleNamespace(
                    ui_state={"roundId": 0},
                    game_frame_id=self.index + i + 1,
                    simulation_tick=(self.index + i + 1) * 2,
                )
                for i in range(frames)
            )
        )
        self.index += frames
        return (
            np.full((frames, 704, 1280, 3), 7, np.uint8),
            np.zeros((3, 704, 1280, 3), np.uint8),
            packet,
            (self.held,) * frames,
        )

    def close(self):
        self.closed = True


async def app(monkeypatch):
    monkeypatch.setattr(edenlive, "EdenGame", FakeGame)
    instance = EdenLive()
    model = EdenModel()
    model._backend = FakeBackend()
    instance._engine = model
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
    model.load(source=Path("/source"), weights=Path("/weights"),
               attention_backend="fa4", cuda_graph=True)
    assert calls == [dict(source=Path("/source"), weights=Path("/weights"),
                          attention_backend="fa4", cuda_graph=True, compile_model=False)]
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
        ("flash_attn.cute", dict(flash_attn_func=flash)),
        ("flashdreams.core.attention.cp", dict(ContextParallelAttention=Attention)),
        ("flashdreams.recipes.wan.transformer.impl.modules", dict(MultiHeadAttention=MultiHead)),
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
    assert output.main_video.shape == (9, 704, 2560, 3)
    assert np.all(output.main_video[:, :, :1280] == 7)
    assert np.all(output.main_video[:, :, 1280:] == 0)
    assert output.__metadata__["main_video"][0]["frame_id"] == 1
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
    await instance.on_disconnected()
    assert instance.state._started and instance.state._world_id == 1
    assert not instance.state.keys and model._backend.resets == 0
    await instance.process_input()
    assert instance._game.keys == [()]


async def test_restart_and_session_seed_isolation(monkeypatch):
    instance, model, _ = await ready(monkeypatch)
    first_game = instance._game
    await instance.start(seed=11)
    assert first_game.closed
    assert instance.state._world_id == 2
    assert (await instance.process_input()).anchor.seed == 11
    instance.on_session_ended()
    assert model._backend.resets == 1 and instance._game is None
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
    await instance.process_input()
    assert instance._game.keys == [()]
    assert instance.state._sampled_keys == ((),) * 9
    step = EdenResult(
        world_id=1,
        chunk_index=1,
        frames=np.zeros((9, 704, 1280, 3), np.uint8),
        frames_wanted=12,
    )
    await instance.process_output(StepOutcome(result=step))
    assert instance.state._sampled_keys == ()


async def test_terminal_game_pauses_until_fresh_start(monkeypatch):
    instance, _, _ = await ready(monkeypatch)
    snapshot = await instance.process_input()
    instance.state._packet.frames[-1].ui_state = {"state": "WON"}
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
                )
            )
        future = Future()
        future.set_result(SimpleNamespace(frames=packets))
        return future

    game.capture = SimpleNamespace(submit=submit)
    game.prepare(9)
    game.prepare(12)  # Completed captures also occupy the single slot.
    assert len(delivered) == 9
    _, _, _, sampled = game.collect()
    assert sampled == ((), ()) + (("f",),) * 7
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
    monkeypatch.setattr(instance._game, "ready", lambda: False)
    with pytest.raises(ApplicationError, match="draft chunk"):
        await instance.process_input()
    await instance.set_keys(keys=["w"])
    await instance.set_keys(keys=["d"])
    assert instance._game.held == ("d",)
    assert model._backend.steps == []


async def test_pipeline_starts_successor_before_model_and_keeps_pairing(monkeypatch):
    instance, model, _ = await ready(monkeypatch)
    first = await instance.process_input()
    assert instance._game.pending is None  # Wait for native first-result geometry.
    await instance.process_output(StepOutcome(result=instance.generate(first)))
    for chunk in range(2, 13):
        snapshot = await instance.process_input()
        assert instance._game.pending == 12  # Capture overlaps this generate.
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
        game.collect()
