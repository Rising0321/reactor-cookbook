import queue
from types import SimpleNamespace

import numpy as np

from liveavatar_parallel import ParallelBackend


def test_parallel_delivery_keeps_48khz_and_one_clip_demand(tmp_path):
    backend = ParallelBackend.__new__(ParallelBackend)
    backend.directory = SimpleNamespace(name=str(tmp_path))
    backend.active = True
    backend.pending_ack = False
    backend.results = queue.Queue()
    backend.ack = queue.Queue()
    backend.audio = np.arange(48000 * 4, dtype=np.float32)
    backend.offset = 0
    backend.processes = []
    np.save(tmp_path / "chunk.npy", np.zeros((45, 2, 2, 3), np.uint8))
    backend.results.put(("chunk", 45))
    video, audio = backend.next()
    assert len(video) == 45 and audio.shape == (1, 86400)
    assert backend.ack.empty()
    assert backend.pending_ack
    backend.results.put(("end",))
    assert backend.next() is None
    assert backend.ack.get_nowait() is True
    assert not backend.active
    assert backend.offset == 86400


def test_idle_close_retains_workers():
    backend = ParallelBackend.__new__(ParallelBackend)
    backend.active = False
    backend.processes = [object()]
    backend.close()
    assert len(backend.processes) == 1


def test_shutdown_is_repeatable_without_workers():
    import multiprocessing as mp

    backend = ParallelBackend.__new__(ParallelBackend)
    backend.processes = []
    backend.directory = None
    context = mp.get_context("spawn")
    backend.commands = context.Queue(maxsize=1)
    backend.results = context.Queue(maxsize=2)
    backend.ack = context.Queue(maxsize=1)
    backend.shutdown()
    backend.shutdown()
    assert not backend.active
    assert not backend.pending_ack
