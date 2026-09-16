"""A demand-driven bridge to the released single-GPU four-step pipeline."""

from __future__ import annotations

import logging
import os
import queue
import sys
import threading
from pathlib import Path

import numpy as np

from liveavatar_assets import SOURCE, WORK, prepare_assets
from liveavatar_audio import OUTPUT_SAMPLE_RATE, playback_audio


class GenerationCancelled(Exception):
    pass


class ClipBridge:
    """Bound producer memory to one clip and retain one inference call per take."""

    def __init__(self, produce):
        self._produce = produce
        self._queue = queue.Queue(maxsize=1)
        self._demand = threading.Semaphore(0)
        self._cancel = threading.Event()
        self._first = True
        self._thread = threading.Thread(
            target=self._run, name="liveavatar-take", daemon=True
        )
        self._thread.start()

    def _emit(self, clip):
        if self._cancel.is_set():
            raise GenerationCancelled()
        self._queue.put(clip)
        self._demand.acquire()
        if self._cancel.is_set():
            raise GenerationCancelled()

    def _run(self):
        try:
            self._produce(self._emit)
            result = None
        except GenerationCancelled:
            return
        except Exception as error:
            logging.getLogger(__name__).exception(
                "LiveAvatar upstream generation failed"
            )
            result = error
        while not self._cancel.is_set():
            try:
                self._queue.put(result, timeout=0.1)
                return
            except queue.Full:
                continue

    def next(self):
        if not self._first:
            self._demand.release()
        self._first = False
        result = self._queue.get()
        if isinstance(result, Exception):
            raise result
        return result

    def close(self):
        self._cancel.set()
        self._demand.release()
        # Let the current CUDA call finish; never terminate another process.
        self._thread.join()


class LiveAvatarBackend:
    def __init__(self):
        base, lora = prepare_assets()
        sys.path.insert(0, str(SOURCE))
        import torch
        import torch.distributed as dist
        import yaml
        from liveavatar.models.wan.causal_s2v_pipeline import WanS2V
        from liveavatar.models.wan.wan_2_2.configs import WAN_CONFIGS

        torch.cuda.set_device(0)
        if not dist.is_initialized():
            import tempfile

            fd, path = tempfile.mkstemp(prefix="distributed-", dir=WORK)
            os.close(fd)
            os.unlink(path)
            dist.init_process_group(
                "nccl", init_method=f"file://{path}", rank=0, world_size=1
            )
        self.pipeline = WanS2V(
            config=WAN_CONFIGS["s2v-14B"],
            checkpoint_dir=str(base),
            device_id=0,
            rank=0,
            single_gpu=True,
            sp_size=1,
            convert_model_dtype=True,
            init_on_cpu=False,
            offload_kv_cache=False,
        )
        settings = yaml.safe_load(
            (SOURCE / "liveavatar/configs/s2v_causal_sft.yaml").read_text()
        )
        self.pipeline.noise_model = self.pipeline.add_lora_to_model(
            self.pipeline.noise_model,
            lora_rank=settings["lora_rank"],
            lora_alpha=settings["lora_alpha"],
            lora_target_modules=settings["lora_target_modules"],
            init_lora_weights=settings["init_lora_weights"],
            pretrained_lora_path=str(lora / "liveavatar.safetensors"),
            load_lora_weight_only=False,
        )
        self._bridge: ClipBridge | None = None
        self._audio = np.zeros(0, dtype=np.float32)
        self._offset = 0

    def start(
        self,
        *,
        image: Path,
        audio: Path,
        pose: Path | None,
        prompt: str,
        negative_prompt: str,
        seed: int,
        max_chunks: int,
    ):
        import soundfile as sf

        self.close()
        driving_audio, rate = sf.read(audio, dtype="float32")
        self._audio = playback_audio(driving_audio, rate)
        self._offset = 0

        def produce(emit):
            import torch

            torch.cuda.set_device(0)
            try:
                self.pipeline.generate(
                    input_prompt=prompt,
                    ref_image_path=str(image),
                    audio_path=str(audio),
                    pose_video=str(pose) if pose else None,
                    num_repeat=max_chunks,
                    generate_size="704*384",
                    max_area=704 * 384,
                    infer_frames=48,
                    sampling_steps=4,
                    sample_solver="euler",
                    shift=3.0,
                    guide_scale=0,
                    n_prompt=negative_prompt,
                    seed=seed,
                    offload_model=False,
                    enable_online_decode=False,
                    num_gpus_dit=1,
                    enable_vae_parallel=False,
                    chunk_callback=emit,
                )
            finally:
                self.pipeline.kv_cache1 = None
                self.pipeline.shared_cond_cache = None
                self.pipeline.crossattn_cache = None
                self.pipeline._sampler_timesteps = None
                self.pipeline._sampler_sigmas = None

        self._bridge = ClipBridge(produce)

    def next(self):
        if self._bridge is None:
            raise RuntimeError("No active take")
        clip = self._bridge.next()
        if clip is None:
            return None
        video = (
            ((clip.float().clamp(-1, 1) + 1) * 127.5)
            .round()
            .to(dtype=__import__("torch").uint8)
        )
        video = video.permute(1, 2, 3, 0).numpy()
        count = len(video) * OUTPUT_SAMPLE_RATE // 25
        audio = self._audio[self._offset : self._offset + count]
        self._offset += count
        audio = np.pad(audio, (0, count - len(audio)))
        return video, audio[None, :]

    def close(self):
        if self._bridge is not None:
            self._bridge.close()
            self._bridge = None
        self._audio = np.zeros(0, dtype=np.float32)
        self._offset = 0
