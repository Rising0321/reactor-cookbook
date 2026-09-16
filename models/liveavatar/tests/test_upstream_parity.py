"""Execute upstream and patched control flow with CPU surrogate neural modules.

This checks scheduling/history equivalence, not learned-model output quality.
"""

import ast
import contextlib
import gc
import random
import subprocess
import sys
from copy import deepcopy
from types import SimpleNamespace

import numpy as np
import pytest
import torch
from diffusers import FlowMatchEulerDiscreteScheduler
from PIL import Image
from torchvision import transforms

from liveavatar_assets import SOURCE, SOURCE_REVISION


def load_generate(source):
    tree = ast.parse(source)
    cls = next(
        n for n in tree.body if isinstance(n, ast.ClassDef) and n.name == "WanS2V"
    )
    fn = next(
        n for n in cls.body if isinstance(n, ast.FunctionDef) and n.name == "generate"
    )
    fake_torch = SimpleNamespace(**vars(torch))
    for name in ("ones", "randn", "tensor"):

        def cpu_call(*args, _fn=getattr(torch, name), **kwargs):  # noqa: B008 - capture each loop function
            kwargs["device"] = "cpu"
            return _fn(*args, **kwargs)

        setattr(fake_torch, name, cpu_call)
    fake_torch.Generator = lambda device: torch.Generator(device="cpu")
    fake_torch.amp = SimpleNamespace(autocast=lambda *a, **k: contextlib.nullcontext())
    scope = {
        "torch": fake_torch,
        "np": np,
        "transforms": transforms,
        "Image": Image,
        "random": random,
        "sys": sys,
        "deepcopy": deepcopy,
        "gc": gc,
        "tqdm": lambda values: values,
        "dist": SimpleNamespace(get_rank=lambda: 0),
        "FlowMatchEulerDiscreteScheduler": FlowMatchEulerDiscreteScheduler,
    }
    exec(  # noqa: S102 - pinned upstream test code
        compile(ast.Module(body=[fn], type_ignores=[]), "<upstream-generate>", "exec"),
        scope,
    )
    return scope["generate"]


class DeviceStub:
    def to(self, *args, **kwargs):
        return self

    def cpu(self):
        return self

    def requires_grad_(self, value):
        return self

    def eval(self):
        return self


class VAE:
    model = DeviceStub()
    dtype = torch.float32
    device = torch.device("cpu")

    def encode(self, images):
        return [
            image.mean(0, keepdim=True).repeat(16, 1, 1, 1)[:, ::4, ::8, ::8]
            for image in images
        ]

    def decode(self, latents):
        return [
            latent[:3]
            .repeat_interleave(4, 1)
            .repeat_interleave(8, 2)
            .repeat_interleave(8, 3)
            for latent in latents
        ]


class Noise(DeviceStub):
    num_layers = 1

    def __init__(self):
        self.trace = []

    def __call__(self, latents, **kwargs):
        self.trace.append(
            (
                kwargs["current_start"],
                kwargs["current_end"],
                kwargs.get("sink_flag", False),
                kwargs["motion_latents"].clone(),
                kwargs["ref_latents"].clone(),
            )
        )
        return [latents[0] * 0.03 + len(self.trace) * 0.001]


class Pipeline:
    def __init__(self):
        self.device = torch.device("cpu")
        self.param_dtype = torch.float32
        self.vae = VAE()
        self.audio_encoder = SimpleNamespace(model=DeviceStub())
        self.noise_model = Noise()
        self.motion_frames = 73
        self.num_frames_per_block = 3
        self.sample_neg_prompt = "negative"
        self.num_train_timesteps = 1000
        self.init_on_cpu = False
        self.offload_kv_cache = True  # avoid allocating real multi-GiB caches on CPU
        self.rank = 0
        self.allocations = []

    def get_gen_size(self, **kwargs):
        return 32, 32

    def encode_audio(self, *args, **kwargs):
        return torch.zeros(1, 144), 3

    def load_pose_cond(self, **kwargs):
        return [torch.zeros(1, 16, 12, 4, 4)]

    def encode_prompt(self, *args):
        return [torch.zeros(1, 4)], [torch.zeros(1, 4)]

    def _initialize_kv_cache(self, **kwargs):
        self.allocations.append(kwargs.copy())
        self.kv_cache1[str(kwargs["gpu_id"])] = []

    def _initialize_crossattn_cache(self, **kwargs):
        self.crossattn_cache = []

    def _move_kv_cache_to_working_gpu(self, *args):
        pass


@pytest.mark.parametrize("online", [False, True])
def test_streamed_clips_equal_original_deferred_decode(tmp_path, online):
    path = "liveavatar/models/wan/causal_s2v_pipeline.py"
    original = subprocess.check_output(
        ["git", "-C", str(SOURCE), "show", f"{SOURCE_REVISION}:{path}"], text=True
    )
    patched = (SOURCE / path).read_text()
    image = tmp_path / "test.png"
    Image.new("RGB", (32, 32), (128, 100, 60)).save(image)
    kwargs = {
        "input_prompt": "test",
        "ref_image_path": str(image),
        "audio_path": "test.wav",
        "infer_frames": 48,
        "num_repeat": 3,
        "sampling_steps": 4,
        "sample_solver": "euler",
        "offload_model": False,
        "enable_online_decode": online,
        "seed": 420,
    }
    reference_pipeline, stream_pipeline = Pipeline(), Pipeline()
    expected, _ = load_generate(original)(reference_pipeline, **kwargs)
    chunks = []
    actual, _ = load_generate(patched)(
        stream_pipeline, **kwargs, chunk_callback=chunks.append
    )
    assert actual is None
    assert [clip.shape[1] for clip in chunks] == [45, 48, 48]
    torch.testing.assert_close(torch.cat(chunks, 1), expected, rtol=0, atol=0)
    assert reference_pipeline.allocations == stream_pipeline.allocations
    assert len(reference_pipeline.noise_model.trace) == len(
        stream_pipeline.noise_model.trace
    )
    for a, b in zip(
        reference_pipeline.noise_model.trace, stream_pipeline.noise_model.trace
    ):
        assert a[:3] == b[:3]
        torch.testing.assert_close(a[3], b[3], rtol=0, atol=0)
        torch.testing.assert_close(a[4], b[4], rtol=0, atol=0)


def test_cache_methods_unchanged():
    path = "liveavatar/models/wan/causal_s2v_pipeline.py"
    original = subprocess.check_output(
        ["git", "-C", str(SOURCE), "show", f"{SOURCE_REVISION}:{path}"], text=True
    )

    def methods(source):
        return {
            node.name: ast.dump(node)
            for node in ast.walk(ast.parse(source))
            if isinstance(node, ast.FunctionDef) and "cache" in node.name
        }

    assert methods(original) == methods((SOURCE / path).read_text())
