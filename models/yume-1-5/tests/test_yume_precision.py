"""The released sampler retains float32 VAE weights and latent arithmetic."""

import sys
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock

import pytest


def test_backend_keeps_native_vae_dtype(monkeypatch):
    torch = pytest.importorskip("torch")
    from yume_backend import YumeBackend

    transformer = Mock()
    transformer.to.return_value = transformer
    transformer.eval.return_value = transformer
    transformer.requires_grad_.return_value = transformer
    vae_model = torch.nn.Linear(2, 2)
    monkeypatch.setattr(vae_model, "to", lambda *_a, **_kw: vae_model)
    wan = SimpleNamespace(model=transformer, vae=SimpleNamespace(model=vae_model))
    monkeypatch.setitem(sys.modules, "wan23", SimpleNamespace(Yume=lambda **_kw: wan))
    monkeypatch.setitem(
        sys.modules, "wan23.configs", SimpleNamespace(WAN_CONFIGS={"ti2v-5B": {}})
    )
    monkeypatch.setattr(torch.cuda, "is_available", lambda: True)
    monkeypatch.setattr(torch.cuda, "current_device", lambda: 0)
    monkeypatch.setattr(torch, "Generator", lambda **_kw: Mock())
    YumeBackend(SimpleNamespace(source_path=Path("."), checkpoint_path=Path(".")))
    assert next(vae_model.parameters()).dtype == torch.float32


@pytest.mark.parametrize("video", [False, True])
def test_conditioning_matches_native_float_tensor_resize(video):
    import numpy as np

    torch = pytest.importorskip("torch")
    from yume_backend import _resize_conditioning

    pixels = np.random.default_rng(5).integers(0, 256, (3, 7, 11, 3), dtype=np.uint8)
    if not video:
        pixels = pixels[0]
    source = (
        torch.from_numpy(pixels if video else pixels[None]).permute(0, 3, 1, 2).float()
        / 255.0
    )
    if video:
        expected = torch.nn.functional.interpolate(
            (source - 0.5) / 0.5, size=(13, 17), mode="bicubic"
        ).clamp(-1, 1)
    else:
        expected = (
            torch.nn.functional.interpolate(
                source, size=(13, 17), mode="bilinear", align_corners=False
            )
            - 0.5
        ) * 2
    actual = _resize_conditioning(pixels, video=video, height=13, width=17)
    torch.testing.assert_close(actual, expected, rtol=0, atol=0)
    assert actual.dtype == torch.float32
