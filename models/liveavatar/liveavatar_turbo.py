"""Opt-in two-GPU turbo mode for the native TPP backend.

The released path runs the four denoising stages across four GPUs with a fifth
dedicated to streaming VAE. Turbo instead packs the stages 2+2 across two GPUs
with the VAE sharing the second rank (a bit-exact stage repack, reusing
``install_grouped_generate``), ``torch.compile``s the DiT while keeping the VAE
eager (compiling the VAE regressed it), and runs three denoising steps. That
lands ~31 FPS gap-free on two GPUs -- the five-GPU throughput on far less
hardware -- and measured quality-equivalent to the 4-step/5-GPU reference
(reactor_bench within noise, lip-sync held, no long-take drift across five
subjects incl. a 52 s take).

Turbo is off unless ``LIVEAVATAR_TURBO=1``; with it unset every value below
reproduces the released five-GPU behaviour exactly, so the default path is
untouched.
"""

from __future__ import annotations

import os


def turbo_enabled() -> bool:
    return os.environ.get("LIVEAVATAR_TURBO", "0") == "1"


def turbo_plan() -> dict:
    """Serving parameters for the active mode.

    ``world_size``    processes/GPUs to spawn.
    ``num_gpus_dit``  ranks that run DiT stages (turbo: both; released: four,
                      leaving rank four for the dedicated VAE).
    ``output_rank``   rank that decodes and delivers clips to Runtime.
    ``sampling_steps`` denoising steps (turbo 3, released 4). Overridable with
                      ``LIVEAVATAR_STEPS`` for experiments.
    ``shared_vae``    pack the VAE onto the last DiT rank via the stage repack.
    ``compile``       DiT-only ``torch.compile``.
    """
    if turbo_enabled():
        return {
            "world_size": 2,
            "num_gpus_dit": 2,
            "output_rank": 1,
            "sampling_steps": int(os.environ.get("LIVEAVATAR_STEPS", "3")),
            "shared_vae": True,
            "compile": True,
        }
    return {
        "world_size": 5,
        "num_gpus_dit": 4,
        "output_rank": 4,
        "sampling_steps": int(os.environ.get("LIVEAVATAR_STEPS", "4")),
        "shared_vae": False,
        "compile": False,
    }


def install_dit_compile() -> None:
    """Compile the DiT only; keep the streaming VAE eager.

    Must run before ``causal_s2v_pipeline_tpp`` is imported so the pinned
    ``@conditional_compile`` decorators pick up the selective wrapper. The
    upstream decorator would otherwise also compile ``stream_decode``, which
    regressed steady-state decode.
    """
    import importlib

    import torch

    os.environ["ENABLE_COMPILE"] = "true"
    inference_utils = importlib.import_module("liveavatar.models.wan.inference_utils")
    eager = {"stream_decode"}

    def selective(func):
        if func.__name__ in eager:
            return func
        return torch.compile(mode=None, backend="inductor", dynamic=None)(func)

    inference_utils.conditional_compile = selective
