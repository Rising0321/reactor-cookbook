"""Opt-in three-GPU turbo mode for the native TPP backend.

The released path runs the four denoising stages across four GPUs with a fifth
dedicated to streaming VAE. Turbo instead packs the stages 2+2 across two GPUs
with a third rank running the dedicated streaming VAE (a stage repack reusing
``install_grouped_generate``) and ``torch.compile``s the DiT while keeping the
VAE eager (compiling the VAE regressed it). It keeps all four denoising steps --
no step reduction -- and lands ~31 FPS gap-free on three GPUs, matching the
released four-GPU throughput on less hardware.

Because the model is run-to-run nondeterministic (FA4 / parallel reductions,
amplified by the autoregressive rollout), bit-exactness is unattainable for any
layout; "lossless" here means within that intrinsic variance. The repack and the
DiT ``torch.compile`` both stay inside it -- their decoded-frame MSE against the
five-GPU path is no larger than two identical five-GPU runs differ from each
other, and ``reactor_bench`` identity / temporal-flicker / reference-fidelity
land in the same band.

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

    ``world_size``    processes/GPUs to spawn (turbo three, released five).
    ``num_gpus_dit``  ranks that run DiT stages (turbo two, packed 2+2, with a
                      third dedicated VAE rank; released four, leaving rank four
                      for the dedicated VAE).
    ``output_rank``   rank that decodes and delivers clips to Runtime.
    ``sampling_steps`` denoising steps -- four in both modes (no reduction).
                      Overridable with ``LIVEAVATAR_STEPS`` for experiments.
    ``shared_vae``    whether the VAE shares the last DiT rank; turbo keeps a
                      dedicated VAE rank (``False``) via the stage repack.
    ``compile``       DiT-only ``torch.compile``.
    """
    if turbo_enabled():
        return {
            "world_size": 3,
            "num_gpus_dit": 2,
            "output_rank": 2,
            "sampling_steps": int(os.environ.get("LIVEAVATAR_STEPS", "4")),
            "shared_vae": False,
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
