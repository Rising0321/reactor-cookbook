# Copyright (c) 2026 Reactor Technologies, Inc. All rights reserved.
"""Resolve the authorized local source and checkpoint tree in the application."""

import json
import sys
from pathlib import Path

import yaml

GAMES = ("freeze", "spiders", "tracer", "bioshock", "godzilla", "arcadeduel", "dust2")


def resolve_assets(config_path: Path | None, root: Path):
    if config_path is None:
        raise ValueError("EdenLive requires its edenlive.yaml config")
    config = yaml.safe_load(config_path.read_text())
    execution = {
        "attention_backend": config.get("attention_backend", "native"),
        "cuda_graph": config.get("cuda_graph", False),
        "compile_model": config.get("compile_model", False),
        "warmup": config.get("warmup", False),
    }
    if execution["attention_backend"] not in ("native", "fa4"):
        raise ValueError("attention_backend must be 'native' or 'fa4'")
    for key in ("cuda_graph", "compile_model", "warmup"):
        if type(execution[key]) is not bool:
            raise ValueError(f"{key} must be a boolean")
    capture = config.get("game_capture", [1280, 704])
    if (
        not isinstance(capture, list)
        or len(capture) != 2
        or not all(type(side) is int and side >= 64 for side in capture)
    ):
        raise ValueError("game_capture must be [width, height] in pixels")
    source = (root / config["source"]).resolve()
    weights = (root / config["checkpoint"]).resolve()
    registry = source / "realtime/games/registry.json"
    required = (
        registry,
        weights / "dit-5b/diffusion_pytorch_model.safetensors",
        weights / "vae-5b/Wan2.2_VAE.pth",
        weights / "decoder-5b/lighttaew2_2.pth",
        weights / "vision-encoder/projection.safetensors",
        weights / "vision-encoder/backbone/model.safetensors",
        weights / "umt5/text_encoder/config.json",
    )
    for path in required:
        if not path.is_file():
            raise FileNotFoundError(f"Required EdenLive asset is missing: {path}")
    entries = {entry["slug"]: entry for entry in json.loads(registry.read_text())["games"]}
    for slug in GAMES:
        if (
            slug not in entries
            or not (source / "realtime/games" / slug / entries[slug]["cover"]).is_file()
        ):
            raise FileNotFoundError(f"Missing official game or cover: {slug}")
    # Only the CPU game renderer is imported in the application process.
    sys.path.insert(0, str(source / "realtime/integrations_v2"))
    return source, weights, entries, execution, tuple(capture)
