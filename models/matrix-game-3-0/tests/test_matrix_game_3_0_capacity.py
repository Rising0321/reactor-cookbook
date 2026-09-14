"""Validate configurable capacity without changing native inference settings."""

from __future__ import annotations

from pathlib import Path
from tempfile import TemporaryDirectory
from unittest.mock import patch

import yaml
from matrix_game_3_0_assets import read_config
from matrix_game_3_0_backend import MatrixGame30Backend


def test_capacity_is_forwarded_to_native_pipeline_without_changing_chunk_settings() -> (
    None
):
    """Forward the recipe's capacity through the existing upstream argument."""
    config = read_config(Path(__file__).resolve().parents[1] / "matrix_game_3_0.yaml")
    assert config.max_chunks == 12
    args = MatrixGame30Backend(config)._build_args()
    assert args.num_iterations == 12
    assert args.num_inference_steps == 3
    assert args.size == "704*1280"
    assert args.use_int8 is True
    assert args.vae_type == "mg_lightvae_v2"


def test_capacity_accepts_positive_integers_and_retains_upstream_fallback() -> None:
    """Accept session lengths independently of the upstream default."""
    recipe = Path(__file__).resolve().parents[1] / "matrix_game_3_0.yaml"
    document = yaml.safe_load(recipe.read_text())
    with (
        TemporaryDirectory() as temporary,
        patch("matrix_game_3_0_assets.get_weights_path", return_value=Path(temporary)),
    ):
        for value in (1, 12, 64, 128):
            document["stream"]["max_chunks"] = value
            with patch("matrix_game_3_0_assets.yaml.safe_load", return_value=document):
                assert read_config(recipe).max_chunks == value
        del document["stream"]["max_chunks"]
        with patch("matrix_game_3_0_assets.yaml.safe_load", return_value=document):
            assert read_config(recipe).max_chunks == 12


def test_capacity_rejects_non_positive_or_non_integer_values() -> None:
    """Reject invalid capacities instead of silently truncating or coercing."""
    recipe = Path(__file__).resolve().parents[1] / "matrix_game_3_0.yaml"
    document = yaml.safe_load(recipe.read_text())
    for value in (0, -1, True, False, 1.5, "64", None):
        document["stream"]["max_chunks"] = value
        with patch("matrix_game_3_0_assets.yaml.safe_load", return_value=document):
            try:
                read_config(recipe)
            except ValueError as error:
                assert "stream.max_chunks must be a positive integer" in str(error)
            else:
                raise AssertionError(f"Accepted invalid capacity: {value!r}")
