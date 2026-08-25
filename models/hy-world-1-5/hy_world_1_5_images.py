"""Validate and decode HY-World 1.5 reference images and output chunks."""

from __future__ import annotations

import io
from pathlib import Path

import numpy as np
from PIL import Image, ImageOps, UnidentifiedImageError
from reactor_runtime import CommandError, UploadedFile

_UPLOAD_MAX_BYTES = 25 * 1024 * 1024
_UPLOAD_MAX_PIXELS = 100_000_000
_UPLOAD_FORMATS = {"BMP", "JPEG", "PNG", "WEBP"}


def validate_uploaded_image(image: UploadedFile) -> None:
    """Reject oversized, mislabeled, or undecodable uploaded image bytes."""
    if not image.mime_type.startswith("image/"):
        raise CommandError("unsupported_media", f"{image.name} is not an image.")
    if not image.data:
        raise CommandError("invalid_image", f"{image.name} is empty.")
    if image.size > _UPLOAD_MAX_BYTES:
        raise CommandError(
            "image_too_large",
            f"{image.name} exceeds the {_UPLOAD_MAX_BYTES // (1024 * 1024)} MiB limit.",
        )
    try:
        with Image.open(io.BytesIO(image.data)) as decoded:
            if decoded.format not in _UPLOAD_FORMATS:
                raise CommandError(
                    "unsupported_media",
                    f"{image.name} must be JPEG, PNG, WebP, or BMP.",
                )
            width, height = decoded.size
            if width <= 0 or height <= 0 or width * height > _UPLOAD_MAX_PIXELS:
                raise CommandError(
                    "image_too_large",
                    f"{image.name} exceeds the {_UPLOAD_MAX_PIXELS}-pixel limit.",
                )
            decoded.verify()
    except CommandError:
        raise
    except (
        Image.DecompressionBombError,
        UnidentifiedImageError,
        OSError,
        ValueError,
    ) as error:
        raise CommandError(
            "invalid_image", f"{image.name} cannot be decoded."
        ) from error


def load_reference_image(value: Path | UploadedFile) -> Image.Image:
    """Return a detached RGB image with EXIF orientation applied."""
    source = io.BytesIO(value.data) if isinstance(value, UploadedFile) else value
    try:
        with Image.open(source) as image:
            return ImageOps.exif_transpose(image).convert("RGB").copy()
    except (
        Image.DecompressionBombError,
        UnidentifiedImageError,
        OSError,
        ValueError,
    ) as error:
        raise RuntimeError(f"Selected image cannot be decoded: {value.name}") from error


def normalize_output_frames(value: np.ndarray, *, first_chunk: bool) -> np.ndarray:
    """Return one contiguous uint8 RGB causal chunk with its native length."""
    frames = np.asarray(value)
    expected = 13 if first_chunk else 16
    if frames.ndim != 4 or frames.shape[-1] != 3:
        raise RuntimeError(
            f"HY-World output must have shape (T, H, W, 3), got {frames.shape}"
        )
    if int(frames.shape[0]) != expected:
        raise RuntimeError(
            f"HY-World chunk must contain {expected} frames, got {int(frames.shape[0])}"
        )
    if frames.dtype != np.uint8:
        frames = np.clip(frames, 0, 255).astype(np.uint8)
    return np.ascontiguousarray(frames)
