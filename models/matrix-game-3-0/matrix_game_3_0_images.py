"""Validate Matrix-Game 3.0 input images and generated RGB chunks."""

from __future__ import annotations

import io

import numpy as np
from numpy.typing import NDArray
from PIL import Image, UnidentifiedImageError
from reactor_runtime import CommandError, UploadedFile

FIRST_CHUNK_FRAMES = 57
LATER_CHUNK_FRAMES = 40
FRAMES_PER_CHUNK = max(FIRST_CHUNK_FRAMES, LATER_CHUNK_FRAMES)

_UPLOAD_MAX_BYTES = 25 * 1024 * 1024
_UPLOAD_MAX_PIXELS = 100_000_000
_UPLOAD_FORMATS = {"BMP", "JPEG", "PNG", "WEBP"}


def normalize_output_frames(
    value: NDArray[np.generic], chunk_index: int
) -> NDArray[np.uint8]:
    """Return one contiguous uint8 RGB native Matrix chunk."""
    frames = np.asarray(value)
    if frames.ndim != 4 or frames.shape[-1] != 3:
        raise RuntimeError(
            f"Matrix output must have shape (T, H, W, 3), got {frames.shape}"
        )
    expected = FIRST_CHUNK_FRAMES if chunk_index == 0 else LATER_CHUNK_FRAMES
    if int(frames.shape[0]) != expected:
        raise RuntimeError(
            f"Matrix chunk {chunk_index + 1} must contain {expected} frames, "
            f"got {int(frames.shape[0])}"
        )
    if frames.dtype != np.uint8:
        frames = np.clip(frames, 0, 255).astype(np.uint8)
    return np.ascontiguousarray(frames)


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
            width, height = decoded.size
            image_format = decoded.format
            if image_format not in _UPLOAD_FORMATS:
                raise CommandError(
                    "unsupported_media",
                    f"{image.name} must be JPEG, PNG, WebP, or BMP.",
                )
            if width <= 0 or height <= 0 or width * height > _UPLOAD_MAX_PIXELS:
                raise CommandError(
                    "image_too_large",
                    f"{image.name} exceeds the {_UPLOAD_MAX_PIXELS}-pixel limit.",
                )
            decoded.load()
    except CommandError:
        raise
    except (UnidentifiedImageError, OSError, ValueError) as error:
        raise CommandError(
            "invalid_image", f"{image.name} cannot be decoded."
        ) from error
