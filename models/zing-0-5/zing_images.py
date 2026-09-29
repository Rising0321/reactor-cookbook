"""Validate and materialize Zing TI2V input images."""

from __future__ import annotations

import io

from PIL import Image, UnidentifiedImageError
from reactor_runtime import CommandError, UploadedFile

_MAX_BYTES = 25 * 1024 * 1024
_FORMATS = {"JPEG", "PNG", "WEBP", "BMP"}


def validate_image(upload: UploadedFile) -> None:
    if len(upload.data) > _MAX_BYTES:
        raise CommandError("image_too_large", "The first-frame image must be at most 25 MiB.")
    try:
        with Image.open(io.BytesIO(upload.data)) as image:
            if image.format not in _FORMATS:
                raise CommandError("image_format", "Use a JPEG, PNG, WebP, or BMP image.")
            if image.width * image.height > 100_000_000:
                raise CommandError("image_dimensions", "The image must contain at most 100 million pixels.")
            image.verify()
    except (UnidentifiedImageError, OSError) as exc:
        raise CommandError("invalid_image", "The uploaded bytes are not a readable image.") from exc
