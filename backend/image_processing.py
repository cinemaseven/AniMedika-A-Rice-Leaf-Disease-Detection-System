from __future__ import annotations

from dataclasses import dataclass
from io import BytesIO

import numpy as np
from PIL import Image, ImageOps, UnidentifiedImageError
from pillow_heif import register_heif_opener

from backend.config import ALLOWED_IMAGE_FORMATS, IMAGE_SIZE, INPUT_SCALE_MODE

register_heif_opener()

class InvalidImageError(ValueError):
    """Raised when an uploaded file cannot safely be used as an image."""


@dataclass(frozen=True)
class PreparedImage:
    batch: np.ndarray
    original_width: int
    original_height: int
    image_format: str


def decode_uploaded_image(image_bytes: bytes) -> tuple[Image.Image, str]:
    if not image_bytes:
        raise InvalidImageError("The uploaded image is empty.")

    try:
        with Image.open(BytesIO(image_bytes)) as source:
            image_format = (source.format or "").upper()
            if image_format not in ALLOWED_IMAGE_FORMATS:
                raise InvalidImageError(
                    "Unsupported image format. Please use JPEG, PNG, WEBP, HEIC, or HEIF."
                )

            source.verify()

        with Image.open(BytesIO(image_bytes)) as source:
            corrected = ImageOps.exif_transpose(source)
            image = corrected.convert("RGB").copy()

    except (UnidentifiedImageError, OSError, ValueError) as exc:
        if isinstance(exc, InvalidImageError):
            raise
        raise InvalidImageError("The uploaded file is not a readable image.") from exc

    return image, image_format

def prepare_image_for_model(image_bytes: bytes) -> PreparedImage:
    """Decode and prepare one uploaded image without saving it to disk."""
    image, image_format = decode_uploaded_image(image_bytes)
    original_width, original_height = image.size

    image_array = np.asarray(image, dtype=np.float32)
    
    import tensorflow as tf

    image_tensor = tf.convert_to_tensor(
        image_array,
        dtype=tf.float32,
    )

    image_tensor = tf.image.resize(
        image_tensor,
        IMAGE_SIZE,
        method="bilinear",
        antialias=False,
    )
    
    image_array = image_tensor.numpy()

    if INPUT_SCALE_MODE == "zero_one":
        image_array = image_array / 255.0
    elif INPUT_SCALE_MODE != "efficientnet_internal":
        raise RuntimeError(
            "ANIMEDIKA_INPUT_SCALE_MODE must be 'efficientnet_internal' or 'zero_one'."
        )

    batch = np.expand_dims(image_array, axis=0)

    return PreparedImage(
        batch=batch,
        original_width=original_width,
        original_height=original_height,
        image_format=image_format,
    )
