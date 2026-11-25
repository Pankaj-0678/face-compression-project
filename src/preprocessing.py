"""Image preparation shared by compression and quality evaluation."""
from __future__ import annotations

import numpy as np
from PIL import Image

IMAGE_SIZE = (128, 128)


def preprocess_image(image: Image.Image | np.ndarray) -> np.ndarray:
    """Convert an image to normalized RGB float32 batch of shape (1,128,128,3)."""
    if isinstance(image, Image.Image):
        if image.width < 1 or image.height < 1:
            raise ValueError("Image dimensions must be positive")
        image = image.convert("RGB").resize(IMAGE_SIZE, Image.Resampling.LANCZOS)
        array = np.asarray(image, dtype=np.float32)
    else:
        array = np.asarray(image)
        if array.ndim != 3 or array.shape[2] not in (1, 3, 4):
            raise ValueError("Expected an HxWxC image with 1, 3, or 4 channels")
        if array.size == 0 or not np.isfinite(array).all():
            raise ValueError("Image must be non-empty and contain finite values")
        if np.issubdtype(array.dtype, np.floating):
            if array.min() < 0 or array.max() > 1:
                raise ValueError("Floating point images must be in [0, 1]")
            array = array * 255.0
        array = np.clip(array, 0, 255).astype(np.uint8)
        image = Image.fromarray(array).convert("RGB").resize(IMAGE_SIZE, Image.Resampling.LANCZOS)
        array = np.asarray(image, dtype=np.float32)
    batch = array / np.float32(255.0)
    if batch.shape != (128, 128, 3) or not np.isfinite(batch).all():
        raise ValueError("Preprocessing produced an invalid image")
    return batch[np.newaxis, ...].astype(np.float32)


def image_to_uint8(image: Image.Image | np.ndarray, size: tuple[int, int] | None = None) -> np.ndarray:
    """Convert to RGB uint8, optionally resizing to (width, height)."""
    if isinstance(image, Image.Image):
        im = image
    else:
        array = np.asarray(image)
        if array.ndim != 3 or array.shape[2] not in (1, 3, 4):
            raise ValueError("Expected an HxWxC image with 1, 3, or 4 channels")
        if array.size == 0 or not np.isfinite(array).all():
            raise ValueError("Image must be non-empty and contain finite values")
        # PIL cannot create RGB images directly from float32 arrays. Decoder
        # output is normalized to [0, 1], while other callers may pass [0, 255].
        if np.issubdtype(array.dtype, np.floating):
            if array.min() < 0 or array.max() > 255:
                raise ValueError("Floating point image values must be in [0, 1] or [0, 255]")
            if array.max() <= 1:
                array = array * 255.0
        array = np.clip(array, 0, 255).astype(np.uint8)
        if array.shape[2] == 1:
            array = array[:, :, 0]
        im = Image.fromarray(array)
    im = im.convert("RGB")
    if size is not None:
        im = im.resize(size, Image.Resampling.LANCZOS)
    return np.asarray(im, dtype=np.uint8)
