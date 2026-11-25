"""Small presentation and file helpers."""
from __future__ import annotations
import io
import numpy as np
from PIL import Image


def png_bytes(image: np.ndarray) -> bytes:
    arr = np.clip(np.asarray(image) * (255 if np.asarray(image).max() <= 1 else 1), 0, 255).astype(np.uint8)
    stream = io.BytesIO(); Image.fromarray(arr, mode="RGB").save(stream, format="PNG")
    return stream.getvalue()


def format_bytes(value: int | None) -> str:
    if value is None: return "Unavailable"
    size = float(value)
    for unit in ("B", "KB", "MB", "GB"):
        if size < 1024 or unit == "GB": return f"{size:.1f} {unit}"
        size /= 1024
    return f"{size:.1f} GB"
