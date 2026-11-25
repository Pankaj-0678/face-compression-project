"""Per-array min-max uniform 8-bit latent quantization."""
from __future__ import annotations
import numpy as np


def quantize_latent(latent: np.ndarray) -> tuple[np.ndarray, float, float]:
    """Quantize finite numeric latent values to uint8 and return (q,min,max)."""
    arr = np.asarray(latent)
    if arr.size == 0 or not np.issubdtype(arr.dtype, np.number) or not np.isfinite(arr).all():
        raise ValueError("Latent must be a non-empty finite numeric array")
    lo, hi = float(np.min(arr)), float(np.max(arr))
    if hi == lo:
        quantized = np.zeros(arr.shape, dtype=np.uint8)
    else:
        quantized = np.rint((arr.astype(np.float64) - lo) / (hi - lo) * 255.0).astype(np.uint8)
    return quantized, lo, hi


def dequantize_latent(quantized: np.ndarray, latent_min: float, latent_max: float) -> np.ndarray:
    """Restore floating-point latent values from uint8 and stored bounds."""
    q = np.asarray(quantized)
    if q.dtype != np.uint8 or q.size == 0 or not np.isfinite([latent_min, latent_max]).all() or latent_max < latent_min:
        raise ValueError("Invalid quantized values or min/max metadata")
    if latent_max == latent_min:
        return np.full(q.shape, latent_min, dtype=np.float32)
    return (q.astype(np.float64) / 255.0 * (latent_max - latent_min) + latent_min).astype(np.float32)
