"""Compression statistics and consistently ranged image quality metrics."""
from __future__ import annotations
import numpy as np
from skimage.metrics import peak_signal_noise_ratio, structural_similarity


def compression_metrics(original_image_bytes: int, latent_bytes: int, quantized_bytes: int,
                        huffman_bytes: int, facec_bytes: int, source_file_bytes: int | None = None) -> dict:
    """Return clearly distinguished size, ratio, and storage reduction values."""
    return {"original_image_size": original_image_bytes, "original_file_size": source_file_bytes,
            "original_float32_latent_size": latent_bytes, "quantized_latent_size": quantized_bytes,
            "huffman_payload_size": huffman_bytes, "facec_file_size": facec_bytes,
            "latent_compression_ratio": latent_bytes / facec_bytes if facec_bytes else None,
            "raw_image_compression_ratio": original_image_bytes / facec_bytes if facec_bytes else None,
            "storage_reduction_percent": (1 - facec_bytes / original_image_bytes) * 100 if original_image_bytes else None}


def image_quality(original: np.ndarray, reconstructed: np.ndarray) -> dict:
    """Compute PSNR/SSIM after mapping both RGB images to [0,1]."""
    a, b = np.asarray(original, dtype=np.float32), np.asarray(reconstructed, dtype=np.float32)
    if a.shape != b.shape or a.ndim != 3 or a.shape[2] != 3: raise ValueError("Images must have matching HxWx3 shapes")
    if a.max() > 1: a = a / 255.0
    if b.max() > 1: b = b / 255.0
    a, b = np.clip(a, 0, 1), np.clip(b, 0, 1)
    return {"psnr": float(peak_signal_noise_ratio(a, b, data_range=1.0)),
            "ssim": float(structural_similarity(a, b, channel_axis=2, data_range=1.0))}
