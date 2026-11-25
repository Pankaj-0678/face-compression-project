"""End-to-end FACE-C encode and decode pipelines."""
from __future__ import annotations
from pathlib import Path
from typing import Any
import numpy as np
from PIL import Image
from src.preprocessing import preprocess_image
from src.quantization import quantize_latent, dequantize_latent
from src.huffman import encode
from src.facec_format import read_facec_file, write_facec_file
from config import EXPECTED_LATENT_SHAPE


def _predict(model: Any, batch: np.ndarray) -> np.ndarray:
    result = model.predict(batch, verbose=0) if hasattr(model, "predict") else model(batch, training=False)
    if isinstance(result, (list, tuple)): raise ValueError("Model must have one tensor output")
    arr = np.asarray(result)
    return arr


def encode_image(image: np.ndarray, encoder: Any) -> np.ndarray:
    """Run encoder on one preprocessed batch and validate latent output."""
    batch = np.asarray(image, dtype=np.float32)
    if batch.shape != (1, 128, 128, 3) or not np.isfinite(batch).all() or batch.min() < 0 or batch.max() > 1:
        raise ValueError("Expected preprocessed image with shape (1,128,128,3) in [0,1]")
    latent = _predict(encoder, batch)
    expected_shape = (1, *EXPECTED_LATENT_SHAPE)
    if latent.shape != expected_shape or not np.issubdtype(latent.dtype, np.number) or not np.isfinite(latent).all():
        raise ValueError(f"Encoder output must have shape {expected_shape} and finite values; got {latent.shape}")
    return latent[0].astype(np.float32)


def compress_image(input_image: Image.Image | np.ndarray, encoder: Any, output_path: str | Path) -> dict:
    """Compress image and return metrics plus inspectable intermediate arrays."""
    batch = preprocess_image(input_image)
    latent = encode_image(batch, encoder)
    quantized, lo, hi = quantize_latent(latent)
    payload, bit_length, codebook = encode(quantized)
    metadata = write_facec_file(output_path, quantized, lo, hi, payload, bit_length, codebook)
    file_size = Path(output_path).stat().st_size
    source = input_image if isinstance(input_image, Image.Image) else Image.fromarray(np.asarray(input_image).astype(np.uint8))
    return {"path": str(output_path), "metadata": metadata, "latent": latent,
            "quantized": quantized, "huffman_payload": payload, "bit_length": bit_length,
            "codebook": codebook, "original_image_size": len(source.convert("RGB").tobytes()),
            "original_file_size": None, "original_latent_size": latent.nbytes,
            "quantized_latent_size": quantized.nbytes, "huffman_payload_size": len(payload),
            "facec_file_size": file_size}


def decompress_image(facec_file: str | Path | bytes, decoder: Any) -> dict:
    """Decode a FACE-C file and return normalized reconstructed RGB image and intermediates."""
    metadata, quantized = read_facec_file(facec_file)
    latent = dequantize_latent(quantized, metadata["latent_min"], metadata["latent_max"])
    if latent.shape != EXPECTED_LATENT_SHAPE:
        raise ValueError(f"Decoder latent shape mismatch: expected {EXPECTED_LATENT_SHAPE}, got {latent.shape}")
    reconstructed = _predict(decoder, latent[np.newaxis, ...])
    if reconstructed.shape != (1, 128, 128, 3): raise ValueError(f"Decoder output must have shape (1,128,128,3); got {reconstructed.shape}")
    if not np.isfinite(reconstructed).all(): raise ValueError("Decoder produced non-finite values")
    reconstructed = np.clip(reconstructed[0], 0, 1).astype(np.float32)
    return {"image": reconstructed, "metadata": metadata, "quantized": quantized, "latent": latent}
