"""Validated FACE-C v1 container format."""
from __future__ import annotations
import json
import math
import hashlib
import struct
from pathlib import Path
from typing import BinaryIO
import numpy as np
from config import MAX_HEADER_BYTES, MAX_PAYLOAD_BYTES
from src.huffman import decode

MAGIC = b"FAC1"
VERSION = 1
MAX_SYMBOLS = 1_000_000


def _validate_meta(meta: dict) -> tuple[int, ...]:
    if not isinstance(meta, dict) or meta.get("version") != VERSION: raise ValueError("Unsupported FACE-C version")
    shape = meta.get("latent_shape")
    if not isinstance(shape, list) or not shape or len(shape) > 8 or any(type(x) is not int or x <= 0 for x in shape):
        raise ValueError("Invalid latent shape")
    count = math.prod(shape)
    if count > MAX_SYMBOLS: raise ValueError("Latent shape exceeds safety limit")
    for key in ("latent_min", "latent_max"):
        if not isinstance(meta.get(key), (int, float)) or not math.isfinite(meta[key]): raise ValueError(f"Invalid {key}")
    if meta["latent_max"] < meta["latent_min"]: raise ValueError("Invalid latent bounds")
    if type(meta.get("bit_length")) is not int or meta["bit_length"] < 0: raise ValueError("Invalid Huffman bit length")
    book = meta.get("codebook")
    if not isinstance(book, dict) or len(book) > 256: raise ValueError("Invalid Huffman codebook")
    return tuple(shape)


def write_facec_file(output_path: str | Path, quantized: np.ndarray, latent_min: float,
                     latent_max: float, payload: bytes, bit_length: int,
                     codebook: dict[int, str]) -> dict:
    q = np.asarray(quantized)
    if q.dtype != np.uint8 or q.size == 0 or q.size > MAX_SYMBOLS: raise ValueError("Invalid quantized latent array")
    meta = {"version": VERSION, "latent_shape": list(q.shape), "latent_min": float(latent_min),
            "latent_max": float(latent_max), "bit_length": int(bit_length),
            "codebook": {str(k): v for k, v in codebook.items()},
            "payload_sha256": hashlib.sha256(payload).hexdigest()}
    _validate_meta(meta)
    if len(payload) > MAX_PAYLOAD_BYTES or bit_length > len(payload) * 8 or len(payload) != (bit_length + 7) // 8:
        raise ValueError("Invalid Huffman payload length")
    if decode(payload, bit_length, codebook, q.size).size != q.size:
        raise ValueError("Payload does not encode expected latent count")
    header = json.dumps(meta, separators=(",", ":"), allow_nan=False).encode("utf-8")
    if len(header) > MAX_HEADER_BYTES: raise ValueError("Header too large")
    path = Path(output_path); path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(MAGIC + struct.pack(">I", len(header)) + header + payload)
    return meta


def read_facec_file(source: str | Path | bytes | BinaryIO) -> tuple[dict, np.ndarray]:
    """Read container and return (metadata, exact quantized array)."""
    if isinstance(source, bytes): data = source
    elif hasattr(source, "read"): data = source.read(MAX_HEADER_BYTES + MAX_PAYLOAD_BYTES + 9)
    else:
        path = Path(source)
        if path.stat().st_size > MAX_HEADER_BYTES + MAX_PAYLOAD_BYTES + 8: raise ValueError("FACE-C file exceeds safety limit")
        data = path.read_bytes()
    if len(data) < 8 or data[:4] != MAGIC: raise ValueError("Invalid FACE-C magic signature")
    header_len = struct.unpack(">I", data[4:8])[0]
    if not 1 <= header_len <= MAX_HEADER_BYTES or len(data) < 8 + header_len: raise ValueError("Invalid FACE-C header length")
    try: meta = json.loads(data[8:8 + header_len].decode("utf-8"))
    except (UnicodeDecodeError, json.JSONDecodeError) as exc: raise ValueError("Malformed FACE-C metadata") from exc
    shape = _validate_meta(meta)
    payload = data[8 + header_len:]
    if len(payload) > MAX_PAYLOAD_BYTES or len(payload) != (meta["bit_length"] + 7) // 8:
        raise ValueError("Invalid FACE-C payload length")
    digest = meta.get("payload_sha256")
    if not isinstance(digest, str) or digest != hashlib.sha256(payload).hexdigest():
        raise ValueError("FACE-C payload checksum mismatch")
    try:
        book = {int(k): v for k, v in meta["codebook"].items()}
        q = decode(payload, meta["bit_length"], book, math.prod(shape)).reshape(shape)
    except (ValueError, TypeError, KeyError) as exc: raise ValueError(f"Invalid FACE-C Huffman data: {exc}") from exc
    return meta, q
