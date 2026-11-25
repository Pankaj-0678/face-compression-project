"""Lazy, cached loading and runtime validation for trained Keras models."""
from __future__ import annotations
from pathlib import Path
from typing import Any
from config import EXPECTED_LATENT_SHAPE


def _shape(value: Any) -> tuple[int | None, ...]:
    return tuple(None if x is None else int(x) for x in value)


def validate_models(encoder: Any, decoder: Any) -> dict[str, tuple]:
    """Validate runtime shapes and return actual input/output shapes."""
    enc_in, enc_out = _shape(encoder.input_shape), _shape(encoder.output_shape)
    dec_in, dec_out = _shape(decoder.input_shape), _shape(decoder.output_shape)
    expected_latent = (None, *EXPECTED_LATENT_SHAPE)
    expected = {
        "encoder_input": (None, 128, 128, 3), "encoder_output": expected_latent,
        "decoder_input": expected_latent, "decoder_output": (None, 128, 128, 3),
    }
    actual = {"encoder_input": enc_in, "encoder_output": enc_out,
              "decoder_input": dec_in, "decoder_output": dec_out}
    problems = [f"{k}: expected {expected[k]}, got {v}" for k, v in actual.items() if v != expected[k]]
    if problems:
        raise ValueError("Trained model shape mismatch: " + "; ".join(problems))
    return actual


def load_models(encoder_path: str | Path, decoder_path: str | Path) -> tuple[Any, Any, dict]:
    """Load actual Keras models and validate without modifying architecture."""
    ep, dp = Path(encoder_path), Path(decoder_path)
    missing = [str(p) for p in (ep, dp) if not p.is_file()]
    if missing:
        raise FileNotFoundError("Missing trained model file(s): " + ", ".join(missing))
    try:
        from tensorflow.keras.models import load_model
    except ImportError as exc:
        raise RuntimeError("TensorFlow is required to load the trained models") from exc
    encoder, decoder = load_model(ep, compile=False), load_model(dp, compile=False)
    return encoder, decoder, validate_models(encoder, decoder)
