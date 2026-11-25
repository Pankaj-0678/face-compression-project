import numpy as np
import pytest
from src.quantization import quantize_latent, dequantize_latent


def test_quantization_round_trip_bounds():
    x = np.linspace(-2.5, 4.0, 100, dtype=np.float32).reshape(10, 10)
    q, lo, hi = quantize_latent(x)
    assert q.dtype == np.uint8 and q.shape == x.shape
    restored = dequantize_latent(q, lo, hi)
    assert np.max(np.abs(restored - x)) < (hi - lo) / 255


def test_constant_latent():
    q, lo, hi = quantize_latent(np.full((3,), 2.5))
    assert np.array_equal(q, np.zeros(3, dtype=np.uint8))
    assert np.array_equal(dequantize_latent(q, lo, hi), np.full(3, 2.5, dtype=np.float32))


def test_reject_bad_values():
    with pytest.raises(ValueError): quantize_latent(np.array([np.nan]))
