import numpy as np
import pytest
from src.huffman import encode, decode


@pytest.mark.parametrize("arr", [np.array([], dtype=np.uint8), np.array([5] * 100, dtype=np.uint8),
                                  np.arange(256, dtype=np.uint8), np.random.default_rng(9).integers(0, 32, 4096, dtype=np.uint8)])
def test_huffman_exact_roundtrip(arr):
    payload, bits, book = encode(arr)
    assert np.array_equal(arr, decode(payload, bits, book, arr.size))


def test_incomplete_rejected():
    with pytest.raises(ValueError): decode(b"\x00", 1, {1: "00", 2: "01"}, 1)
