import numpy as np
import pytest
from src.huffman import encode
from src.facec_format import write_facec_file, read_facec_file


def test_file_roundtrip(tmp_path):
    q = np.arange(8192, dtype=np.uint16).astype(np.uint8).reshape(16, 16, 32)
    payload, bits, book = encode(q)
    path = tmp_path / "sample.facec"
    write_facec_file(path, q, -1.25, 2.75, payload, bits, book)
    meta, actual = read_facec_file(path)
    assert meta["latent_shape"] == [16, 16, 32]
    assert np.array_equal(q, actual)


@pytest.mark.parametrize("data", [b"BAD!", b"FAC1\xff\xff\xff\xff"])
def test_bad_file_rejected(data):
    with pytest.raises(ValueError): read_facec_file(data)


def test_corruption_rejected(tmp_path):
    q = np.arange(32, dtype=np.uint8).reshape(2, 4, 4)
    payload, bits, book = encode(q)
    path = tmp_path / "sample.facec"
    write_facec_file(path, q, 0, 31, payload, bits, book)
    raw = bytearray(path.read_bytes()); raw[-1] ^= 0xff; path.write_bytes(raw)
    with pytest.raises(ValueError): read_facec_file(path)
