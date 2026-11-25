import numpy as np
import pytest
from PIL import Image
from src.codec import compress_image, decompress_image


class Encoder:
    def predict(self, batch, verbose=0):
        return np.repeat(batch.mean(axis=(1, 2, 3))[:, None, None, None], 16*16*32).reshape(1, 16, 16, 32)


class Decoder:
    def predict(self, latent, verbose=0):
        return np.repeat(latent.mean(axis=(1, 2, 3))[:, None, None, None], 128*128*3).reshape(1, 128, 128, 3)


def test_pipeline_roundtrip_shape(tmp_path):
    image = Image.new("RGB", (128, 128), (120, 100, 200))
    path = tmp_path / "test.facec"
    result = compress_image(image, Encoder(), path)
    assert np.array_equal(result["quantized"], np.zeros((16, 16, 32), dtype=np.uint8))
    decoded = decompress_image(path, Decoder())
    assert decoded["image"].shape == (128, 128, 3)


def test_encoder_bad_shape_rejected():
    class Bad:
        def predict(self, batch, verbose=0): return np.zeros((1, 4, 4, 64))
    with pytest.raises(ValueError): compress_image(Image.new("RGB", (128,128)), Bad(), "bad.facec")
