import numpy as np
from PIL import Image
from src.preprocessing import preprocess_image


def test_preprocess_rgb_shape_and_range():
    result = preprocess_image(Image.new("RGBA", (48, 72), (10, 20, 30, 255)))
    assert result.shape == (1, 128, 128, 3)
    assert result.dtype == np.float32
    assert 0 <= result.min() <= result.max() <= 1
