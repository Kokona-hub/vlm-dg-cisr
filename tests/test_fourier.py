import numpy as np
from PIL import Image

from fs_pairgrpo.fourier import fourier_style_extrapolation


def _image(array):
    return Image.fromarray(np.asarray(array, dtype=np.uint8), mode="RGB")


def test_fourier_output_shape_and_mode():
    source = np.zeros((32, 40, 3), dtype=np.uint8)
    source[8:24, 12:28] = [220, 220, 220]
    donor = np.zeros((20, 20, 3), dtype=np.uint8)
    donor[:, ::2] = [180, 30, 30]
    result = fourier_style_extrapolation(_image(source), _image(donor), alpha=1.1)
    assert result.size == (40, 32)
    assert result.mode == "RGB"
    assert np.asarray(result).dtype == np.uint8


def test_alpha_zero_recovers_source_up_to_rounding():
    rng = np.random.default_rng(7)
    source = rng.integers(0, 256, size=(24, 24, 3), dtype=np.uint8)
    donor = rng.integers(0, 256, size=(24, 24, 3), dtype=np.uint8)
    result = np.asarray(fourier_style_extrapolation(_image(source), _image(donor), alpha=0.0))
    assert np.abs(result.astype(int) - source.astype(int)).max() <= 1
