import numpy as np

from slam.global_place_descriptor import (
    DescriptorSpec,
    l2_normalize,
    preprocess_descriptor_image,
)


def test_descriptor_preprocessing_uses_configured_shape():
    image = np.full((20, 30, 3), 127, dtype=np.uint8)
    spec = DescriptorSpec("test", None, 32, 48)
    tensor = preprocess_descriptor_image(image, spec)
    assert tensor.shape == (1, 3, 32, 48)
    assert tensor.dtype == np.float32
    assert tensor.flags["C_CONTIGUOUS"]


def test_l2_normalize_returns_unit_descriptor():
    result = l2_normalize([3.0, 4.0])
    assert np.allclose(result, [0.6, 0.8])
    assert np.isclose(np.linalg.norm(result), 1.0)


def test_l2_normalize_rejects_zero_descriptor():
    try:
        l2_normalize([0.0, 0.0])
    except RuntimeError as error:
        assert "zero/invalid" in str(error)
    else:
        raise AssertionError("zero descriptor was accepted")
