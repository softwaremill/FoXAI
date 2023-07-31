# pylint: disable = missing-class-docstring
from typing import List

import numpy as np
import pytest
import torch

from foxai.explainer.computer_vision.algorithm.xrai import (
    XRAI,
    _normalize_image,
    _resize_image,
    _unpack_segs_to_masks,
)


def test__unpack_segs_to_masks():
    """Check if _unpack_segs_to_masks function returns array of the same shape
    with boolean data type.
    """
    segment_list: List[np.ndarray] = [
        np.ones((5, 5), dtype=np.int64) * i for i in range(1, 5)
    ]
    result = _unpack_segs_to_masks(
        segment_list=segment_list,
    )
    assert len(result) == len(segment_list)
    for r in result:
        assert isinstance(r, np.ndarray)
        assert r.dtype == bool


def test___normalize_image():
    """Test if __normalize_image will rescale images to given value range."""
    image_batch = np.random.random(size=(10, 28, 28, 3))

    result = _normalize_image(image_batch=image_batch, value_range=[10, 20])

    assert isinstance(result, np.ndarray)
    np.testing.assert_almost_equal(result.max(), 20)
    np.testing.assert_almost_equal(result.min(), 10)


def test___resize_image():
    """Test if __resize_image will reshape images to given shape."""
    resize_shape = (224, 224)
    image_batch = np.zeros((10, 28, 28, 3))

    result = _resize_image(image_batch=image_batch, resize_shape=resize_shape)

    assert isinstance(result, np.ndarray)
    assert result.shape == (10, 224, 224, 3)


class TestXRAI:
    """XRAI test class."""

    def test___validate_baselines_shold_pass(self):
        """Baseline validation function should check if every element of
        baselines batch matches input_data shape.
        """
        shape = (5, 10)
        batch_size = 5
        input_data = torch.randn(shape)
        baselines = torch.randn((batch_size,) + shape)
        XRAI._validate_baselines(  # pylint: disable = (protected-access)
            input_data=input_data, baselines=baselines
        )

    def test___validate_baselines_should_raise_exception(self):
        """Baseline validation function should check if every element of
        baselines batch matches input_data shape and raise ValueError if
        don't.
        """
        batch_size = 5
        input_data = torch.randn((5, 10))
        baselines = torch.randn((batch_size,) + (10, 5))
        with pytest.raises(ValueError):
            XRAI._validate_baselines(  # pylint: disable = (protected-access)
                input_data=input_data, baselines=baselines
            )

    def test___aggregate_single_result_list_into_batch(self):
        """List of arrays representing single sample XRAI result should be
        merged into batch array.
        """
        attribute_dimension = 5
        batch_size = 5
        result_list: List[np.ndarray] = []
        for _ in range(0, batch_size):
            result_list.append(np.ones((attribute_dimension, attribute_dimension)))

        result = XRAI._aggregate_single_result_list_into_batch(  # pylint: disable = (protected-access)
            result_list=result_list
        )
        assert isinstance(result, np.ndarray)
        assert result.shape == (batch_size, 1, attribute_dimension, attribute_dimension)

    def test___aggregate_single_result_list_into_batch_for_nested_list(self):
        """Nested list of arrays representing single sample XRAI result should be
        merged into batch array.
        """
        attribute_dimension = 5
        batch_size = 5
        nested_size = 10
        result_list: List[List[np.ndarray]] = []
        for _ in range(0, batch_size):
            inner_list = []
            for _ in range(0, nested_size):
                inner_list.append(np.ones((attribute_dimension, attribute_dimension)))

            result_list.append(inner_list)

        result = XRAI._aggregate_single_result_list_into_batch(  # pylint: disable = (protected-access)
            result_list=result_list
        )
        assert isinstance(result, np.ndarray)
        assert result.shape == (
            batch_size,
            nested_size,
            attribute_dimension,
            attribute_dimension,
        )
