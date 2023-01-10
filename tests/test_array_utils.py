"""File contains unit tests for array_utils.py file."""
import numpy as np

from autoxai.array_utils import convert_float_to_uint8


def test_convert_float_to_uint8() -> None:
    """Test if function correctly scales values to uint8."""
    array = np.array(
        [
            [0.5, 0.15, 1.0, 25.0],
            [-25.0, 0.5, 0.15, 1.0],
        ],
        dtype=float,
    )
    expected_array = np.array(
        [
            [130, 128, 132, 255],
            [0, 130, 128, 132],
        ],
        dtype=np.uint8,
    )

    result = convert_float_to_uint8(array=array)

    np.testing.assert_array_equal(expected_array, result)
