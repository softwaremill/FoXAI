"""File contains unit tests for array_utils.py file."""
import numpy as np
import torch

from autoxai.array_utils import convert_float_to_uint8, reshape_and_convert_matrix


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


def test_reshape_and_convert_matrix_should_add_additional_dimensions() -> None:
    tensor = torch.tensor(
        [
            [
                [1, 2, 3, 4, 5, 6],
                [7, 8, 9, 10, 11, 12],
                [13, 14, 15, 16, 17, 18],
                [19, 20, 21, 22, 23, 24],
                [25, 26, 27, 28, 29, 30],
                [31, 32, 33, 34, 35, 36],
            ],
        ],
        dtype=torch.uint8,
    )

    expected_array = np.array(
        [
            [
                [1, 1, 1],
                [2, 2, 2],
                [3, 3, 3],
                [4, 4, 4],
                [5, 5, 5],
                [6, 6, 6],
            ],
            [
                [7, 7, 7],
                [8, 8, 8],
                [9, 9, 9],
                [10, 10, 10],
                [11, 11, 11],
                [12, 12, 12],
            ],
            [
                [13, 13, 13],
                [14, 14, 14],
                [15, 15, 15],
                [16, 16, 16],
                [17, 17, 17],
                [18, 18, 18],
            ],
            [
                [19, 19, 19],
                [20, 20, 20],
                [21, 21, 21],
                [22, 22, 22],
                [23, 23, 23],
                [24, 24, 24],
            ],
            [
                [25, 25, 25],
                [26, 26, 26],
                [27, 27, 27],
                [28, 28, 28],
                [29, 29, 29],
                [30, 30, 30],
            ],
            [
                [31, 31, 31],
                [32, 32, 32],
                [33, 33, 33],
                [34, 34, 34],
                [35, 35, 35],
                [36, 36, 36],
            ],
        ]
    )

    assert tensor.shape == (1, 6, 6)
    assert expected_array.shape == (6, 6, 3)

    array = reshape_and_convert_matrix(tensor=tensor)

    assert isinstance(array, np.ndarray)
    assert array.shape == expected_array.shape
    np.testing.assert_array_equal(array, expected_array)


def test_reshape_and_convert_matrix_should_not_additional_dimensions() -> None:
    tensor = torch.tensor(
        [
            [
                [1, 2, 3, 4, 5, 6],
                [7, 8, 9, 10, 11, 12],
                [13, 14, 15, 16, 17, 18],
                [19, 20, 21, 22, 23, 24],
                [25, 26, 27, 28, 29, 30],
                [31, 32, 33, 34, 35, 36],
            ],
            [
                [37, 38, 39, 40, 41, 42],
                [43, 44, 45, 46, 47, 48],
                [49, 50, 51, 52, 53, 54],
                [55, 56, 57, 58, 59, 60],
                [61, 62, 63, 64, 65, 66],
                [67, 68, 69, 70, 71, 72],
            ],
            [
                [73, 74, 75, 76, 77, 78],
                [79, 80, 81, 82, 83, 84],
                [85, 86, 87, 88, 89, 90],
                [91, 92, 93, 94, 95, 96],
                [97, 98, 99, 100, 101, 102],
                [103, 104, 105, 106, 107, 108],
            ],
        ],
        dtype=torch.uint8,
    )

    expected_array = np.array(
        [
            [
                [1, 37, 73],
                [2, 38, 74],
                [3, 39, 75],
                [4, 40, 76],
                [5, 41, 77],
                [6, 42, 78],
            ],
            [
                [7, 43, 79],
                [8, 44, 80],
                [9, 45, 81],
                [10, 46, 82],
                [11, 47, 83],
                [12, 48, 84],
            ],
            [
                [13, 49, 85],
                [14, 50, 86],
                [15, 51, 87],
                [16, 52, 88],
                [17, 53, 89],
                [18, 54, 90],
            ],
            [
                [19, 55, 91],
                [20, 56, 92],
                [21, 57, 93],
                [22, 58, 94],
                [23, 59, 95],
                [24, 60, 96],
            ],
            [
                [25, 61, 97],
                [26, 62, 98],
                [27, 63, 99],
                [28, 64, 100],
                [29, 65, 101],
                [30, 66, 102],
            ],
            [
                [31, 67, 103],
                [32, 68, 104],
                [33, 69, 105],
                [34, 70, 106],
                [35, 71, 107],
                [36, 72, 108],
            ],
        ]
    )

    assert tensor.shape == (3, 6, 6)
    assert expected_array.shape == (6, 6, 3)

    array = reshape_and_convert_matrix(tensor=tensor)
    assert isinstance(array, np.ndarray)
    assert array.shape == expected_array.shape
    np.testing.assert_array_equal(array, expected_array)
