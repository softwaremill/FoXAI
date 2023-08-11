"""File contains unit tests for array_utils.py file."""
import numpy as np
import pytest
import torch

from foxai.array_utils import (
    convert_standardized_float_to_uint8,
    normalize_attributes,
    resize_attributes,
    retain_only_positive,
    standardize_array,
    transpose_color_first_in_array_np,
    transpose_color_first_in_array_pt,
    transpose_color_last_in_array_np,
    transpose_color_last_in_array_pt,
)


def test_standardize_array() -> None:
    """Test if function correctly scales values to range [0-1]."""
    array = np.array(
        [
            [0.5, 0.15, 1.0, 25.0],
            [-25.0, 0.5, 0.15, 1.0],
        ],
        dtype=float,
    )
    expected_array = np.array(
        [
            [0.51, 0.503, 0.52, 1.0],
            [0, 0.51, 0.503, 0.52],
        ],
        dtype=float,
    )

    result = standardize_array(array=array)
    np.testing.assert_array_equal(expected_array, result)


def test_convert_standardized_float_to_uint8() -> None:
    """Test if function correctly scales values to range [0-255]."""
    array = np.array(
        [
            [0.51, 0.503, 0.52, 1.0],
            [0, 0.51, 0.503, 0.52],
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

    result = convert_standardized_float_to_uint8(array=array)
    np.testing.assert_array_equal(expected_array, result)


def test_convert_standardized_float_to_uint8_should_raise_exception_when_incorrect_data_type() -> None:
    """Test if function raises ValueError if passed array is not of type np.float."""
    array = np.array(
        [
            [130, 128, 132, 255],
            [0, 130, 128, 132],
        ],
        dtype=np.uint8,
    )

    with pytest.raises(ValueError):
        _ = convert_standardized_float_to_uint8(array=array)


def test_transpose_color_last_in_array_np_should_add_additional_dimensions() -> None:
    array = np.array(
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
    )

    expected_array = np.array(
        [
            [
                [1],
                [2],
                [3],
                [4],
                [5],
                [6],
            ],
            [
                [7],
                [8],
                [9],
                [10],
                [11],
                [12],
            ],
            [
                [13],
                [14],
                [15],
                [16],
                [17],
                [18],
            ],
            [
                [19],
                [20],
                [21],
                [22],
                [23],
                [24],
            ],
            [
                [25],
                [26],
                [27],
                [28],
                [29],
                [30],
            ],
            [
                [31],
                [32],
                [33],
                [34],
                [35],
                [36],
            ],
        ]
    )

    assert array.shape == (1, 6, 6)
    assert expected_array.shape == (6, 6, 1)

    array = transpose_color_last_in_array_np(array=array)

    assert array.shape == expected_array.shape
    np.testing.assert_array_equal(array, expected_array)


def test_transpose_color_last_in_array_np_should_not_additional_dimensions() -> None:
    array = np.array(
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

    assert array.shape == (3, 6, 6)
    assert expected_array.shape == (6, 6, 3)

    array = transpose_color_last_in_array_np(array=array)

    assert array.shape == expected_array.shape
    np.testing.assert_array_equal(array, expected_array)


def test_transpose_color_last_in_array_np_with_batch_size_should_work() -> None:
    """Test if function transposes data with shapes (B x C x H x W)"""
    array = np.array(
        [
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
            ]
        ],
    )

    expected_array = np.array(
        [
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
        ]
    )

    assert array.shape == (1, 3, 6, 6)
    assert expected_array.shape == (1, 6, 6, 3)

    array = transpose_color_last_in_array_np(array=array)

    assert array.shape == expected_array.shape
    np.testing.assert_array_equal(array, expected_array)


def test_transpose_color_last_in_array_np_with_should_raise_exception_for_2d_data() -> None:
    """Test if function raises RuntimeError when data has less than 3 dimensions."""
    array = np.zeros((6, 6))

    with pytest.raises(RuntimeError):
        _ = transpose_color_last_in_array_np(array=array)


def test_transpose_color_last_in_array_np_with_should_raise_exception_for_5d_data() -> None:
    """Test if function raises RuntimeError when data has more than 4 dimensions."""
    array = np.zeros((1, 1, 6, 6, 3))

    with pytest.raises(RuntimeError):
        _ = transpose_color_last_in_array_np(array=array)


def test_transpose_color_first_in_array_np_should_add_additional_dimensions() -> None:
    array = np.array(
        [
            [
                [1],
                [2],
                [3],
                [4],
                [5],
                [6],
            ],
            [
                [7],
                [8],
                [9],
                [10],
                [11],
                [12],
            ],
            [
                [13],
                [14],
                [15],
                [16],
                [17],
                [18],
            ],
            [
                [19],
                [20],
                [21],
                [22],
                [23],
                [24],
            ],
            [
                [25],
                [26],
                [27],
                [28],
                [29],
                [30],
            ],
            [
                [31],
                [32],
                [33],
                [34],
                [35],
                [36],
            ],
        ]
    )

    expected_array = np.array(
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
    )

    assert array.shape == (6, 6, 1)
    assert expected_array.shape == (1, 6, 6)

    array = transpose_color_first_in_array_np(array=array)

    assert array.shape == expected_array.shape
    np.testing.assert_array_equal(array, expected_array)


def test_transpose_color_first_in_array_np_should_not_additional_dimensions() -> None:
    array = np.array(
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

    expected_array = np.array(
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
    )

    assert array.shape == (6, 6, 3)
    assert expected_array.shape == (3, 6, 6)

    array = transpose_color_first_in_array_np(array=array)

    assert array.shape == expected_array.shape
    np.testing.assert_array_equal(array, expected_array)


def test_transpose_color_first_in_array_np_with_batch_size_should_work() -> None:
    """Test if function transposes data with shapes (B x H x W x C)"""
    array = np.array(
        [
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
        ]
    )

    expected_array = np.array(
        [
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
            ]
        ],
    )

    assert array.shape == (1, 6, 6, 3)
    assert expected_array.shape == (1, 3, 6, 6)

    array = transpose_color_first_in_array_np(array=array)

    assert array.shape == expected_array.shape
    np.testing.assert_array_equal(array, expected_array)


def test_transpose_color_first_in_array_np_with_should_raise_exception_for_2d_data() -> None:
    """Test if function raises RuntimeError when data has less than 3 dimensions."""
    array = np.zeros((6, 6))

    with pytest.raises(RuntimeError):
        _ = transpose_color_first_in_array_np(array=array)


def test_transpose_color_first_in_array_np_with_should_raise_exception_for_5d_data() -> None:
    """Test if function raises RuntimeError when data has more than 4 dimensions."""
    array = np.zeros((1, 1, 6, 6, 3))

    with pytest.raises(RuntimeError):
        _ = transpose_color_first_in_array_np(array=array)


def test_transpose_color_last_in_array_pt_should_add_additional_dimensions() -> None:
    array = torch.tensor(
        np.array(
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
        )
    )

    expected_array = torch.tensor(
        np.array(
            [
                [
                    [1],
                    [2],
                    [3],
                    [4],
                    [5],
                    [6],
                ],
                [
                    [7],
                    [8],
                    [9],
                    [10],
                    [11],
                    [12],
                ],
                [
                    [13],
                    [14],
                    [15],
                    [16],
                    [17],
                    [18],
                ],
                [
                    [19],
                    [20],
                    [21],
                    [22],
                    [23],
                    [24],
                ],
                [
                    [25],
                    [26],
                    [27],
                    [28],
                    [29],
                    [30],
                ],
                [
                    [31],
                    [32],
                    [33],
                    [34],
                    [35],
                    [36],
                ],
            ]
        )
    )

    assert array.shape == (1, 6, 6)
    assert expected_array.shape == (6, 6, 1)

    array = transpose_color_last_in_array_pt(array=array)

    assert array.shape == expected_array.shape
    torch.testing.assert_allclose(array, expected_array)


def test_transpose_color_last_in_array_pt_should_not_additional_dimensions() -> None:
    array = torch.tensor(
        np.array(
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
        )
    )

    expected_array = torch.tensor(
        np.array(
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
    )

    assert array.shape == (3, 6, 6)
    assert expected_array.shape == (6, 6, 3)

    array = transpose_color_last_in_array_pt(array=array)

    assert array.shape == expected_array.shape
    torch.testing.assert_allclose(array, expected_array)


def test_transpose_color_last_in_array_pt_with_batch_size_should_work() -> None:
    """Test if function transposes data with shapes (B x C x H x W)"""
    array = torch.tensor(
        np.array(
            [
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
                ]
            ],
        )
    )

    expected_array = torch.tensor(
        np.array(
            [
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
            ]
        )
    )

    assert array.shape == (1, 3, 6, 6)
    assert expected_array.shape == (1, 6, 6, 3)

    array = transpose_color_last_in_array_pt(array=array)

    assert array.shape == expected_array.shape
    torch.testing.assert_allclose(array, expected_array)


def test_transpose_color_first_in_array_pt_should_add_additional_dimensions() -> None:
    array = torch.tensor(
        np.array(
            [
                [
                    [1],
                    [2],
                    [3],
                    [4],
                    [5],
                    [6],
                ],
                [
                    [7],
                    [8],
                    [9],
                    [10],
                    [11],
                    [12],
                ],
                [
                    [13],
                    [14],
                    [15],
                    [16],
                    [17],
                    [18],
                ],
                [
                    [19],
                    [20],
                    [21],
                    [22],
                    [23],
                    [24],
                ],
                [
                    [25],
                    [26],
                    [27],
                    [28],
                    [29],
                    [30],
                ],
                [
                    [31],
                    [32],
                    [33],
                    [34],
                    [35],
                    [36],
                ],
            ]
        )
    )

    expected_array = torch.tensor(
        np.array(
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
        )
    )

    assert array.shape == (6, 6, 1)
    assert expected_array.shape == (1, 6, 6)

    array = transpose_color_first_in_array_pt(array=array)

    assert array.shape == expected_array.shape
    torch.testing.assert_allclose(array, expected_array)


def test_transpose_color_first_in_array_pt_should_not_additional_dimensions() -> None:
    array = torch.tensor(
        np.array(
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
    )

    expected_array = torch.tensor(
        np.array(
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
        )
    )

    assert array.shape == (6, 6, 3)
    assert expected_array.shape == (3, 6, 6)

    array = transpose_color_first_in_array_pt(array=array)

    assert array.shape == expected_array.shape
    torch.testing.assert_allclose(array, expected_array)


def test_transpose_color_first_in_array_pt_with_batch_size_should_work() -> None:
    """Test if function transposes data with shapes (B x H x W x C)"""
    array = torch.tensor(
        np.array(
            [
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
            ]
        )
    )

    expected_array = torch.tensor(
        np.array(
            [
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
                ]
            ],
        )
    )

    assert array.shape == (1, 6, 6, 3)
    assert expected_array.shape == (1, 3, 6, 6)

    array = transpose_color_first_in_array_pt(array=array)

    assert array.shape == expected_array.shape
    torch.testing.assert_allclose(array, expected_array)


def test_transpose_color_last_in_array_pt_with_should_raise_exception_for_2d_data() -> None:
    """Test if function raises RuntimeError when data has less than 3 dimensions."""
    array = torch.zeros((6, 6))

    with pytest.raises(RuntimeError):
        _ = transpose_color_last_in_array_pt(array=array)


def test_transpose_color_last_in_array_pt_with_should_raise_exception_for_5d_data() -> None:
    """Test if function raises RuntimeError when data has more than 4 dimensions."""
    array = torch.zeros((1, 1, 6, 6, 3))

    with pytest.raises(RuntimeError):
        _ = transpose_color_last_in_array_pt(array=array)


def test_transpose_color_first_in_array_pt_with_should_raise_exception_for_2d_data() -> None:
    """Test if function raises RuntimeError when data has less than 3 dimensions."""
    array = torch.zeros((6, 6))

    with pytest.raises(RuntimeError):
        _ = transpose_color_first_in_array_pt(array=array)


def test_transpose_color_first_in_array_pt_with_should_raise_exception_for_5d_data() -> None:
    """Test if function raises RuntimeError when data has more than 4 dimensions."""
    array = torch.zeros((1, 1, 6, 6, 3))

    with pytest.raises(RuntimeError):
        _ = transpose_color_first_in_array_pt(array=array)


def test_resize_attributes() -> None:
    """Test if function resizes array to desired shape."""
    array = np.array(
        [
            [1.0, 2.0],
            [3.0, 4.0],
        ]
    )

    result = resize_attributes(
        attributes=array,
        dest_height=4,
        dest_width=3,
    )

    assert result.shape == (4, 3)


def test_normalize_attributes_should_return_the_same_array_if_has_2_dimensions() -> None:
    """Test if function returns unmodified array if it has only 2 dimensions."""
    array = np.array(
        [
            [1, 2, 3],
            [4, 5, 6],
            [7, 8, 9],
        ]
    )

    result = normalize_attributes(
        attributes=array,
    )

    np.testing.assert_array_equal(result, array)


def test_normalize_attributes_should_return_mean_on_the_array_if_has_3_dimensions() -> None:
    """Test if function returns mean array over first dimension if it has 3 dimensions."""
    array = np.array(
        [
            [
                [1, 2, 3],
                [4, 5, 6],
                [7, 8, 9],
            ],
            [
                [1, 2, 3],
                [4, 5, 6],
                [7, 8, 9],
            ],
            [
                [1, 2, 3],
                [4, 5, 6],
                [7, 8, 9],
            ],
        ]
    )

    expected_array = np.array(
        [
            [1, 2, 3],
            [4, 5, 6],
            [7, 8, 9],
        ]
    )

    assert len(array.shape) == 3
    result = normalize_attributes(
        attributes=array,
    )

    assert len(result.shape) == 2
    np.testing.assert_array_equal(result, expected_array)


def test_normalize_attributes_should_raise_error_if_has_more_than_3_dimensions() -> None:
    """Test if function raises ValueError if array has more than 3 dimensions."""
    array = np.array(
        [
            [
                [
                    [1, 2, 3],
                    [4, 5, 6],
                    [7, 8, 9],
                ]
            ]
        ]
    )

    with pytest.raises(ValueError):
        _ = normalize_attributes(
            attributes=array,
        )


def test_retain_only_positive() -> None:
    """Test if function replaces all negative numbers with zero."""
    array = np.array(
        [0, -1, 1, -2, 2],
    )
    excpected_array = np.array(
        [0, 0, 1, 0, 2],
    )
    result = retain_only_positive(array=array)

    np.testing.assert_array_equal(result, excpected_array)
