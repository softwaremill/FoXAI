"""File contains functions to handle numpy arrays."""
import sys

import cv2
import numpy as np
import torch

from foxai.types import AttributionsType


def standardize_array(array: np.ndarray) -> np.ndarray:
    """Standardize array values to range [0-1].

    Args:
        array: Numpy array of floats.

    Returns:
        Numpy array with scaled values.

    Raises:
        ValueError: if array is not of type np.float.
    """
    if not array.dtype == np.dtype(float):
        raise ValueError(
            f"Array should be of type: np.float, current type: {array.dtype}"
        )

    return (array - np.min(array)) / (
        (np.max(array) - np.min(array)) + sys.float_info.epsilon
    )


def convert_standardized_float_to_uint8(array: np.ndarray) -> np.ndarray:
    """Convert float standardize float array to uint8 with values scaling.

    Args:
        array: Numpy array of floats.

    Returns:
        Numpy array with scaled values with type np.uint8.

    Raises:
        ValueError: if array is not of type np.float.
    """
    if not array.dtype == np.dtype(float):
        raise ValueError(
            f"Array should be of type: np.float, current type: {array.dtype}"
        )

    return (array * 255).astype(np.uint8)


def retain_only_positive(array: np.ndarray) -> np.ndarray:
    """Retain only positive values from array.

    Args:
        array: Array.

    Returns:
        Array with negative values replaced by zero.
    """
    array[array < 0] = 0
    return array


def normalize_attributes(
    attributes: np.ndarray,
) -> np.ndarray:
    """Normalize attributes.

    For attributes with color dimension calculate mean over all colors.

    Args:
        attributes: Array of attributes.

    Returns:
        Single channel array of attributes.

    Raises:
        ValueError: if shape of attribute array is incorrect.
    """
    ret_array: np.ndarray
    if len(attributes.shape) == 3:
        # if we have attributes with shape (C x H x W)
        # where C is color, W is width and H is height dimension
        # calculate mean over attributes for all colors
        ret_array = np.mean(attributes, axis=0)
    elif len(attributes.shape) == 2:
        ret_array = attributes
    else:
        raise ValueError(f"Incorrect shape of attributions: {attributes.shape}")

    return ret_array


def resize_attributes(
    attributes: np.ndarray,
    dest_width: int,
    dest_height: int,
) -> np.ndarray:
    """Resize attributes to match desired shape.

    Args:
        attributes: Array of attributes.
        dest_width: Desired width of attributes array.
        dest_height: Desired height of attributes array.

    Returns:
        Resized attributes array.
    """
    # resize attributes matrix to match input image
    single_channel_attributes: np.ndarray = np.array(
        cv2.resize(
            attributes,
            (dest_width, dest_height),
        ),
        dtype=np.dtype(float),
    )

    return single_channel_attributes


def transpose_color_last_in_array_np(array: np.ndarray) -> np.ndarray:
    """Transpose color from the first to the last dimension of an array.

    More technically transpose array from (C x H x W) to (H x W x C) shape or from
    (B x C x H x W) to (B x H x W x C).

    B stands for batch size, C stands for color, H stands for height and W
    stands for width.

    Args:
        array: Array of shape (C x H x W) or (B x C x H x W)).

    Returns:
        Array of shape (H x W x C) or (B x H x W x C)).
    """
    if len(array.shape) == 3:
        return np.transpose(array, (1, 2, 0))
    elif len(array.shape) == 4:
        return np.transpose(array, (0, 2, 3, 1))
    else:
        raise RuntimeError(
            f"Wrong shape of array to transpose. Expected 3 or 4 dimensions but got: {len(array.shape)} dimensions."
        )


def transpose_color_first_in_array_np(array: np.ndarray) -> np.ndarray:
    """Transpose color from the last to the first dimension of an array.

    More technically transpose array from (H x W x C) to (C x H x W) shape or from
    (B x H x W x C) to (B x C x H x W).

    B stands for batch size, C stands for color, H stands for height and W
    stands for width.

    Args:
        array: Array of shape (H x W x C) or (B x H x W x C)).

    Returns:
        Array of shape (C x H x W) or (B x C x H x W)).
    """
    if len(array.shape) == 3:
        return np.transpose(array, (2, 0, 1))
    elif len(array.shape) == 4:
        return np.transpose(array, (0, 3, 1, 2))
    else:
        raise RuntimeError(
            f"Wrong shape of array to transpose. Expected 3 or 4 dimensions but got: {len(array.shape)} dimensions."
        )


def transpose_color_last_in_array_pt(array: torch.Tensor) -> torch.Tensor:
    """Transpose color from the first to the last dimension of an tensor.

    More technically transpose tensor from (C x H x W) to (H x W x C) shape or from
    (B x C x H x W) to (B x H x W x C).

    B stands for batch size, C stands for color, H stands for height and W
    stands for width.

    Args:
        array: Tensor of shape (C x H x W) or (B x C x H x W)).

    Returns:
        Tensor of shape (H x W x C) or (B x H x W x C)).
    """
    if len(array.shape) == 3:
        return array.permute(1, 2, 0)
    elif len(array.shape) == 4:
        return array.permute(0, 2, 3, 1)
    else:
        raise RuntimeError(
            f"Wrong shape of array to transpose. Expected 3 or 4 dimensions but got: {len(array.shape)} dimensions."
        )


def transpose_color_first_in_array_pt(array: torch.Tensor) -> torch.Tensor:
    """Transpose color from the last to the first dimension of a tensor.

    More technically transpose tensor from (H x W x C) to (C x H x W) shape or from
    (B x H x W x C) to (B x C x H x W).

    B stands for batch size, C stands for color, H stands for height and W
    stands for width.

    Args:
        array: Tensor of shape (H x W x C) or (B x H x W x C)).

    Returns:
        Tensor of shape (C x H x W) or (B x C x H x W)).
    """
    if len(array.shape) == 3:
        return array.permute(2, 0, 1)
    elif len(array.shape) == 4:
        return array.permute(0, 3, 1, 2)
    else:
        raise RuntimeError(
            f"Wrong shape of array to transpose. Expected 3 or 4 dimensions but got: {len(array.shape)} dimensions."
        )


def validate_result(attributions: AttributionsType) -> None:
    """Validate calculated attributes.

    Args:
        attributions: Tensor with calculated attributions.

    Raises:
        RuntimeError if tensor is empty.
    """
    if attributions.shape[0] == 0:
        raise RuntimeError(
            "Error occured during attribution calculation. "
            + "Make sure You are applying this method to CNN network.",
        )
