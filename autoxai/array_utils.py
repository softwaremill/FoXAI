"""File contains functions to handle numpy arrays."""
import sys

import numpy as np
import torch


def convert_float_to_uint8(array: np.ndarray) -> np.ndarray:
    """Convert numpy array with float values to uint8 with scaled values.

    Args:
        array: Numpy array with float values.

    Returns:
        Numpy array with scaled values in uint8.
    """
    return (
        (
            (array - np.min(array))
            / ((np.max(array) - np.min(array)) + sys.float_info.epsilon)
        )
        * 255
    ).astype(np.uint8)


def reshape_and_convert_matrix(tensor: torch.Tensor) -> np.ndarray:
    """Reshape single tensor and convert it to numpy array.

    Input tensor is exptected to follow (B x C x H x W) shape with optional
    batch size as the first dimension. B stands for batch size, C stands for color
    dimension, H and W stand for height and width. As a result matrix will be
    reshaped into (H x W x C).

    Args:
        tensor: Tensor to convert.

    Returns:
        Reshaped numpy array.
    """
    # for single color, e.g. MNIST data copy one colour channel 3 times to simulate RGB
    if len(tensor.shape) == 4 and tensor.shape[1] == 1:
        tensor = tensor.expand(1, 3, tensor.shape[2], tensor.shape[3])
    elif len(tensor.shape) == 3 and tensor.shape[0] == 1:
        tensor = tensor.expand(3, tensor.shape[1], tensor.shape[2])

    # change dimension from (C x H x W) to (H x W x C)
    # where C is colour dimension, H and W are height and width dimensions
    matrix_np: np.ndarray = tensor.squeeze().detach().cpu().numpy()
    if len(tensor.shape) >= 3:
        matrix_np = np.transpose(matrix_np, (1, 2, 0))

    return matrix_np
