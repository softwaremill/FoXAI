"""File contains functions to handle numpy arrays."""
import sys

import numpy as np


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
