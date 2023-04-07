"""
File contains functions to handle images, coordinates and calcualte IoU.

Based on code: https://github.com/pooya-mohammadi/yolov5-gradcam.
"""

from typing import Tuple

import cv2
import numpy as np


def resize_image(
    image: np.ndarray,
    new_shape: Tuple[int, int] = (640, 640),
    change_original_ratio: bool = False,
    scaleup: bool = True,
) -> np.ndarray:
    """Resize image to given shape.

    Args:
        image: Image to resize.
        new_shape: Desired shape of image. Defaults to (640, 640).
        change_original_ratio: If resized image should have different height to
            width ratio than original image. Defaults to False.
        scaleup: If scale up image. Defaults to True.

    Returns:
        Resized image.
    """
    # Resize and pad image while meeting stride-multiple constraints
    # image has shape (H x W x C)
    shape = image.shape[:2]  # current shape [height, width]

    # Scale ratio (new / old)
    # get minimum of width and height ratios
    r = min(new_shape[0] / shape[0], new_shape[1] / shape[1])
    if not scaleup:  # only scale down, do not scale up (for better val mAP)
        r = min(r, 1.0)

    # do not change the width/height ratio of the image, assign the same ratio
    new_unpad = int(round(shape[1] * r)), int(round(shape[0] * r))
    if change_original_ratio:  # stretch
        new_unpad = (new_shape[1], new_shape[0])

    # if current shape is different than desired shape call resize function
    if shape[::-1] != new_unpad:  # resize
        image = cv2.resize(image, new_unpad, interpolation=cv2.INTER_LINEAR)

    return image
