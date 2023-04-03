"""
Based on code: https://github.com/pooya-mohammadi/yolov5-gradcam.
"""

from typing import Tuple

import cv2
import numpy as np
import torch


def box_iou(box1: torch.Tensor, box2: torch.Tensor) -> torch.Tensor:
    """
    Return intersection-over-union (Jaccard index) of boxes.

    Both sets of boxes are expected to be in (x1, y1, x2, y2) format.
    Original code: https://github.com/pytorch/vision/blob/master/torchvision/ops/boxes.py

    Args:
        box1: First box.
        box2: Second box.
    Returns:
        The NxM matrix containing the pairwise IoU values for every element
            in boxes1 and boxes2
    """

    def box_area(box):
        # box = 4xn
        return (box[2] - box[0]) * (box[3] - box[1])

    area1 = box_area(box1.T)
    area2 = box_area(box2.T)

    # inter(N,M) = (rb(N,M,2) - lt(N,M,2)).clamp(0).prod(2)
    inter = (
        (
            torch.min(box1[:, None, 2:], box2[:, 2:])
            - torch.max(box1[:, None, :2], box2[:, :2])
        )
        .clamp(0)
        .prod(2)
    )
    return inter / (
        area1[:, None] + area2 - inter
    )  # iou = inter / (area1 + area2 - inter)


def xywh2xyxy(x: torch.Tensor) -> torch.Tensor:
    """Convert tensor in [x, y, w, h] to [x1, y1, x2, y2].

    Args:
        x: Tensor in [x, y, w, h] to convert.

    Returns:
        Tensor in [x1, y1, x2, y2].
    """
    # Convert nx4 boxes from [x, y, w, h] to [x1, y1, x2, y2] where xy1=top-left, xy2=bottom-right
    y = x.clone() if isinstance(x, torch.Tensor) else np.copy(x)
    y[:, 0] = x[:, 0] - x[:, 2] / 2  # top left x
    y[:, 1] = x[:, 1] - x[:, 3] / 2  # top left y
    y[:, 2] = x[:, 0] + x[:, 2] / 2  # bottom right x
    y[:, 3] = x[:, 1] + x[:, 3] / 2  # bottom right y
    return y


def resize_image(
    image: torch.Tensor,
    new_shape: Tuple[int, int] = (640, 640),
    change_original_ratio: bool = False,
    scaleup: bool = True,
) -> torch.Tensor:
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
