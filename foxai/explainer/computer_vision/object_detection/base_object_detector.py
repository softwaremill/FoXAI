"""File contains abstract base ObjectDetector class."""
from abc import ABC, abstractmethod
from typing import List, Tuple

import numpy as np
import torch
from torch import nn

from foxai.explainer.computer_vision.object_detection.types import PredictionOutput
from foxai.explainer.computer_vision.object_detection.utils import resize_image


class BaseObjectDetector(nn.Module, ABC):
    """Base ObjectDetector class which returns predictions with logits to explain.

    Code based on https://github.com/pooya-mohammadi/yolov5-gradcam.
    """

    @abstractmethod
    def forward(
        self,
        image: torch.Tensor,
    ) -> Tuple[List[PredictionOutput], List[torch.Tensor]]:
        """Forward pass of the network.

        Args:
            image: Image to process.

        Returns:
            Tuple of 2 values, first is tuple of predictions containing bounding-boxes,
            class number, class name and confidence; second value is list of tensors
            with logits per each detection.
        """

    @staticmethod
    def preprocessing(
        img: np.ndarray,
        new_shape: Tuple[int, int] = (640, 640),
        change_original_ratio: bool = False,
        scaleup: bool = True,
    ) -> torch.Tensor:
        """Preprocess image before prediction.

        Preprocessing is a process consisting of steps:
        * adding batch dimension
        * resizing images to desired shapes
        * adjusting image channels to (B x C x H x W)
        * convertion to float

        Args:
            img: Image to preprocess.
            new_shape: Desired shape of image. Defaults to (640, 640).
            change_original_ratio: If resized image should have different height to
                width ratio than original image. Defaults to False.
            scaleup: If scale up image. Defaults to True.

        Returns:
            Tensor containing preprocessed image.
        """
        if len(img.shape) != 4:
            # add batch dimension
            img = np.expand_dims(img, axis=0)

        # resize all images from batch
        img = np.array(
            [
                resize_image(
                    image=im,
                    new_shape=new_shape,
                    change_original_ratio=change_original_ratio,
                    scaleup=scaleup,
                )
                for im in img
            ]
        )
        # convert array from (B x H x W x C) to (B x C x H x W)
        img = img.transpose((0, 3, 1, 2))
        img_tensor = torch.from_numpy(img)

        # convert from uint8 to float
        img_tensor = img_tensor / 255.0
        return img_tensor
