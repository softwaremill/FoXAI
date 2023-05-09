"""
File contains GradCAM algorithm for object detection task.

Code based on https://github.com/pooya-mohammadi/yolov5-gradcam.
"""

from typing import List

import torch

from foxai.explainer.computer_vision.image_classification.gradcam import (
    LayerBaseGradCAM,
)
from foxai.explainer.computer_vision.object_detection.base_object_detector import (
    BaseObjectDetector,
)
from foxai.explainer.computer_vision.object_detection.types import ObjectDetectionOutput


class LayerGradCAMObjectDetection(LayerBaseGradCAM):
    """Layer GradCAM for object detection task."""

    def __init__(
        self,
        model: BaseObjectDetector,
        target_layer: torch.nn.Module,
    ):
        super().__init__(target_layer=target_layer)
        self.model = model

    def forward(
        self,
        input_img: torch.Tensor,
    ) -> ObjectDetectionOutput:
        """Forward pass of GradCAM aglorithm.

        Args:
            input_img: Input image with shape of (B, C, H, W).

        Returns:
            ObjectDetectionOutput object.
        """
        saliency_maps: List[torch.Tensor] = []
        _, _, height, width = input_img.size()
        predictions, logits = self.model.forward(input_img)
        for logit, cls in zip(logits[0], [p.class_number for p in predictions]):
            score = logit[cls]
            # clear gradients
            self.model.zero_grad()

            # calculate gradients
            score.backward(retain_graph=True)

            saliency_maps.append(
                self.get_saliency_map(
                    height=height,
                    width=width,
                    gradients=self.gradients,
                    activations=self.activations,
                )
            )
        return ObjectDetectionOutput(
            saliency_maps=saliency_maps,
            logits=logits,
            predictions=predictions,
        )
