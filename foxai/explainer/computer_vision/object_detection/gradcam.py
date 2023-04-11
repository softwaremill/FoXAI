"""
File contains GradCAM algorithm for object detection task.

Code based on https://github.com/pooya-mohammadi/yolov5-gradcam.
"""

from typing import Dict, List

import torch
import torch.nn.functional as F

from foxai.explainer.computer_vision.object_detection.base_object_detector import (
    BaseObjectDetector,
)
from foxai.explainer.computer_vision.object_detection.types import ObjectDetectionOutput


class GradCAMObjectDetection:
    """GradCAM for object detection task."""

    def __init__(
        self,
        model: BaseObjectDetector,
        target_layer: torch.nn.Module,
    ):
        self.model = model
        self.gradients: Dict[str, torch.Tensor] = {}
        self.activations: Dict[str, torch.Tensor] = {}
        self.target_layer = target_layer

        def backward_hook(
            module,  # pylint: disable = (unused-argument)
            grad_input,  # pylint: disable = (unused-argument)
            grad_output,
        ):
            self.gradients["value"] = grad_output[0]

        def forward_hook(
            module,  # pylint: disable = (unused-argument)
            input,  # pylint: disable = (unused-argument,redefined-builtin)
            output,
        ):
            self.activations["value"] = output

        self.target_layer.register_forward_hook(forward_hook)
        self.target_layer.register_backward_hook(backward_hook)

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
                    gradients=self.gradients["value"],
                    activations=self.activations["value"],
                )
            )
        return ObjectDetectionOutput(
            saliency_maps=saliency_maps,
            logits=logits,
            predictions=predictions,
        )

    def get_saliency_map(
        self,
        height: int,
        width: int,
        gradients: torch.Tensor,
        activations: torch.Tensor,
    ) -> torch.Tensor:
        """Generate saliency map.

        Args:
            height: Original image height.
            width: Original image width.
            gradients: Layer gradients.
            activations: Layer activations.

        Returns:
            Saliency map.
        """
        b, k, _, _ = gradients.size()
        alpha = gradients.view(b, k, -1).mean(2)
        weights = alpha.view(b, k, 1, 1)
        saliency_map = (weights * activations).sum(1, keepdim=True)
        saliency_map = F.relu(saliency_map)
        saliency_map = F.upsample(
            saliency_map, size=(height, width), mode="bilinear", align_corners=False
        )
        saliency_map_min, saliency_map_max = saliency_map.min(), saliency_map.max()
        saliency_map = (
            (saliency_map - saliency_map_min)
            .div(saliency_map_max - saliency_map_min)
            .data
        )
        return saliency_map

    def __call__(
        self,
        input_img: torch.Tensor,
    ) -> ObjectDetectionOutput:
        return self.forward(input_img)
