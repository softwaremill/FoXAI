"""
Code based on https://github.com/pooya-mohammadi/yolov5-gradcam.
"""

from dataclasses import dataclass
from typing import Dict, List, Tuple

import torch
import torch.nn.functional as F

from foxai.explainer.computer_vision.object_detection.object_detector import (
    BaseObjectDetector,
    PredictionOutput,
)


@dataclass
class ObjectDetectionOutput:
    """Data class for model predictions for object detection.

    It contains heatmaps, logits and predictions in YOLO style.
    """

    saliency_maps: List[torch.Tensor]
    logits: List[torch.Tensor]
    predictions: List[PredictionOutput]


class GradCAMObjectDetection:
    """GradCAM for object detection task."""

    def __init__(
        self,
        model: BaseObjectDetector,
        target_layer: torch.nn.Module,
        img_size: Tuple[int, int] = (640, 640),
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

        device = (
            "cuda"
            if isinstance(self.model.model, torch.nn.Module)
            and next(self.model.model.parameters()).is_cuda
            else "cpu"
        )
        self.model(torch.zeros(1, 3, *img_size, device=device))

    def forward(
        self,
        input_img: torch.Tensor,
        class_idx: bool = True,
    ) -> ObjectDetectionOutput:
        """Forward pass of GradCAM aglorithm.

        Args:
            input_img: Input image with shape of (B, C, H, W).

        Returns:
            ObjectDetectionOutput object.
        """
        saliency_maps: List[torch.Tensor] = []
        b, _, h, w = input_img.size()
        predictions, logits = self.model.forward(input_img)
        for logit, cls in zip(logits[0], [p.class_number for p in predictions]):
            if class_idx:
                score = logit[cls]
            else:
                score = logit.max()
            self.model.zero_grad()
            score.backward(retain_graph=True)
            gradients = self.gradients["value"]
            activations = self.activations["value"]
            b, k, _, _ = gradients.size()
            alpha = gradients.view(b, k, -1).mean(2)
            weights = alpha.view(b, k, 1, 1)
            saliency_map = (weights * activations).sum(1, keepdim=True)
            saliency_map = F.relu(saliency_map)
            saliency_map = F.upsample(
                saliency_map, size=(h, w), mode="bilinear", align_corners=False
            )
            saliency_map_min, saliency_map_max = saliency_map.min(), saliency_map.max()
            saliency_map = (
                (saliency_map - saliency_map_min)
                .div(saliency_map_max - saliency_map_min)
                .data
            )
            saliency_maps.append(saliency_map)
        return ObjectDetectionOutput(
            saliency_maps=saliency_maps,
            logits=logits,
            predictions=predictions,
        )

    def __call__(
        self,
        input_img: torch.Tensor,
    ) -> ObjectDetectionOutput:
        return self.forward(input_img)
