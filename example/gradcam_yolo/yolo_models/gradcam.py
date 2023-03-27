"""
Code based on https://github.com/pooya-mohammadi/yolov5-gradcam.
"""

from typing import List, Tuple

import torch
import torch.nn.functional as F

from example.gradcam_yolo.yolo_models.object_detector import BaseObjectDetector


def find_yolo_layer(model: BaseObjectDetector, layer_name: str) -> torch.nn.Module:
    """Find yolov5 layer to calculate GradCAM and GradCAM++

    Args:
        model: yolov5 model.
        layer_name (str): the name of layer with its hierarchical information.

    Return:
        target_layer: found layer
    """
    hierarchy = layer_name.split("_")
    target_layer = model.model.model._modules[  # pylint: disable = (protected-access)
        hierarchy[0]
    ]

    for h in hierarchy[1:]:
        target_layer = target_layer._modules[h]  # pylint: disable = (protected-access)
    return target_layer


class GradCAMObjectDetection:
    """GradCAM for object detection task."""

    def __init__(
        self,
        model: BaseObjectDetector,
        layer_name: str,
        img_size: Tuple[int, int] = (640, 640),
    ):
        self.model = model
        self.gradients = {}
        self.activations = {}

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

        target_layer = find_yolo_layer(self.model, layer_name)
        target_layer.register_forward_hook(forward_hook)
        target_layer.register_backward_hook(backward_hook)

        device = "cuda" if next(self.model.model.parameters()).is_cuda else "cpu"
        self.model(torch.zeros(1, 3, *img_size, device=device))

    def forward(
        self,
        input_img: torch.Tensor,
        class_idx: bool = True,
    ) -> Tuple[List[torch.Tensor], List[torch.Tensor], torch.Tensor]:
        """
        Args:
            input_img: input image with shape of (1, 3, H, W)
        Return:
            mask: saliency map of the same spatial dimension with input
            logit: model output
            preds: The object predictions
        """
        saliency_maps = []
        b, _, h, w = input_img.size()
        preds, logits = self.model(input_img)
        for logit, cls, _ in zip(logits[0], preds[1][0], preds[2][0]):
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
        return saliency_maps, logits, preds

    def __call__(
        self,
        input_img: torch.Tensor,
    ) -> Tuple[List[torch.Tensor], List[torch.Tensor], torch.Tensor]:
        return self.forward(input_img)
