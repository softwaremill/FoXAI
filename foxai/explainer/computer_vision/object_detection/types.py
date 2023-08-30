"""File contains data class for object detection unified model output."""
from dataclasses import dataclass
from typing import List, Tuple

import torch

from foxai.types import BboxType

DetectionOutput = Tuple[torch.Tensor, torch.Tensor]


@dataclass
class PredictionOutput:
    """Data class for model prediction output in YOLO style."""

    bbox: BboxType
    class_number: int
    class_name: str
    confidence: float


@dataclass
class ObjectDetectionOutput:
    """Data class for model predictions for object detection.

    It contains heatmaps, logits and predictions in YOLO style.
    """

    saliency_maps: List[torch.Tensor]
    logits: List[torch.Tensor]
    predictions: List[PredictionOutput]
