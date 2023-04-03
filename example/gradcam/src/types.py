from dataclasses import dataclass
from typing import List


@dataclass
class PredictionOutput:
    """Data class for model prediction output in YOLO style."""

    bbox: List[List[int]]
    class_number: List[List[int]]
    class_name: List[List[str]]
    confidence: List[List[float]]
