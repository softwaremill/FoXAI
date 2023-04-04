from dataclasses import dataclass
from typing import List


@dataclass
class PredictionOutput:
    """Data class for model prediction output in YOLO style."""

    bbox: List[int]
    class_number: int
    class_name: str
    confidence: float
