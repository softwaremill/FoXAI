"""Model loading and prediction functions."""
from typing import Any, Dict

import torch
import torch.nn.functional as F
from torchvision import models


def load_model() -> Any:
    """Load model to explain.

    Returns:
        Any model type.
    """
    model = models.resnet18(pretrained=True)
    model = model.eval()

    return model


def load_model_idx_to_label() -> Dict[int, str]:
    """Load index to label mapping for model.

    Returns:
        Dictionary containgin index to label mapping.
    """
    weights = models.ResNet18_Weights.DEFAULT
    return {  # pylint: disable = (unnecessary-comprehension)
        i: val for i, val in enumerate(weights.meta["categories"])
    }


def get_prediction(model: Any, input_data: torch.Tensor) -> torch.Tensor:
    """Get predicted label from model.

    Args:
        model: Any model type.
        input_data: Input data tensor.

    Returns:
        Tensor with predicted label.
    """
    output = model(input_data)
    output = F.softmax(output, dim=1)
    _, pred_label_idx = torch.topk(output, 1)  # pylint: disable = (no-member)

    pred_label_idx.squeeze_()
    return pred_label_idx
