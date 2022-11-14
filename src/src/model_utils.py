from typing import Any
import torch
from torchvision import models
import torch.nn.functional as F


def load_model() -> Any:
    """Load model to explain.

    Returns:
        Any: Any model type.
    """
    model = models.resnet18(pretrained=True)
    model = model.eval()
    return model


def get_prediction(model: Any, input_data: torch.Tensor) -> torch.Tensor:
    """Get predicted label from model.

    Args:
        model (Any): Any model type.
        input_data (torch.Tensor): Input data tensor.

    Returns:
        torch.Tensor: Tensor with predicted label.
    """
    output = model(input_data)
    output = F.softmax(output, dim=1)
    _, pred_label_idx = torch.topk(output, 1)

    pred_label_idx.squeeze_()
    return pred_label_idx
