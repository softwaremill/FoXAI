"""File with functions to manipulate models."""

from typing import List

import onnx
import onnx2torch
import torch
from torch import fx


def load_model(model_path: str) -> fx.GraphModule:
    """Load model from local path and convert in into torch.fx.GraphModule.

    Args:
        model_path: Path to local ONNX model.

    Returns:
        Converted ONNX model to torch.fx.GraphModule.
    """
    return onnx2torch.convert(onnx.load(model_path))


def get_model_layers(model: fx.GraphModule) -> List[torch.nn.Module]:
    """Get all layers from given model.

    Args:
        model: Model to inspect.

    Returns:
        List of model's layers.
    """
    layers = []
    for module in model.modules():
        if not isinstance(module, fx.graph_module.GraphModule):
            if isinstance(module, torch.nn.Conv2d):
                layers.append(module)

    return layers
