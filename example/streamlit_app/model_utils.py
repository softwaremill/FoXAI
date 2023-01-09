"""File with functions to manipulate models."""

from typing import List

import torch
from mnist_model import LitMNIST  # pylint: disable = (import-error)


def load_model(model_path: str) -> torch.nn.Module:
    """Load model's state dict from local path.

    Args:
        model_path: Path to local model's state dict.

    Returns:
        Model with loaded state dict.
    """
    model = LitMNIST(batch_size=1, data_dir=".")
    model.load_state_dict(torch.load(model_path))
    model.eval()
    return model


def get_model_layers(model: torch.nn.Module) -> List[torch.nn.Module]:
    """Get all layers from given model.

    Args:
        model: Model to inspect.

    Returns:
        List of model's layers.
    """
    layers = []
    for module in model.modules():
        if isinstance(module, torch.nn.Conv2d):
            layers.append(module)

    return layers
