"""File contains functions to modifiy DNN models."""
from typing import List

import torch


def modify_modules(model: torch.nn.Module) -> torch.nn.Module:
    """Modify modules of given model.

    Function iterates over all modules and sets property `inplace`
    to `False` for every `torch.nn.ReLU` activation function.

    Args:
        model: Neural network object to be modified.

    Returns:
        Modified neural network object.
    """
    for module in model.modules():  # pylint: disable = (duplicate-code)
        if isinstance(module, torch.nn.ReLU):
            module.inplace = False

    return model


def get_last_conv_model_layer(model: torch.nn.Module) -> torch.nn.Module:
    """Get the last convolutional layer from the torch model.

    Args:
        model: torch.nn.Module

    Returns:
       The last convolutional layer of the model.

    Raises:
        ValueError if the model does not contain convolutional layers.
    """

    conv_layers: List[torch.nn.Module] = []
    for module in model.modules():
        if isinstance(module, torch.nn.Conv2d):
            conv_layers.append(module)

    if not conv_layers:
        raise ValueError("The model does not contain convolution layers.")

    return conv_layers[-1]
