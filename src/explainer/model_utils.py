"""File contains functions to modifiy DNN models."""

import torch
from torch import fx


def modify_modules(model: fx.GraphModule) -> fx.GraphModule:
    """Modify modules of given model.

    Function iterates over all modules and sets property `inplace`
    to `False` for every `torch.nn.ReLU` activation function.

    Args:
        model: DNN object to be modified.

    Returns:
        Modified DNN object.
    """
    for module in model.modules():  # pylint: disable = (duplicate-code)
        if isinstance(module, torch.nn.ReLU):
            module.inplace = False

    return model
