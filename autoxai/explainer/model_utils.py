"""File contains functions to modifiy DNN models."""
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
