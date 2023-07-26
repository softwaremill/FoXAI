"""File contains functions to modifiy DNN models."""
from typing import List, Tuple, Union

import torch
from captum._utils.typing import BaselineType

from foxai.types import LayerType, ModelType


def modify_modules(model: ModelType) -> ModelType:
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


def get_last_conv_model_layer(model: ModelType) -> LayerType:
    """Get the last convolutional layer from the torch model.

    Args:
        model: torch.nn.Module

    Returns:
       The last convolutional layer of the model.

    Raises:
        ValueError if the model does not contain convolutional layers.
    """

    conv_layers: List[LayerType] = []
    for module in model.modules():
        if isinstance(module, torch.nn.Conv2d):
            conv_layers.append(module)

    if not conv_layers:
        raise ValueError("The model does not contain convolution layers.")

    return conv_layers[-1]


def preprocess_baselines(
    baselines: BaselineType,
    input_data_shape: torch.Size,
) -> Tuple[List[BaselineType], bool]:
    """Adjust provided baselines to match input data dimension.

    When baselines should have the same dimension as data samples or should have
    additional first dimension to compute averaged attributes across all baselines.

    Args:
        baselines: Single tensor matching data sample dimensions or tensor
            containing batch of baselines for each data sample.
        input_data_shape: Shape of data sample batch.

    Returns:
        Tuple of baselines list and flag indicating if attributes should be
            aggregated.
    """
    aggregate_attributes: bool = False
    baselines_list: List[Union[None, torch.Tensor]] = [baselines]

    if isinstance(baselines, torch.Tensor):
        # if dimension of baselines is greater than batch data user have provided
        # multiple baselines to aggregate results
        if len(baselines.shape) == len(input_data_shape) + 1:
            aggregate_attributes = True
            baselines_list = list(baselines)

    return baselines_list, aggregate_attributes
