# pylint: disable = missing-class-docstring

import pytest
import torch

from foxai.explainer.model_utils import get_last_conv_model_layer, modify_modules
from tests.sample_model import CNN, AutoEncoder


def test_modify_modules_should_replace_relu() -> None:
    """Test if function modifies ReLU activations."""
    model = AutoEncoder()

    for module in model.modules():
        if isinstance(module, torch.nn.ReLU):
            module.inplace = True

    new_model = modify_modules(model=model)

    for module in new_model.modules():
        if isinstance(module, torch.nn.ReLU):
            assert module.inplace is False


def test_get_last_conv_model_layer_should_raise_exception() -> None:
    """Test if function raises ValueError if no Conv2d layer present in model."""
    model = AutoEncoder()

    with pytest.raises(ValueError):
        _ = get_last_conv_model_layer(model=model)


def test_get_last_conv_model_layer_should_return_last_conv_layer() -> None:
    """Test if function returns last Conv2d layer of the network."""
    model = CNN()

    layer = get_last_conv_model_layer(model=model)
    assert layer == model.conv2
