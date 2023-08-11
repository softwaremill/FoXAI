# pylint: disable = missing-class-docstring
import logging
from typing import Optional

import pytest
import torch

from foxai.explainer.fullgradient import FullGrad
from foxai.logger import create_logger

_LOGGER: Optional[logging.Logger] = None


def log() -> logging.Logger:
    """Get or create logger."""
    # pylint: disable = global-statement
    global _LOGGER
    if _LOGGER is None:
        _LOGGER = create_logger(__name__)
    return _LOGGER


class BatchSampleModel(torch.nn.Module):
    """Sample pytorch model for experiment."""

    def __init__(
        self,
    ):
        super().__init__()
        self.norm = torch.nn.BatchNorm2d(num_features=1)
        self.norm.running_mean = torch.tensor([0.0], dtype=torch.float32)
        self.norm.running_var = torch.tensor([1.0], dtype=torch.float32)

    def forward(self, x_tensor: torch.Tensor) -> torch.Tensor:
        """Forward methid for the module."""
        return torch.mean(torch.mean(self.norm(x_tensor), dim=-1), dim=0)


class TestFullGradients:
    """Test all explainers, if they run witout error."""

    @pytest.fixture
    def model(self) -> BatchSampleModel:
        """Sample model to run FullGradients explainer on."""
        return BatchSampleModel()

    def test_eval_mode_switch(
        self,
        model: BatchSampleModel,
    ):
        """Test whether the switch to eval for the time of function call and
        back to inital mode after the function call works correctly.

        Args:
            model: model to run tests on.
        """
        zero_tensor = torch.zeros(size=(1, 1, 20, 20), dtype=torch.float32)
        ones_tensor = torch.ones(size=(1, 1, 20, 20), dtype=torch.float32)
        input_tensor = torch.cat((zero_tensor, ones_tensor), dim=0)
        model.eval()
        out1 = model(input_tensor)

        fullgradients = FullGrad(model=model, image_size=input_tensor.shape[1:])
        model.train()
        fullgradients.check_completeness()
        assert (
            model.training
        ), "The FullGrad class should restore model state after complete."

        model.eval()
        out2 = model(input_tensor)

        fullgradients.check_completeness()
        assert (
            not model.training
        ), "The FullGrad class should restore model state after complete."

        out3 = model(input_tensor)

        model.train()
        out4 = model(input_tensor)

        correct_output = torch.full(size=(20, 1), fill_value=0.5, dtype=torch.float32)
        assert torch.allclose(
            out1, correct_output
        ), "The output should be equalt to 'correct_output'."
        assert torch.allclose(
            out1, out2
        ), "The model should not change the value of running mean and running variance in batch layer."
        assert torch.allclose(
            out1, out3
        ), "The model should not change the value of running mean and running in batch layer."
        assert not torch.allclose(
            out1, out4
        ), "The model should change the value of running mean and running in batch layer."
