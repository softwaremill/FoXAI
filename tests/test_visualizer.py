# pylint: disable = missing-class-docstring
import matplotlib
import pytest
import torch

from autoxai.explainer.base_explainer import CVExplainer


class TestVisualizer:
    @pytest.fixture
    def ones_image(self) -> torch.Tensor:
        return torch.ones(size=(1, 1, 50, 50), dtype=torch.float32)

    @pytest.fixture
    def minus_ones_image(self) -> torch.Tensor:
        return -torch.ones(size=(1, 1, 50, 50), dtype=torch.float32)

    @pytest.fixture
    def zero_image(self) -> torch.Tensor:
        return torch.zeros(size=(1, 1, 50, 50), dtype=torch.float32)

    def test_no_negative_values(self, ones_image: torch.Tensor):
        _: matplotlib.pyplot.Figure = CVExplainer.visualize(
            attributions=ones_image,
            transformed_img=ones_image,
        )

    def test_no_positive_values(self, minus_ones_image: torch.Tensor):
        _: matplotlib.pyplot.Figure = CVExplainer.visualize(
            attributions=minus_ones_image,
            transformed_img=minus_ones_image,
        )

    def test_zero_image(self, zero_image: torch.Tensor):
        _: matplotlib.pyplot.Figure = CVExplainer.visualize(
            attributions=zero_image,
            transformed_img=zero_image,
        )
