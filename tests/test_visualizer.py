# pylint: disable = missing-class-docstring
import matplotlib
import pytest
import torch

from autoxai.explainer.base_explainer import (
    CVExplainer,
    determine_visualization_methods,
)


class TestVisualizer:
    """Test if wrapper on captum visualize
    function works fine, for different attributes.
    """

    @pytest.fixture
    def ones_image(self) -> torch.Tensor:
        return torch.ones(size=(1, 1, 50, 50), dtype=torch.float32)

    @pytest.fixture
    def minus_ones_image(self) -> torch.Tensor:
        return -torch.ones(size=(1, 1, 50, 50), dtype=torch.float32)

    @pytest.fixture
    def zeros_image(self) -> torch.Tensor:
        return torch.zeros(size=(1, 1, 50, 50), dtype=torch.float32)

    @pytest.fixture
    def minus_and_plus_once_image(self) -> torch.Tensor:
        image: torch.tensor = torch.ones(size=(1, 1, 50, 50), dtype=torch.float32)
        image[:, :, 20:30, 20:30] = -1.0
        return image

    def test_no_negative_values(self, ones_image: torch.Tensor):
        _: matplotlib.pyplot.Figure = CVExplainer.visualize(
            attributions=ones_image,
            transformed_img=ones_image,
        )

        visualization_methods = determine_visualization_methods(
            attributions_np=ones_image.numpy()
        )
        assert len(visualization_methods) == 3

    def test_no_positive_values(self, minus_ones_image: torch.Tensor):
        _: matplotlib.pyplot.Figure = CVExplainer.visualize(
            attributions=minus_ones_image,
            transformed_img=minus_ones_image,
        )

        visualization_methods = determine_visualization_methods(
            attributions_np=minus_ones_image.numpy()
        )
        assert len(visualization_methods) == 3

    def test_zeros_image(self, zeros_image: torch.Tensor):
        _: matplotlib.pyplot.Figure = CVExplainer.visualize(
            attributions=zeros_image,
            transformed_img=zeros_image,
        )

        visualization_methods = determine_visualization_methods(
            attributions_np=zeros_image.numpy()
        )
        assert len(visualization_methods) == 1

    def test_iamge_with_positive_and_negative_attributes(
        self, minus_and_plus_once_image: torch.tensor
    ):
        _: matplotlib.pyplot.Figure = CVExplainer.visualize(
            attributions=minus_and_plus_once_image,
            transformed_img=minus_and_plus_once_image,
        )

        visualization_methods = determine_visualization_methods(
            attributions_np=minus_and_plus_once_image.numpy()
        )
        assert len(visualization_methods) == 4
