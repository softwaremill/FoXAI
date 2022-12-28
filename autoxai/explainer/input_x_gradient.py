"""File with Input X Gradient algorithm explainer classes."""

from abc import abstractmethod
from typing import Optional, Union

import torch
from captum.attr import InputXGradient, LayerGradientXActivation

from autoxai.explainer.base_explainer import CVExplainer


class BaseInputXGradientSHAPCVExplainer(CVExplainer):
    """Base Input X Gradient algorithm explainer."""

    @abstractmethod
    def create_explainer(
        self, **kwargs
    ) -> Union[InputXGradient, LayerGradientXActivation]:
        """Create explainer object.

        Raises:
            RuntimeError: When passed arguments are invalid.

        Returns:
            Explainer object.
        """

    def calculate_features(
        self,
        model: torch.nn.Module,
        input_data: torch.Tensor,
        pred_label_idx: int,
        **kwargs,
    ) -> torch.Tensor:
        """Generate features image with Input X Gradient algorithm explainer.

        Args:
            model: Any DNN model You want to use.
            input_data: Input image.
            pred_label_idx: Predicted label.

        Returns:
            Features matrix.
        """
        layer: Optional[torch.nn.Module] = kwargs.get("layer", None)

        input_x_gradient = self.create_explainer(model=model, layer=layer)

        attributions = input_x_gradient.attribute(
            input_data,
            target=pred_label_idx,
        )
        return attributions


class InputXGradientCVExplainer(BaseInputXGradientSHAPCVExplainer):
    """Input X Gradient algorithm explainer."""

    def create_explainer(
        self, **kwargs
    ) -> Union[InputXGradient, LayerGradientXActivation]:
        """Create explainer object.

        Raises:
            RuntimeError: When passed arguments are invalid.

        Returns:
            Explainer object.
        """
        model: Optional[torch.nn.Module] = kwargs.get("model", None)
        if model is None:
            raise RuntimeError(f"Missing or `None` argument `model` passed: {kwargs}")

        return InputXGradient(forward_func=model)


class LayerInputXGradientCVExplainer(BaseInputXGradientSHAPCVExplainer):
    """Layer Input X Gradient algorithm explainer."""

    def create_explainer(
        self, **kwargs
    ) -> Union[InputXGradient, LayerGradientXActivation]:
        """Create explainer object.

        Raises:
            RuntimeError: When passed arguments are invalid.

        Returns:
            Explainer object.
        """
        model: Optional[torch.nn.Module] = kwargs.get("model", None)
        layer: Optional[torch.nn.Module] = kwargs.get("layer", None)
        if model is None or layer is None:
            raise RuntimeError(
                f"Missing or `None` arguments `model` or `layer` passed: {kwargs}"
            )

        return LayerGradientXActivation(forward_func=model, layer=layer)
