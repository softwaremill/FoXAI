"""File with Integrated Gradients algorithm explainer classes."""

from abc import abstractmethod
from typing import Optional, Union

import torch
from captum.attr import IntegratedGradients, LayerIntegratedGradients

from src.explainer.base_explainer import CVExplainer


class BaseIntegratedGradientsCVExplainer(CVExplainer):
    """Base Integrated Gradients algorithm explainer."""

    @abstractmethod
    def create_explainer(
        self, **kwargs
    ) -> Union[IntegratedGradients, LayerIntegratedGradients]:
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
        """Generate features image with Integrated Gradients algorithm explainer.

        Args:
            model: Any DNN model You want to use.
            layer: Layer from DNN to explain.
            input_data: Input image.
            pred_label_idx: Predicted label.

        Returns:
            Features matrix.
        """
        n_steps = kwargs.get("n_steps", 100)
        layer: Optional[torch.nn.Module] = kwargs.get("layer", None)

        integrated_gradients = self.create_explainer(model=model, layer=layer)
        attributions = integrated_gradients.attribute(
            input_data, target=pred_label_idx, n_steps=n_steps
        )
        return attributions


class IntegratedGradientsCVExplainer(BaseIntegratedGradientsCVExplainer):
    """Integrated Gradients algorithm explainer."""

    def create_explainer(
        self, **kwargs
    ) -> Union[IntegratedGradients, LayerIntegratedGradients]:
        """Create explainer object.

        Raises:
            RuntimeError: When passed arguments are invalid.

        Returns:
            Explainer object.
        """
        model: Optional[torch.nn.Module] = kwargs.get("model", None)
        if model is None:
            raise RuntimeError(f"Missing or `None` argument `model` passed: {kwargs}")

        return IntegratedGradients(forward_func=model)


class LayerIntegratedGradientsCVExplainer(BaseIntegratedGradientsCVExplainer):
    """Layer Integrated Gradients algorithm explainer."""

    def create_explainer(
        self, **kwargs
    ) -> Union[IntegratedGradients, LayerIntegratedGradients]:
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

        return LayerIntegratedGradients(forward_func=model, layer=layer)
