"""File with Integrated Gradients algorithm explainer classes."""

from abc import abstractmethod
from typing import Optional, Union

import torch
from captum.attr import IntegratedGradients, LayerIntegratedGradients

from autoxai.explainer.base_explainer import CVExplainer
from autoxai.explainer.model_utils import get_last_conv_model_layer


class BaseIntegratedGradientsCVExplainer(CVExplainer):
    """Base Integrated Gradients algorithm explainer."""

    @abstractmethod
    def create_explainer(
        self,
        model: torch.nn.Module,
        **kwargs,
    ) -> Union[IntegratedGradients, LayerIntegratedGradients]:
        """Create explainer object.

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
        self,
        model: torch.nn.Module,
        **kwargs,
    ) -> Union[IntegratedGradients, LayerIntegratedGradients]:
        """Create explainer object.

        Returns:
            Explainer object.
        """

        return IntegratedGradients(forward_func=model)


class LayerIntegratedGradientsCVExplainer(BaseIntegratedGradientsCVExplainer):
    """Layer Integrated Gradients algorithm explainer."""

    def create_explainer(
        self,
        model: torch.nn.Module,
        layer: Optional[torch.nn.Module] = None,
        **kwargs,
    ) -> Union[IntegratedGradients, LayerIntegratedGradients]:
        """Create explainer object.

        Uses parameter `layer` from `kwargs`. If not provided function will call
        `get_last_conv_model_layer` function to obtain last `torch.nn.Conv2d` layer
        from provided model.

        Returns:
            Explainer object.

        Raises:
            ValueError: if model does not contain conv layers
        """
        if layer is None:
            layer = get_last_conv_model_layer(model=model)

        return LayerIntegratedGradients(forward_func=model, layer=layer)
