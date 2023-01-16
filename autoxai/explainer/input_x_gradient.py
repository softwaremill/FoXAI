"""File with Input X Gradient algorithm explainer classes."""

from abc import abstractmethod
from typing import Optional, Union

import torch
from captum.attr import InputXGradient, LayerGradientXActivation

from autoxai.explainer.base_explainer import CVExplainer
from autoxai.explainer.model_utils import get_last_conv_model_layer


class BaseInputXGradientSHAPCVExplainer(CVExplainer):
    """Base Input X Gradient algorithm explainer."""

    @abstractmethod
    def create_explainer(
        self,
        model: torch.nn.Module,
        **kwargs,
    ) -> Union[InputXGradient, LayerGradientXActivation]:
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
        if attributions.shape[0] == 0:
            raise RuntimeError(
                "Error occured during attribution calculation. "
                + "Make sure You are applying this method to CNN network.",
            )
        return attributions


class InputXGradientCVExplainer(BaseInputXGradientSHAPCVExplainer):
    """Input X Gradient algorithm explainer."""

    def create_explainer(
        self,
        model: torch.nn.Module,
        **kwargs,
    ) -> Union[InputXGradient, LayerGradientXActivation]:
        """Create explainer object.

        Returns:
            Explainer object.
        """

        return InputXGradient(forward_func=model)


class LayerInputXGradientCVExplainer(BaseInputXGradientSHAPCVExplainer):
    """Layer Input X Gradient algorithm explainer."""

    def create_explainer(
        self,
        model: torch.nn.Module,
        layer: Optional[torch.nn.Module] = None,
        **kwargs,
    ) -> Union[InputXGradient, LayerGradientXActivation]:
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

        return LayerGradientXActivation(forward_func=model, layer=layer)
