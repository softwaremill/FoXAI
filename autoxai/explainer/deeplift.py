"""File with DeepLIFT algorithm explainer classes."""

from abc import abstractmethod
from typing import Optional, Union

import torch
from captum.attr import DeepLift, LayerDeepLift

from autoxai.array_utils import validate_result
from autoxai.explainer.base_explainer import CVExplainer
from autoxai.explainer.model_utils import get_last_conv_model_layer, modify_modules


class BaseDeepLIFTCVExplainer(CVExplainer):
    """Base DeepLIFT algorithm explainer."""

    @abstractmethod
    def create_explainer(
        self,
        model: torch.nn.Module,
        **kwargs,
    ) -> Union[DeepLift, LayerDeepLift]:
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
        """Generate features image with DeepLIFT algorithm explainer.

        Args:
            model: Any DNN model You want to use.
            input_data: Input image.
            pred_label_idx: Predicted label.

        Returns:
            Features matrix.

        Raises:
            RuntimeError: if attribution has shape (0).
        """
        layer: Optional[torch.nn.Module] = kwargs.get("layer", None)

        deeplift = self.create_explainer(model=model, layer=layer)

        attributions = deeplift.attribute(
            input_data,
            target=pred_label_idx,
        )
        validate_result(attributions=attributions)
        return attributions


class DeepLIFTCVExplainer(BaseDeepLIFTCVExplainer):
    """DeepLIFTC algorithm explainer."""

    def create_explainer(
        self,
        model: torch.nn.Module,
        **kwargs,
    ) -> Union[DeepLift, LayerDeepLift]:
        """Create explainer object.

        Returns:
            Explainer object.
        """
        model = modify_modules(model)

        return DeepLift(model=model)


class LayerDeepLIFTCVExplainer(BaseDeepLIFTCVExplainer):
    """Layer DeepLIFT algorithm explainer."""

    def create_explainer(
        self,
        model: torch.nn.Module,
        layer: Optional[torch.nn.Module] = None,
        **kwargs,
    ) -> Union[DeepLift, LayerDeepLift]:
        """Create explainer object.

        Uses parameter `layer` from `kwargs`. If not provided function will call
        `get_last_conv_model_layer` function to obtain last `torch.nn.Conv2d` layer
        from provided model.

        Returns:
            Explainer object.

        Raises:
            ValueError: if model does not contain conv layers.
        """
        if layer is None:
            layer = get_last_conv_model_layer(model=model)

        model = modify_modules(model)

        return LayerDeepLift(model=model, layer=layer)
