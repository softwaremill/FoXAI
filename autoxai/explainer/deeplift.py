"""File with DeepLIFT algorithm explainer classes."""

from abc import abstractmethod
from typing import Optional, Union

import torch
from captum.attr import DeepLift, LayerDeepLift

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
        """Generate features image with DeepLIFT algorithm explainer.

        Args:
            model: Any DNN model You want to use.
            input_data: Input image.
            pred_label_idx: Predicted label.

        Returns:
            Features matrix.
        """
        layer: Optional[torch.nn.Module] = kwargs.get("layer", None)

        deeplift = self.create_explainer(model=model, layer=layer)

        attributions = deeplift.attribute(
            input_data,
            target=pred_label_idx,
        )
        if attributions.shape[0] == 0:
            raise RuntimeError(
                "Error occured during attribution calculation. "
                + "Make sure You are applying this method to CNN network.",
            )
        return attributions


class DeepLIFTCVExplainer(BaseDeepLIFTCVExplainer):
    """DeepLIFTC algorithm explainer."""

    def create_explainer(
        self,
        model: torch.nn.Module,
        **kwargs,
    ) -> Union[DeepLift, LayerDeepLift]:
        """Create explainer object.

        Raises:
            RuntimeError: When passed arguments are invalid.

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

        Raises:
            RuntimeError: When passed arguments are invalid.

        Returns:
            Explainer object.
        """
        if layer is None:
            layer = get_last_conv_model_layer(model=model)

        model = modify_modules(model)

        return LayerDeepLift(model=model, layer=layer)
