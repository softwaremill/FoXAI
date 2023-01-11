"""File with DeepLIFT SHAP algorithm explainer classes."""

from abc import abstractmethod
from typing import Optional, Union

import torch
from captum.attr import DeepLiftShap, LayerDeepLiftShap

from autoxai.explainer.base_explainer import CVExplainer
from autoxai.explainer.model_utils import get_last_conv_model_layer, modify_modules


class BaseDeepLIFTSHAPCVExplainer(CVExplainer):
    """Base DeepLIFT SHAP algorithm explainer."""

    @abstractmethod
    def create_explainer(
        self,
        model: torch.nn.Module,
        **kwargs,
    ) -> Union[DeepLiftShap, LayerDeepLiftShap]:
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
        """Generate features image with DeepLIFT SHAP algorithm explainer.

        Args:
            model: Any DNN model You want to use.
            input_data: Input image.
            pred_label_idx: Predicted label.

        Returns:
            Features matrix.
        """
        layer: Optional[torch.nn.Module] = kwargs.get("layer", None)
        number_of_samples: int = kwargs.get("number_of_samples", 100)

        deeplift = self.create_explainer(model=model, layer=layer)
        baselines = torch.randn(  # pylint: disable = (no-member)
            number_of_samples,
            input_data.shape[1],
            input_data.shape[2],
            input_data.shape[3],
            requires_grad=True,
        ).to(device=input_data.device)
        attributions = deeplift.attribute(
            input_data,
            target=pred_label_idx,
            baselines=baselines,
        )
        if attributions.shape[0] == 0:
            raise RuntimeError(
                "Error occured during attribution calculation. "
                + "Make sure You are applying this method to CNN network.",
            )
        return attributions


class DeepLIFTSHAPCVExplainer(BaseDeepLIFTSHAPCVExplainer):
    """DeepLIFTC SHAP algorithm explainer."""

    def create_explainer(
        self,
        model: torch.nn.Module,
        **kwargs,
    ) -> Union[DeepLiftShap, LayerDeepLiftShap]:
        """Create explainer object.

        Raises:
            RuntimeError: When passed arguments are invalid.

        Returns:
            Explainer object.
        """
        model = modify_modules(model)

        return DeepLiftShap(model=model)


class LayerDeepLIFTSHAPCVExplainer(BaseDeepLIFTSHAPCVExplainer):
    """Layer DeepLIFT SHAP algorithm explainer."""

    def create_explainer(
        self,
        model: torch.nn.Module,
        layer: Optional[torch.nn.Module] = None,
        **kwargs,
    ) -> Union[DeepLiftShap, LayerDeepLiftShap]:
        """Create explainer object.

        Returns:
            Explainer object.

        Raises:
            ValueError: if model does not contain conv layers
        """
        if layer is None:
            layer = get_last_conv_model_layer(model=model)

        model = modify_modules(model)

        return LayerDeepLiftShap(model=model, layer=layer)
