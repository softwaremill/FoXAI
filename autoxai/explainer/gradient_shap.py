"""File with Gradient SHAP algorithm explainer classes."""

from abc import abstractmethod
from typing import Optional, Union

import torch
from captum.attr import GradientShap, LayerGradientShap

from autoxai.explainer.base_explainer import CVExplainer
from autoxai.explainer.model_utils import get_last_conv_model_layer


class BaseGradientSHAPCVExplainer(CVExplainer):
    """Base Gradient SHAP algorithm explainer."""

    @abstractmethod
    def create_explainer(
        self,
        model: torch.nn.Module,
        **kwargs,
    ) -> Union[GradientShap, LayerGradientShap]:
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
        """Generate features image with Gradient SHAP algorithm explainer.

        Args:
            model: Any DNN model You want to use.
            input_data: Input image.
            pred_label_idx: Predicted label.

        Returns:
            Features matrix.
        """
        stdevs: float = kwargs.get("stdevs", 0.0001)
        n_samples: int = kwargs.get("n_samples", 50)
        layer: Optional[torch.nn.Module] = kwargs.get("layer", None)

        gradient_shap = self.create_explainer(model=model, layer=layer)

        # Defining baseline distribution of images
        rand_img_dist = torch.cat(  # pylint: disable = (no-member,duplicate-code)
            [input_data * 0, input_data * 1]
        )

        attributions = gradient_shap.attribute(
            input_data,
            n_samples=n_samples,
            stdevs=stdevs,
            baselines=rand_img_dist,
            target=pred_label_idx,
        )
        return attributions


class GradientSHAPCVExplainer(BaseGradientSHAPCVExplainer):
    """Gradient SHAP algorithm explainer."""

    def create_explainer(
        self,
        model: torch.nn.Module,
        **kwargs,
    ) -> Union[GradientShap, LayerGradientShap]:
        """Create explainer object.

        Raises:
            RuntimeError: When passed arguments are invalid.

        Returns:
            Explainer object.
        """

        return GradientShap(forward_func=model)


class LayerGradientSHAPCVExplainer(BaseGradientSHAPCVExplainer):
    """Layer Gradient SHAP algorithm explainer."""

    def create_explainer(
        self,
        model: torch.nn.Module,
        layer: Optional[torch.nn.Module] = None,
        **kwargs,
    ) -> Union[GradientShap, LayerGradientShap]:
        """Create explainer object.

        Returns:
            Explainer object.

        Raises:
            ValueError: if model does not contain conv layers
        """

        if layer is None:
            layer = get_last_conv_model_layer(model=model)

        return LayerGradientShap(forward_func=model, layer=layer)
