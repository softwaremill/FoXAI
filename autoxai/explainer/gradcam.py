"""File with GradCAM algorithm explainer classes."""

from abc import abstractmethod
from typing import Optional, Union

import torch
from captum.attr import GuidedGradCam, LayerGradCam

from autoxai.explainer.base_explainer import CVExplainer
from autoxai.explainer.errors import LAYER_ARGUMENT_MISSING
from autoxai.explainer.model_utils import modify_modules


class BaseGradCAMCVExplainer(CVExplainer):
    """Base GradCAM algorithm explainer."""

    @abstractmethod
    def create_explainer(
        self, model: torch.nn.Module, layer: torch.nn.Module, **kwargs
    ) -> Union[GuidedGradCam, LayerGradCam]:
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
        """Generate features image with GradCAM algorithm explainer.

        Args:
            model: Any DNN model You want to use.
            input_data: Input image.
            pred_label_idx: Predicted label.

        Returns:
            Features matrix.

        Raises:
            ValueError: if layer is None
            RuntimeError: if attributions has shape (0)
        """
        layer: Optional[torch.nn.Module] = kwargs.get("layer", None)
        if layer is None:
            raise ValueError("GradCAMCVExplainer" + LAYER_ARGUMENT_MISSING)

        guided_cam = self.create_explainer(model=model, layer=layer)

        attributions = guided_cam.attribute(
            input_data,
            target=pred_label_idx,
        )
        super().validate_result(attributions=attributions)
        return attributions


class GuidedGradCAMCVExplainer(BaseGradCAMCVExplainer):
    """GuidedGradCAM algorithm explainer."""

    def create_explainer(
        self,
        model: torch.nn.Module,
        layer: torch.nn.Module,
        **kwargs,
    ) -> Union[GuidedGradCam, LayerGradCam]:
        """Create explainer object.

        Returns:
            Explainer object.
        """
        model = modify_modules(model)

        return GuidedGradCam(model=model, layer=layer)


class LayerGradCAMCVExplainer(BaseGradCAMCVExplainer):
    """Layer GradCAM algorithm explainer."""

    def create_explainer(
        self,
        model: torch.nn.Module,
        layer: torch.nn.Module,
        **kwargs,
    ) -> Union[GuidedGradCam, LayerGradCam]:
        """Create explainer object.

        Returns:
            Explainer object.
        """
        model = modify_modules(model)

        return LayerGradCam(forward_func=model, layer=layer)
