"""File with GradCAM algorithm explainer classes."""

from abc import abstractmethod
from typing import Optional, Union

import torch
from captum.attr import GuidedGradCam, LayerGradCam

from src.explainer.base_explainer import CVExplainer
from src.explainer.model_utils import modify_modules


class BaseGradCAMCVExplainer(CVExplainer):
    """Base GradCAM algorithm explainer."""

    @abstractmethod
    def create_explainer(self, **kwargs) -> Union[GuidedGradCam, LayerGradCam]:
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
        """Generate features image with GradCAM algorithm explainer.

        Args:
            model: Any DNN model You want to use.
            input_data: Input image.
            pred_label_idx: Predicted label.

        Returns:
            Features matrix.
        """
        layer: Optional[torch.nn.Module] = kwargs.get("layer", None)

        guided_cam = self.create_explainer(model=model, layer=layer)

        attributions = guided_cam.attribute(
            input_data,
            target=pred_label_idx,
        )
        if attributions.shape[0] == 0:
            raise RuntimeError(
                "Error occured during attribution calculation. "
                + "Make sure You are applying this method to CNN network.",
            )
        return attributions


class GuidedGradCAMCVExplainer(BaseGradCAMCVExplainer):
    """GuidedGradCAM algorithm explainer."""

    def create_explainer(self, **kwargs) -> Union[GuidedGradCam, LayerGradCam]:
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

        model = modify_modules(model)

        return GuidedGradCam(model=model, layer=layer)


class LayerGradCAMCVExplainer(BaseGradCAMCVExplainer):
    """Layer GradCAM algorithm explainer."""

    def create_explainer(self, **kwargs) -> Union[GuidedGradCam, LayerGradCam]:
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

        model = modify_modules(model)

        return LayerGradCam(forward_func=model, layer=layer)
