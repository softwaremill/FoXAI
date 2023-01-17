"""File with Deconvolution algorithm explainer classes."""

from abc import abstractmethod

import torch
from captum.attr import Deconvolution

from autoxai.explainer.base_explainer import CVExplainer
from autoxai.explainer.model_utils import modify_modules


class BaseDeconvolutionCVExplainer(CVExplainer):
    """Base Deconvolution algorithm explainer."""

    @abstractmethod
    def create_explainer(
        self,
        model: torch.nn.Module,
    ) -> Deconvolution:
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
        """Generate features image with Deconvolution algorithm explainer.

        Args:
            model: Any DNN model You want to use.
            input_data: Input image.
            pred_label_idx: Predicted label.

        Returns:
            Features matrix.
        """

        deconv = self.create_explainer(model=model)
        attributions = deconv.attribute(
            input_data,
            target=pred_label_idx,
        )
        super().validate_result(attributions=attributions)
        return attributions


class DeconvolutionCVExplainer(BaseDeconvolutionCVExplainer):
    """Base Deconvolution algorithm explainer."""

    def create_explainer(
        self,
        model: torch.nn.Module,
    ) -> Deconvolution:
        """Create explainer object.

        Returns:
            Explainer object.
        """
        model = modify_modules(model=model)
        deconv = Deconvolution(model=model)

        return deconv
