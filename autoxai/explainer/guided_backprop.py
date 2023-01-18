"""File with Guided Backpropagation algorithm explainer class."""

from abc import abstractmethod

import torch
from captum.attr import GuidedBackprop

from autoxai.array_utils import validate_result
from autoxai.explainer.base_explainer import CVExplainer
from autoxai.explainer.model_utils import modify_modules


class BaseGuidedBackpropCVExplainer(CVExplainer):
    """Base Guided Backpropagation algorithm explainer."""

    @abstractmethod
    def create_explainer(
        self,
        model: torch.nn.Module,
        **kwargs,
    ) -> GuidedBackprop:
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

        Raises:
            RuntimeError: if attribution has shape (0).
        """
        guided_backprop = self.create_explainer(model=model)
        attributions = guided_backprop.attribute(input_data, target=pred_label_idx)
        validate_result(attributions=attributions)
        return attributions


class GuidedBackpropCVExplainer(BaseGuidedBackpropCVExplainer):
    """Guided Backpropagation algorithm explainer."""

    def create_explainer(
        self,
        model: torch.nn.Module,
        **kwargs,
    ) -> GuidedBackprop:
        """Create explainer object.

        Returns:
            Explainer object.
        """

        return GuidedBackprop(model=modify_modules(model))
