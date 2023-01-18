"""File with Guided Backpropagation algorithm explainer class."""

import torch
from captum.attr import GuidedBackprop

from autoxai.array_utils import validate_result
from autoxai.explainer.base_explainer import CVExplainer


class GuidedBackpropCVExplainer(CVExplainer):
    """Guided Backpropagation algorithm explainer."""

    def calculate_features(
        self,
        model: torch.nn.Module,
        input_data: torch.Tensor,
        pred_label_idx: int,
        **kwargs,
    ) -> torch.Tensor:
        """Generate features image with Guided Backpropagation algorithm explainer.

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
        guided_backprop = GuidedBackprop(model=model)
        attributions = guided_backprop.attribute(input_data, target=pred_label_idx)
        validate_result(attributions=attributions)
        return attributions
