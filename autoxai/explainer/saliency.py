"""File with Saliency algorithm explainer classes."""

import torch
from captum.attr import Saliency

from autoxai.array_utils import validate_result
from autoxai.explainer.base_explainer import CVExplainer


class SaliencyCVExplainer(CVExplainer):
    """Saliency algorithm explainer."""

    def calculate_features(
        self,
        model: torch.nn.Module,
        input_data: torch.Tensor,
        pred_label_idx: int,
        **kwargs,  # pylint: disable = (unused-argument)
    ) -> torch.Tensor:
        """Generate features image with Saliency algorithm explainer.

        Args:
            model: Any DNN model You want to use.
            input_data: Input image.
            pred_label_idx: Predicted label.

        Returns:
            Features matrix.

        Raises:
            RuntimeError: if attribution has shape (0).
        """
        saliency = Saliency(forward_func=model)

        attributions = saliency.attribute(
            input_data,
            target=pred_label_idx,
        )
        validate_result(attributions=attributions)
        return attributions
