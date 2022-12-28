"""File with Occulusion algorithm explainer classes."""

import torch
from captum.attr import Occlusion

from autoxai.explainer.base_explainer import CVExplainer


class OcculusionCVExplainer(CVExplainer):
    """Occulusion algorithm explainer."""

    def calculate_features(
        self,
        model: torch.nn.Module,
        input_data: torch.Tensor,
        pred_label_idx: int,
        **kwargs,
    ) -> torch.Tensor:
        """Generate features image with Occulusion algorithm explainer.

        Args:
            model: Any DNN model You want to use.
            input_data: Input image.
            pred_label_idx: Predicted label.

        Returns:
            Features matrix.
        """
        stride_value = kwargs.get("stride_value", 3)
        window_value = kwargs.get("window_value", 3)

        stride = (input_data.shape[1], stride_value, stride_value)
        sliding_window_shapes = (input_data.shape[1], window_value, window_value)
        occlusion = Occlusion(model)

        attributions = occlusion.attribute(
            input_data,
            strides=stride,
            target=pred_label_idx,
            sliding_window_shapes=sliding_window_shapes,
            baselines=0,
        )
        return attributions
