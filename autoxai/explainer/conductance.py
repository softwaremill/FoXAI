"""File with Conductance algorithm explainer classes."""

from typing import Optional

import torch
from captum.attr import LayerConductance

from autoxai.explainer.base_explainer import CVExplainer
from autoxai.explainer.errors import LAYER_ARGUMENT_MISSING


class LayerConductanceCVExplainer(CVExplainer):
    """Layer Conductance algorithm explainer."""

    # pylint: disable = unused-argument
    def create_explainer(
        self,
        model: torch.nn.Module,
        layer: torch.nn.Module,
    ) -> LayerConductance:
        """Create explainer object.

        Returns:
            Explainer object.
        """
        conductance = LayerConductance(forward_func=model, layer=layer)

        return conductance

    def calculate_features(
        self,
        model: torch.nn.Module,
        input_data: torch.Tensor,
        pred_label_idx: int,
        **kwargs,
    ) -> torch.Tensor:
        """Generate features image with Layer Conductance algorithm explainer.

        Args:
            model: Any DNN model You want to use.
            input_data: Input image.
            pred_label_idx: Predicted label.

        Returns:
            Features matrix.

        Raises:
            ValueError: if layer is None
            RuntimeError: if attribution has shape (0)
        """

        layer: Optional[torch.nn.Module] = kwargs.get("layer", None)
        if layer is None:
            raise ValueError("LayerConductanceCVExplainer" + LAYER_ARGUMENT_MISSING)

        conductance = self.create_explainer(model=model, layer=layer)
        attributions = conductance.attribute(
            input_data,
            baselines=torch.rand(  # pylint: disable = (no-member)
                1,
                input_data.shape[1],
                input_data.shape[2],
                input_data.shape[3],
                requires_grad=True,
            ).to(device=input_data.device),
            target=pred_label_idx,
        )
        super().validate_result(attributions=attributions)
        return attributions
