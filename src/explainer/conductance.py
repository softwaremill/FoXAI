"""File with Conductance algorithm explainer classes."""

from typing import Optional

import torch
from captum.attr import LayerConductance

from src.explainer.base_explainer import CVExplainer


class LayerConductanceCVExplainer(CVExplainer):
    """Layer Conductance algorithm explainer."""

    def create_explainer(self, **kwargs) -> LayerConductance:
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
        """
        layer: Optional[torch.nn.Module] = kwargs.get("layer", None)

        conductance = self.create_explainer(model=model, layer=layer)
        attributions = conductance.attribute(
            input_data,
            baselines=torch.rand(  # pylint: disable = (no-member)
                1,
                input_data.shape[1],
                input_data.shape[2],
                input_data.shape[3],
            ),
            target=pred_label_idx,
        )
        if attributions.shape[0] == 0:
            raise RuntimeError(
                "Error occured during attribution calculation. "
                + "Make sure You are applying this method to CNN network.",
            )
        return attributions
