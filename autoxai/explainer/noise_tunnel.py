"""File with Noise Tunnel algorithm explainer classes."""

from abc import abstractmethod
from typing import Optional

import torch
from captum.attr import IntegratedGradients, LayerIntegratedGradients, NoiseTunnel

from autoxai.explainer.model_utils import get_last_conv_model_layer
from autoxai.explainer.occulusion import CVExplainer


class BaseNoiseTunnelCVExplainer(CVExplainer):
    """Base Noise Tunnel algorithm explainer."""

    @abstractmethod
    def create_explainer(self, model: torch.nn.Module, **kwargs) -> NoiseTunnel:
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
        """Generate features image with Noise Tunnel algorithm explainer.

        Args:
            model: Any DNN model You want to use.
            input_data: Input image.
            pred_label_idx: Predicted label.

        Returns:
            Features matrix.
        """
        nt_samples: int = kwargs.get("nt_samples", 10)
        nt_type: str = kwargs.get("nt_type", "smoothgrad_sq")
        layer: Optional[torch.nn.Module] = kwargs.get("layer", None)

        noise_tunnel = self.create_explainer(model=model, layer=layer)

        attributions = noise_tunnel.attribute(
            input_data, nt_samples=nt_samples, nt_type=nt_type, target=pred_label_idx
        )
        if attributions.shape[0] == 0:
            raise RuntimeError(
                "Error occured during attribution calculation. "
                + "Make sure You are applying this method to CNN network.",
            )
        return attributions


class NoiseTunnelCVExplainer(BaseNoiseTunnelCVExplainer):
    """Noise Tunnel algorithm explainer."""

    def create_explainer(
        self,
        model: torch.nn.Module,
        **kwargs,
    ) -> NoiseTunnel:
        """Create explainer object.

        Returns:
            Explainer object.
        """
        integrated_gradients = IntegratedGradients(forward_func=model)
        return NoiseTunnel(integrated_gradients)


class LayerNoiseTunnelCVExplainer(BaseNoiseTunnelCVExplainer):
    """Layer Noise Tunnel algorithm explainer."""

    def create_explainer(
        self,
        model: torch.nn.Module,
        layer: Optional[torch.nn.Module] = None,
        **kwargs,
    ) -> NoiseTunnel:
        """Create explainer object.

        Uses parameter `layer` from `kwargs`. If not provided function will call
        `get_last_conv_model_layer` function to obtain last `torch.nn.Conv2d` layer
        from provided model.

        Returns:
            Explainer object.

        Raises:
            ValueError: if model does not contain conv layers
        """
        if layer is None:
            layer = get_last_conv_model_layer(model=model)

        integrated_gradients = LayerIntegratedGradients(forward_func=model, layer=layer)
        return NoiseTunnel(integrated_gradients)
