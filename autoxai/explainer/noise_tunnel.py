"""File with Noise Tunnel algorithm explainer classes."""

from abc import abstractmethod
from typing import Optional

import torch
from captum.attr import IntegratedGradients, LayerIntegratedGradients, NoiseTunnel

from autoxai.explainer.occulusion import CVExplainer


class BaseNoiseTunnelCVExplainer(CVExplainer):
    """Base Noise Tunnel algorithm explainer."""

    @abstractmethod
    def create_explainer(self, **kwargs) -> NoiseTunnel:
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
        return attributions


class NoiseTunnelCVExplainer(BaseNoiseTunnelCVExplainer):
    """Noise Tunnel algorithm explainer."""

    def create_explainer(self, **kwargs) -> NoiseTunnel:
        """Create explainer object.

        Raises:
            RuntimeError: When passed arguments are invalid.

        Returns:
            Explainer object.
        """
        model: Optional[torch.nn.Module] = kwargs.get("model", None)
        if model is None:
            raise RuntimeError(f"Missing or `None` argument `model` passed: {kwargs}")

        integrated_gradients = IntegratedGradients(forward_func=model)
        return NoiseTunnel(integrated_gradients)


class LayerNoiseTunnelCVExplainer(BaseNoiseTunnelCVExplainer):
    """Layer Noise Tunnel algorithm explainer."""

    def create_explainer(self, **kwargs) -> NoiseTunnel:
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

        integrated_gradients = LayerIntegratedGradients(forward_func=model, layer=layer)
        return NoiseTunnel(integrated_gradients)
