"""File with Noise Tunnel algorithm explainer classes."""

from abc import abstractmethod
from typing import Optional, Union, Tuple

import torch
from captum.attr import IntegratedGradients, LayerIntegratedGradients, NoiseTunnel

from autoxai.array_utils import validate_result
from autoxai.explainer.base_explainer import CVExplainer
from autoxai.explainer.model_utils import get_last_conv_model_layer


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

        Raises:
            RuntimeError: if attribution has shape (0).
        """
        nt_type: str = kwargs.get("nt_type", "smoothgrad_sq")
        nt_samples: int = kwargs.get("nt_samples", 10)
        nt_samples_batch_size: Optional[int] = kwargs.get("nt_samples_batch_size", None)
        stdevs: Union[float, Tuple[float, ...]] = kwargs.get("stdevs", 1.0)
        draw_baseline_from_distrib: bool = kwargs.get("draw_baseline_from_distrib", False)
        
        layer: Optional[torch.nn.Module] = kwargs.get("layer", None)

        noise_tunnel = self.create_explainer(model=model, layer=layer)

        attributions = noise_tunnel.attribute(
            inputs=input_data, 
            nt_type=nt_type, 
            nt_samples=nt_samples,
            nt_samples_batch_size=nt_samples_batch_size,
            stdevs=stdevs,
            draw_baseline_from_distrib=draw_baseline_from_distrib,
            target=pred_label_idx,
        )
        validate_result(attributions=attributions)
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
            ValueError: if model does not contain conv layers.
        """
        if layer is None:
            layer = get_last_conv_model_layer(model=model)

        integrated_gradients = LayerIntegratedGradients(forward_func=model, layer=layer)
        return NoiseTunnel(integrated_gradients)
