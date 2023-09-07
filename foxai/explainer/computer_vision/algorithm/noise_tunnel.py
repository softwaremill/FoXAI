"""File with Noise Tunnel algorithm explainer classes.

Based on https://github.com/pytorch/captum/blob/master/captum/attr/_core/noise_tunnel.py.
"""

from abc import abstractmethod
from typing import Optional

import torch
from captum.attr import IntegratedGradients, LayerIntegratedGradients, NoiseTunnel

from foxai.array_utils import validate_result
from foxai.explainer.base_explainer import Explainer
from foxai.explainer.computer_vision.model_utils import get_last_conv_model_layer
from foxai.types import AttributionsType, LayerType, ModelType, StdevsType


class BaseNoiseTunnelCVExplainer(Explainer):
    """Base Noise Tunnel algorithm explainer."""

    @abstractmethod
    def create_explainer(self, model: ModelType, **kwargs) -> NoiseTunnel:
        """Create explainer object.

        Args:
            model: The forward function of the model or any
                modification of it.

        Returns:
            Explainer object.
        """

    def calculate_features(
        self,
        model: ModelType,
        input_data: torch.Tensor,
        pred_label_idx: Optional[int] = None,
        nt_type: str = "smoothgrad",
        nt_samples: int = 5,
        nt_samples_batch_size: Optional[int] = None,
        stdevs: StdevsType = 1.0,
        draw_baseline_from_distrib: bool = False,
        layer: Optional[LayerType] = None,
        **kwargs,
    ) -> AttributionsType:
        """Generate model's attributes with Noise Tunnel algorithm explainer.

        Args:
            model: The forward function of the model or any
                modification of it.
            input_data: Input for which integrated
                gradients are computed. If forward_func takes a single
                tensor as input, a single input tensor should be provided.
            pred_label_idx: Output indices for
                which gradients are computed (for classification cases,
                this is usually the target class).
                If the network returns a scalar value per example,
                no target index is necessary.
                Default: None
            nt_type: Smoothing type of the attributions.
                `smoothgrad`, `smoothgrad_sq` or `vargrad`
                Default: `smoothgrad` if `type` is not provided.
            nt_samples: The number of randomly generated examples
                per sample in the input batch. Random examples are
                generated by adding gaussian random noise to each sample.
                Default: `5` if `nt_samples` is not provided.
            nt_samples_batch_size: The number of the `nt_samples`
                that will be processed together. With the help
                of this parameter we can avoid out of memory situation and
                reduce the number of randomly generated examples per sample
                in each batch.
                Default: None if `nt_samples_batch_size` is not provided. In
                this case all `nt_samples` will be processed together.
            stdevs: The standard deviation
                of gaussian noise with zero mean that is added to each
                input in the batch. If `stdevs` is a single float value
                then that same value is used for all inputs. If it is
                a tuple, then it must have the same length as the inputs
                tuple. In this case, each stdev value in the stdevs tuple
                corresponds to the input with the same index in the inputs
                tuple.
                Default: `1.0` if `stdevs` is not provided.
            draw_baseline_from_distrib: Indicates whether to
                randomly draw baseline samples from the `baselines`
                distribution provided as an input tensor.
                Default: False
            layer: Layer for which attributions are computed.
                Default: None

        Returns:
            Attribution with respect to each input feature. attributions
            will always be the same size as the provided inputs, with each value
            providing the attribution of the corresponding input index.
            If a single tensor is provided as inputs, a single tensor is
            returned.

        Raises:
            RuntimeError: if attribution has shape (0).
        """
        noise_tunnel = self.create_explainer(model=model, layer=layer)

        attributions: AttributionsType = noise_tunnel.attribute(
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
        model: ModelType,
        **kwargs,
    ) -> NoiseTunnel:
        """Create explainer object.

        Args:
            model: The forward function of the model or any
                modification of it.

        Returns:
            Explainer object.
        """
        integrated_gradients = IntegratedGradients(forward_func=model)
        return NoiseTunnel(integrated_gradients)


class LayerNoiseTunnelCVExplainer(BaseNoiseTunnelCVExplainer):
    """Layer Noise Tunnel algorithm explainer."""

    def create_explainer(
        self,
        model: ModelType,
        layer: Optional[LayerType] = None,
        **kwargs,
    ) -> NoiseTunnel:
        """Create explainer object.

        Uses parameter `layer` from `kwargs`. If not provided function will call
        `get_last_conv_model_layer` function to obtain last `torch.nn.Conv2d` layer
        from provided model.

        Args:
            model: The forward function of the model or any
                modification of it.
            layer: Layer for which attributions are computed.
                Output size of attribute matches this layer's input or
                output dimensions, depending on whether we attribute to
                the inputs or outputs of the layer, corresponding to
                attribution of each neuron in the input or output of
                this layer.
                Default: None

        Returns:
            Explainer object.

        Raises:
            ValueError: if model does not contain conv layers.
        """
        if layer is None:
            layer = get_last_conv_model_layer(model=model)

        integrated_gradients = LayerIntegratedGradients(forward_func=model, layer=layer)
        return NoiseTunnel(integrated_gradients)
