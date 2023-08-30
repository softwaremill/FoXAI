"""File with Integrated Gradients algorithm explainer classes.

Based on https://github.com/pytorch/captum/blob/master/captum/attr/_core/integrated_gradients.py
and https://github.com/pytorch/captum/blob/master/captum/attr/_core/layer/layer_integrated_gradients.py.
"""

from abc import abstractmethod
from typing import Any, List, Optional, Union

import torch
from captum._utils.typing import BaselineType
from captum.attr import IntegratedGradients, LayerIntegratedGradients

from foxai.array_utils import validate_result
from foxai.explainer.base_explainer import Explainer
from foxai.explainer.computer_vision.model_utils import (
    get_last_conv_model_layer,
    preprocess_baselines,
)
from foxai.types import AttributionsType, LayerType, ModelType, TargetType


class BaseIntegratedGradientsCVExplainer(Explainer):
    """Base Integrated Gradients algorithm explainer."""

    @abstractmethod
    def create_explainer(
        self,
        model: ModelType,
        multiply_by_inputs: bool = True,
        **kwargs,
    ) -> Union[IntegratedGradients, LayerIntegratedGradients]:
        """Create explainer object.

        Args:
            model: The forward function of the model or any
                modification of it.
            multiply_by_inputs: Indicates whether to factor
                model inputs' multiplier in the final attribution scores.
                In the literature this is also known as local vs global
                attribution. If inputs' multiplier isn't factored in,
                then that type of attribution method is also called local
                attribution. If it is, then that type of attribution
                method is called global.
                More detailed can be found here:
                https://arxiv.org/abs/1711.06104

                In case of integrated gradients, if `multiply_by_inputs`
                is set to True, final sensitivity scores are being multiplied by
                (inputs - baselines).

                In case of layer integrated gradients, if `multiply_by_inputs`
                is set to True, final sensitivity scores are being multiplied by
                layer activations for inputs - layer activations for baselines.

        Returns:
            Explainer object.
        """

    def calculate_features(
        self,
        model: ModelType,
        input_data: torch.Tensor,
        pred_label_idx: Optional[TargetType] = None,
        baselines: Optional[BaselineType] = None,
        additional_forward_args: Any = None,
        n_steps: int = 50,
        method: str = "gausslegendre",
        internal_batch_size: Optional[int] = None,
        attribute_to_layer_input: bool = False,
        layer: Optional[LayerType] = None,
        **kwargs,
    ) -> AttributionsType:
        """Generate model's attributes with Integrated Gradients algorithm explainer.

        Args:
            model: The forward function of the model or any
                modification of it.
            input_data: Input for which layer integrated
                gradients are computed. If forward_func takes a single
                tensor as input, a single input tensor should be provided.
            pred_label_idx: Output indices for
                which gradients are computed (for classification cases,
                this is usually the target class).
                If the network returns a scalar value per example,
                no target index is necessary.
                For general 2D outputs, targets can be either:

                - a single integer or a tensor containing a single
                    integer, which is applied to all input examples

                - a list of integers or a 1D tensor, with length matching
                    the number of examples in inputs (dim 0). Each integer
                    is applied as the target for the corresponding example.

                For outputs with > 2 dimensions, targets can be either:

                - A single tuple, which contains #output_dims - 1
                    elements. This target index is applied to all examples.

                - A list of tuples with length equal to the number of
                    examples in inputs (dim 0), and each tuple containing
                    #output_dims - 1 elements. Each tuple is applied as the
                    target for the corresponding example.

                Default: None
            baselines:
                Baselines define the starting point from which integral
                is computed and can be provided as:

                - a single tensor, if input_data is a single tensor, with
                    exactly the same dimensions as input_data or the first
                    dimension is one and the remaining dimensions match
                    with input_data.
                - a batch tensor, if input_data is a batch tensor, with
                    each tensor of a batch with exactly the same dimensions as
                    input_data and the first dimension is number of different baselines
                    to compute and their averaged score. Typical usage of batch
                    baselines is to provide random baselines and compute mean
                    attributes from them.

                In the cases when `baselines` is not provided, we internally
                use zero scalar corresponding to each input tensor.
                Default: None
            additional_forward_args: If the forward function
                requires additional arguments other than the inputs for
                which attributions should not be computed, this argument
                can be provided. It must be either a single additional
                argument of a Tensor or arbitrary (non-tuple) type or a
                tuple containing multiple additional arguments including
                tensors or any arbitrary python types. These arguments
                are provided to forward_func in order following the
                arguments in inputs.

                For a tensor, the first dimension of the tensor must
                correspond to the number of examples. It will be
                repeated for each of `n_steps` along the integrated
                path. For all other types, the given argument is used
                for all forward evaluations.

                Note that attributions are not computed with respect
                to these arguments.
                Default: None
            n_steps: The number of steps used by the approximation
                method. Default: 50.
            method: Method for approximating the integral,
                one of `riemann_right`, `riemann_left`, `riemann_middle`,
                `riemann_trapezoid` or `gausslegendre`.
                Default: `gausslegendre` if no method is provided.
            internal_batch_size: Divides total #steps * #examples
                data points into chunks of size at most internal_batch_size,
                which are computed (forward / backward passes)
                sequentially. internal_batch_size must be at least equal to
                #examples.

                For DataParallel models, each batch is split among the
                available devices, so evaluations on each available
                device contain internal_batch_size / num_devices examples.
                If internal_batch_size is None, then all evaluations are
                processed in one batch.
                Default: None
            attribute_to_layer_input: Indicates whether to
                compute the attribution with respect to the layer input
                or output. If `attribute_to_layer_input` is set to True
                then the attributions will be computed with respect to
                layer input, otherwise it will be computed with respect
                to layer output.

                Note that currently it is assumed that either the input
                or the output of internal layer, depending on whether we
                attribute to the input or output, is a single tensor.
                Support for multiple tensors will be added later.
                Default: False
            layer: Layer for which attributions are computed.
                If None provided, last convolutional layer from the model
                is taken.
                Default: None

        Returns:
            Integrated gradients with respect to `layer`'s inputs
            or outputs. Attributions will always be the same size and
            dimensionality as the input or output of the given layer,
            depending on whether we attribute to the inputs or outputs
            of the layer which is decided by the input flag
            `attribute_to_layer_input`.

            For a single layer, attributions are returned in a tuple if
            the layer inputs / outputs contain multiple tensors,
            otherwise a single tensor is returned.

            For multiple layers, attributions will always be
            returned as a list. Each element in this list will be
            equivalent to that of a single layer output, i.e. in the
            case that one layer, in the given layers, inputs / outputs
            multiple tensors: the corresponding output element will be
            a tuple of tensors. The ordering of the outputs will be
            the same order as the layers given in the constructor.

        Raises:
            RuntimeError: if attribution has shape (0).
        """
        attributions: AttributionsType
        integrated_gradients = self.create_explainer(model=model, layer=layer)

        attributions_list: List[AttributionsType] = []
        baselines_list, aggregate_attributes = preprocess_baselines(
            baselines=baselines,
            input_data_shape=input_data.shape,
        )

        for baseline in baselines_list:
            if isinstance(integrated_gradients, LayerIntegratedGradients):
                attributions = integrated_gradients.attribute(
                    input_data,
                    target=pred_label_idx,
                    n_steps=n_steps,
                    baselines=baseline,
                    return_convergence_delta=False,
                    additional_forward_args=additional_forward_args,
                    method=method,
                    internal_batch_size=internal_batch_size,
                    attribute_to_layer_input=attribute_to_layer_input,
                )
            else:
                attributions = integrated_gradients.attribute(
                    input_data,
                    target=pred_label_idx,
                    baselines=baseline,
                    n_steps=n_steps,
                    return_convergence_delta=False,
                    additional_forward_args=additional_forward_args,
                    method=method,
                    internal_batch_size=internal_batch_size,
                )
            validate_result(attributions=attributions)
            # if aggregation of attributes is required make sure that dimension of
            # stacked attributes have baseline number dimension
            if aggregate_attributes:
                attributions = attributions.unsqueeze(0)

            attributions_list.append(attributions)

        attributions = torch.vstack(attributions_list)
        if aggregate_attributes:
            attributions = torch.mean(attributions, dim=0)

        return attributions


class IntegratedGradientsCVExplainer(BaseIntegratedGradientsCVExplainer):
    """Integrated Gradients algorithm explainer."""

    def create_explainer(
        self,
        model: ModelType,
        multiply_by_inputs: bool = True,
        **kwargs,
    ) -> Union[IntegratedGradients, LayerIntegratedGradients]:
        """Create explainer object.

        Args:
            model: The forward function of the model or any
                modification of it.
            multiply_by_inputs: Indicates whether to factor
                model inputs' multiplier in the final attribution scores.
                In the literature this is also known as local vs global
                attribution. If inputs' multiplier isn't factored in,
                then that type of attribution method is also called local
                attribution. If it is, then that type of attribution
                method is called global.
                More detailed can be found here:
                https://arxiv.org/abs/1711.06104

                In case of integrated gradients, if `multiply_by_inputs`
                is set to True, final sensitivity scores are being multiplied by
                (inputs - baselines).

        Returns:
            Explainer object.
        """

        return IntegratedGradients(
            forward_func=model,
            multiply_by_inputs=multiply_by_inputs,
        )


class LayerIntegratedGradientsCVExplainer(BaseIntegratedGradientsCVExplainer):
    """Layer Integrated Gradients algorithm explainer."""

    def create_explainer(
        self,
        model: ModelType,
        multiply_by_inputs: bool = True,
        layer: Optional[LayerType] = None,
        **kwargs,
    ) -> Union[IntegratedGradients, LayerIntegratedGradients]:
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
            multiply_by_inputs: Indicates whether to factor
                model inputs' multiplier in the final attribution scores.
                In the literature this is also known as local vs global
                attribution. If inputs' multiplier isn't factored in,
                then that type of attribution method is also called local
                attribution. If it is, then that type of attribution
                method is called global.
                More detailed can be found here:
                https://arxiv.org/abs/1711.06104

                In case of layer integrated gradients, if `multiply_by_inputs`
                is set to True, final sensitivity scores are being multiplied by
                layer activations for inputs - layer activations for baselines.

        Returns:
            Explainer object.

        Raises:
            ValueError: if model does not contain conv layers.
        """
        if layer is None:
            layer = get_last_conv_model_layer(model=model)

        return LayerIntegratedGradients(
            forward_func=model,
            multiply_by_inputs=multiply_by_inputs,
            layer=layer,
        )
