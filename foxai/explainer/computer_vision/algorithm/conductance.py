"""File with Conductance algorithm explainer classes.

Based on https://github.com/pytorch/captum/blob/master/captum/attr/_core/layer/layer_conductance.py.
"""

from typing import Any, List, Optional

import torch
from captum._utils.typing import BaselineType, TargetType
from captum.attr import LayerConductance

from foxai.array_utils import validate_result
from foxai.explainer.base_explainer import Explainer
from foxai.explainer.computer_vision.model_utils import (
    get_last_conv_model_layer,
    preprocess_baselines,
)
from foxai.types import AttributionsType, LayerType, ModelType


class LayerConductanceCVExplainer(Explainer):
    """Layer Conductance algorithm explainer."""

    # pylint: disable = unused-argument
    def create_explainer(
        self,
        model: ModelType,
        layer: LayerType,
    ) -> LayerConductance:
        """Create explainer object.

        model: The forward function of the model or any
            modification of it.
        layer: Layer for which attributions are computed.
            Output size of attribute matches this layer's input or
            output dimensions, depending on whether we attribute to
            the inputs or outputs of the layer, corresponding to
            attribution of each neuron in the input or output of
            this layer.

        Returns:
            Explainer object.
        """
        conductance = LayerConductance(forward_func=model, layer=layer)

        return conductance

    def calculate_features(
        self,
        model: ModelType,
        input_data: torch.Tensor,
        pred_label_idx: TargetType = None,
        baselines: BaselineType = None,
        additional_forward_args: Any = None,
        n_steps: int = 50,
        method: str = "gausslegendre",
        internal_batch_size: Optional[int] = None,
        attribute_to_layer_input: bool = False,
        layer: Optional[LayerType] = None,
        **kwargs,
    ) -> AttributionsType:
        """Generate model's attributes with Layer Conductance algorithm explainer.

        Args:
            model: The forward function of the model or any
                modification of it.
            input_data: Input for which layer
                conductance is computed. If forward_func takes a single
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
                correspond to the number of examples. It will be repeated
                for each of `n_steps` along the integrated path.
                For all other types, the given argument is used for
                all forward evaluations.
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
                2 * #examples.
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
                layer inputs, otherwise it will be computed with respect
                to layer outputs.
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
            Conductance of each neuron in given layer input or
            output. Attributions will always be the same size as
            the input or output of the given layer, depending on
            whether we attribute to the inputs or outputs
            of the layer which is decided by the input flag
            `attribute_to_layer_input`.
            Attributions are returned in a tuple if
            the layer inputs / outputs contain multiple tensors,
            otherwise a single tensor is returned.

        Raises:
            ValueError: if model does not contain conv layers.
            RuntimeError: if attribution has shape (0)
        """
        attributions: AttributionsType
        if layer is None:
            layer = get_last_conv_model_layer(model=model)

        conductance = self.create_explainer(model=model, layer=layer)

        attributions_list: List[AttributionsType] = []
        baselines_list, aggregate_attributes = preprocess_baselines(
            baselines=baselines,
            input_data_shape=input_data.shape,
        )

        for baseline in baselines_list:
            attributions = conductance.attribute(
                input_data,
                baselines=baseline,
                target=pred_label_idx,
                additional_forward_args=additional_forward_args,
                n_steps=n_steps,
                method=method,
                internal_batch_size=internal_batch_size,
                return_convergence_delta=False,
                attribute_to_layer_input=attribute_to_layer_input,
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
