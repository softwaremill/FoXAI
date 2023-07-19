"""File with Gradient SHAP algorithm explainer classes.

Based on https://github.com/pytorch/captum/blob/master/captum/attr/_core/gradient_shap.py
and https://github.com/pytorch/captum/blob/master/captum/attr/_core/layer/layer_gradient_shap.py.
"""

from abc import abstractmethod
from typing import Any, List, Optional, Union

import torch
from captum._utils.typing import BaselineType, TargetType
from captum.attr import GradientShap, LayerGradientShap

from foxai.array_utils import validate_result
from foxai.explainer.base_explainer import Explainer
from foxai.explainer.computer_vision.model_utils import (
    get_last_conv_model_layer,
    preprocess_baselines,
)
from foxai.types import AttributionsType, LayerType, ModelType, StdevsType


class BaseGradientSHAPCVExplainer(Explainer):
    """Base Gradient SHAP algorithm explainer."""

    @abstractmethod
    def create_explainer(
        self,
        model: ModelType,
        multiply_by_inputs: bool = True,
        **kwargs,
    ) -> Union[GradientShap, LayerGradientShap]:
        """Create explainer object.

        Args:
            model: The forward function of the model or any
                modification of it.
            multiply_by_inputs: Indicates whether to factor
                model inputs' multiplier in the final attribution scores.
                In the literature this is also known as local vs global
                attribution. If inputs' multiplier isn't factored in
                then this type of attribution method is also called local
                attribution. If it is, then that type of attribution
                method is called global.
                More detailed can be found here:
                https://arxiv.org/abs/1711.06104

                In case of gradient shap, if `multiply_by_inputs`
                is set to True, the sensitivity scores of scaled inputs
                are being multiplied by (inputs - baselines).

        Returns:
            Explainer object.
        """

    def calculate_features(
        self,
        model: ModelType,
        input_data: torch.Tensor,
        pred_label_idx: TargetType = None,
        baselines: BaselineType = None,
        n_samples: int = 5,
        stdevs: StdevsType = 0.0,
        additional_forward_args: Any = None,
        attribute_to_layer_input: bool = False,
        layer: Optional[LayerType] = None,
        **kwargs,
    ) -> AttributionsType:
        """Generate model's attributes with Gradient SHAP algorithm explainer.

        Args:
            model: The forward function of the model or any
                modification of it.
            input_data: Input for which SHAP attribution
                values are computed. If `forward_func` takes a single
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
                Baselines define the starting point from which expectation
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

                It is recommended that the number of samples in the baselines'
                tensors is larger than one.
            n_samples: The number of randomly generated examples
                per sample in the input batch. Random examples are
                generated by adding gaussian random noise to each sample.
                Default: `5` if `n_samples` is not provided.
            stdevs: The standard deviation
                of gaussian noise with zero mean that is added to each
                input in the batch. If `stdevs` is a single float value
                then that same value is used for all inputs. If it is
                a tuple, then it must have the same length as the inputs
                tuple. In this case, each stdev value in the stdevs tuple
                corresponds to the input with the same index in the inputs
                tuple.
                Default: 0.0
            additional_forward_args: If the forward function
                requires additional arguments other than the inputs for
                which attributions should not be computed, this argument
                can be provided. It can contain a tuple of ND tensors or
                any arbitrary python type of any shape.
                In case of the ND tensor the first dimension of the
                tensor must correspond to the batch size. It will be
                repeated for each `n_steps` for each randomly generated
                input sample.

                Note that the gradients are not computed with respect
                to these arguments.
                Default: None
            attribute_to_layer_input (bool, optional): Indicates whether to
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
            Attribution score computed based on GradientSHAP with respect
            to each input feature. Attributions will always be
            the same size as the provided inputs, with each value
            providing the attribution of the corresponding input index.
            If a single tensor is provided as inputs, a single tensor is
            returned. If a tuple is provided for inputs, a tuple of
            corresponding sized tensors is returned.

        Raises:
            RuntimeError: if attribution has shape (0).
        """
        gradient_shap = self.create_explainer(model=model, layer=layer)

        attributions_list: List[torch.Tensor] = []
        baselines_list, aggregate_attributes = preprocess_baselines(
            baselines=baselines,
            input_data_shape=input_data.shape,
        )

        for baseline in baselines_list:
            if isinstance(gradient_shap, LayerGradientShap):
                attributions = gradient_shap.attribute(
                    input_data,
                    n_samples=n_samples,
                    stdevs=stdevs,
                    baselines=baseline,
                    target=pred_label_idx,
                    return_convergence_delta=False,
                    additional_forward_args=additional_forward_args,
                    attribute_to_layer_input=attribute_to_layer_input,
                )
            else:
                attributions = gradient_shap.attribute(
                    input_data,
                    n_samples=n_samples,
                    stdevs=stdevs,
                    baselines=baseline,
                    target=pred_label_idx,
                    return_convergence_delta=False,
                    additional_forward_args=additional_forward_args,
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


class GradientSHAPCVExplainer(BaseGradientSHAPCVExplainer):
    """Gradient SHAP algorithm explainer."""

    def create_explainer(
        self,
        model: ModelType,
        multiply_by_inputs: bool = True,
        **kwargs,
    ) -> Union[GradientShap, LayerGradientShap]:
        """Create explainer object.

        Args:
            model: The forward function of the model or any
                modification of it.
            multiply_by_inputs: Indicates whether to factor
                model inputs' multiplier in the final attribution scores.
                In the literature this is also known as local vs global
                attribution. If inputs' multiplier isn't factored in
                then this type of attribution method is also called local
                attribution. If it is, then that type of attribution
                method is called global.
                More detailed can be found here:
                https://arxiv.org/abs/1711.06104

                In case of gradient shap, if `multiply_by_inputs`
                is set to True, the sensitivity scores of scaled inputs
                are being multiplied by (inputs - baselines).

        Returns:
            Explainer object.
        """

        return GradientShap(
            forward_func=model,
            multiply_by_inputs=multiply_by_inputs,
        )


class LayerGradientSHAPCVExplainer(BaseGradientSHAPCVExplainer):
    """Layer Gradient SHAP algorithm explainer."""

    def create_explainer(
        self,
        model: ModelType,
        multiply_by_inputs: bool = True,
        layer: Optional[LayerType] = None,
        **kwargs,
    ) -> Union[GradientShap, LayerGradientShap]:
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
                attribution. If inputs' multiplier isn't factored in
                then this type of attribution method is also called local
                attribution. If it is, then that type of attribution
                method is called global.
                More detailed can be found here:
                https://arxiv.org/abs/1711.06104

                In case of layer gradient shap, if `multiply_by_inputs`
                is set to True, the sensitivity scores for scaled inputs
                are being multiplied by
                layer activations for inputs - layer activations for baselines.
            layer: Layer for which attributions are computed.
                If None provided, last convolutional layer from the model
                is taken.
                Default: None

        Returns:
            Explainer object.

        Raises:
            ValueError: if model does not contain conv layers.
        """

        if layer is None:
            layer = get_last_conv_model_layer(model=model)

        return LayerGradientShap(
            forward_func=model,
            layer=layer,
            multiply_by_inputs=multiply_by_inputs,
        )
