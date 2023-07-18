"""File with DeepLIFT algorithm explainer classes.

Based on https://github.com/pytorch/captum/blob/master/captum/attr/_core/deep_lift.py
and https://github.com/pytorch/captum/blob/master/captum/attr/_core/layer/layer_deep_lift.py.
"""

from abc import abstractmethod
from typing import Any, Callable, Optional, Tuple, Union

import torch
from captum._utils.typing import TargetType
from captum.attr import DeepLift, LayerDeepLift

from foxai.array_utils import validate_result
from foxai.explainer.base_explainer import Explainer
from foxai.explainer.computer_vision.model_utils import (
    get_last_conv_model_layer,
    modify_modules,
)


class BaseDeepLIFTCVExplainer(Explainer):
    """Base DeepLIFT algorithm explainer."""

    @abstractmethod
    def create_explainer(
        self,
        model: torch.nn.Module,
        multiply_by_inputs: bool = True,
        **kwargs,
    ) -> Union[DeepLift, LayerDeepLift]:
        """Create explainer object.

        Args:
            model: The forward function of the model or any
                modification of it.
            multiply_by_inputs: Indicates whether to factor
                model inputs' multiplier in the final attribution scores.
                In the literature this is also known as local vs global
                attribution. If inputs' multiplier isn't factored in
                then that type of attribution method is also called local
                attribution. If it is, then that type of attribution
                method is called global.
                More detailed can be found here:
                https://arxiv.org/abs/1711.06104

                In case of DeepLift, if `multiply_by_inputs`
                is set to True, final sensitivity scores
                are being multiplied by (inputs - baselines).
                This flag applies only if `custom_attribution_func` is
                set to None.

        Returns:
            Explainer object.
        """

    def calculate_features(
        self,
        model: torch.nn.Module,
        input_data: torch.Tensor,
        pred_label_idx: TargetType = None,
        baselines: Union[None, int, float, torch.Tensor] = None,
        additional_forward_args: Any = None,
        custom_attribution_func: Union[
            None, Callable[..., Tuple[torch.Tensor, ...]]
        ] = None,
        attribute_to_layer_input: bool = False,
        **kwargs,
    ) -> torch.Tensor:
        """Generate model's attributes with DeepLIFT algorithm explainer.

        Args:
            model: The forward function of the model or any
                modification of it.
            input_data: Input for which
                attributions are computed. If forward_func takes a single
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
            baselines: Baselines define reference samples that are compared with
                the inputs. In order to assign attribution scores DeepLift
                computes the differences between the inputs/outputs and
                corresponding references.
                Baselines can be provided as:

                - a single tensor, if inputs is a single tensor, with
                    exactly the same dimensions as inputs or the first
                    dimension is one and the remaining dimensions match
                    with inputs.

                - a single scalar, if inputs is a single tensor, which will
                    be broadcasted for each input value in input tensor.

                In the cases when `baselines` is not provided, we internally
                use zero scalar corresponding to each input tensor.
                Default: None
            additional_forward_args: If the forward function
                requires additional arguments other than the inputs for
                which attributions should not be computed, this argument
                can be provided. It must be either a single additional
                argument of a Tensor or arbitrary (non-tuple) type or a tuple
                containing multiple additional arguments including tensors
                or any arbitrary python types. These arguments are provided to
                forward_func in order, following the arguments in inputs.
                Note that attributions are not computed with respect
                to these arguments.
                Default: None
            custom_attribution_func: A custom function for
                computing final attribution scores. This function can take
                at least one and at most three arguments with the
                following signature:

                - custom_attribution_func(multipliers)

                - custom_attribution_func(multipliers, inputs)

                - custom_attribution_func(multipliers, inputs, baselines)

                In case this function is not provided, we use the default
                logic defined as: multipliers * (inputs - baselines)
                It is assumed that all input arguments, `multipliers`,
                `inputs` and `baselines` are provided in tuples of same
                length. `custom_attribution_func` returns a tuple of
                attribution tensors that have the same length as the
                `inputs`.
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

        Returns:
            Attribution score computed based on DeepLift rescale rule with respect
            to each input feature. Attributions will always be
            the same size as the provided inputs, with each value
            providing the attribution of the corresponding input index.
            If a single tensor is provided as inputs, a single tensor is
            returned. If a tuple is provided for inputs, a tuple of
            corresponding sized tensors is returned.

        Raises:
            RuntimeError: if attribution has shape (0).
        """
        layer: Optional[torch.nn.Module] = kwargs.get("layer", None)
        deeplift = self.create_explainer(model=model, layer=layer)

        if baselines is None:
            baselines = torch.randn(
                input_data.shape,
                requires_grad=True,
                device=input_data.device,
            )

        if isinstance(deeplift, LayerDeepLift):
            attributions = deeplift.attribute(
                input_data,
                target=pred_label_idx,
                baselines=baselines,
                return_convergence_delta=False,
                additional_forward_args=additional_forward_args,
                custom_attribution_func=custom_attribution_func,
                attribute_to_layer_input=attribute_to_layer_input,
            )
        else:
            attributions = deeplift.attribute(
                input_data,
                target=pred_label_idx,
                baselines=baselines,
                return_convergence_delta=False,
                additional_forward_args=additional_forward_args,
                custom_attribution_func=custom_attribution_func,
            )
        validate_result(attributions=attributions)
        return attributions


class DeepLIFTCVExplainer(BaseDeepLIFTCVExplainer):
    """DeepLIFTC algorithm explainer."""

    def create_explainer(
        self,
        model: torch.nn.Module,
        multiply_by_inputs: bool = True,
        eps: float = 1e-10,
        **kwargs,
    ) -> Union[DeepLift, LayerDeepLift]:
        """Create explainer object.

        Args:
            model: The forward function of the model or any
                modification of it.
            multiply_by_inputs: Indicates whether to factor
                model inputs' multiplier in the final attribution scores.
                In the literature this is also known as local vs global
                attribution. If inputs' multiplier isn't factored in
                then that type of attribution method is also called local
                attribution. If it is, then that type of attribution
                method is called global.
                More detailed can be found here:
                https://arxiv.org/abs/1711.06104

                In case of DeepLift, if `multiply_by_inputs`
                is set to True, final sensitivity scores
                are being multiplied by (inputs - baselines).
                This flag applies only if `custom_attribution_func` is
                set to None.
            eps: A value at which to consider output/input change
                significant when computing the gradients for non-linear layers.
                This is useful to adjust, depending on your model's bit depth,
                to avoid numerical issues during the gradient computation.
                Default: 1e-10

        Returns:
            Explainer object.
        """
        model = modify_modules(model)

        return DeepLift(
            model=model,
            multiply_by_inputs=multiply_by_inputs,
            eps=eps,
        )


class LayerDeepLIFTCVExplainer(BaseDeepLIFTCVExplainer):
    """Layer DeepLIFT algorithm explainer."""

    def create_explainer(
        self,
        model: torch.nn.Module,
        multiply_by_inputs: bool = True,
        layer: Optional[torch.nn.Module] = None,
        **kwargs,
    ) -> Union[DeepLift, LayerDeepLift]:
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
                then that type of attribution method is also called local
                attribution. If it is, then that type of attribution
                method is called global.
                More detailed can be found here:
                https://arxiv.org/abs/1711.06104

                In case of DeepLift, if `multiply_by_inputs`
                is set to True, final sensitivity scores
                are being multiplied by (inputs - baselines).
                This flag applies only if `custom_attribution_func` is
                set to None.

        Returns:
            Explainer object.

        Raises:
            ValueError: if model does not contain conv layers.
        """
        if layer is None:
            layer = get_last_conv_model_layer(model=model)

        model = modify_modules(model)

        return LayerDeepLift(
            model=model,
            layer=layer,
            multiply_by_inputs=multiply_by_inputs,
        )
