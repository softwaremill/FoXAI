"""File with Input X Gradient algorithm explainer classes.

Based on https://github.com/pytorch/captum/blob/master/captum/attr/_core/input_x_gradient.py
and https://github.com/pytorch/captum/blob/master/captum/attr/_core/layer/layer_gradient_x_activation.py.
"""

from abc import abstractmethod
from typing import Any, Optional, Union

import torch
from captum._utils.typing import TargetType
from captum.attr import InputXGradient, LayerGradientXActivation

from foxai.array_utils import validate_result
from foxai.explainer.base_explainer import CVExplainer
from foxai.explainer.model_utils import get_last_conv_model_layer


class BaseInputXGradientSHAPCVExplainer(CVExplainer):
    """Base Input X Gradient algorithm explainer."""

    @abstractmethod
    def create_explainer(
        self,
        model: torch.nn.Module,
        **kwargs,
    ) -> Union[InputXGradient, LayerGradientXActivation]:
        """Create explainer object.

        Args:
            model: The forward function of the model or any
                modification of it.

        Returns:
            Explainer object.
        """

    def calculate_features(
        self,
        model: torch.nn.Module,
        input_data: torch.Tensor,
        pred_label_idx: TargetType = None,
        additional_forward_args: Any = None,
        attribute_to_layer_input: bool = False,
        **kwargs,
    ) -> torch.Tensor:
        """Generate model's attributes with Input X Gradient algorithm explainer.

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
            additional_forward_args: If the forward function
                requires additional arguments other than the inputs for
                which attributions should not be computed, this argument
                can be provided. It must be either a single additional
                argument of a Tensor or arbitrary (non-tuple) type or a tuple
                containing multiple additional arguments including tensors
                or any arbitrary python types. These arguments are provided to
                forward_func in order following the arguments in inputs.
                Note that attributions are not computed with respect
                to these arguments.
                Default: None
            attribute_to_layer_input: Indicates whether to
                compute the attribution with respect to the layer input
                or output. If `attribute_to_layer_input` is set to True
                then the attributions will be computed with respect to
                layer input, otherwise it will be computed with respect
                to layer output.
                Default: False

        Returns:
            The input x gradient with respect to each input feature or gradient
            and activation for each neuron in given layer output. Attributions
            will always be the same size as the provided inputs, with each value
            providing the attribution of the corresponding input index.
            If a single tensor is provided as inputs, a single tensor is
            returned. If a tuple is provided for inputs, a tuple of
            corresponding sized tensors is returned.

        Raises:
            RuntimeError: if attribution has shape (0).
        """
        layer: Optional[torch.nn.Module] = kwargs.get("layer", None)
        input_x_gradient = self.create_explainer(model=model, layer=layer)

        if isinstance(input_x_gradient, LayerGradientXActivation):
            attributions = input_x_gradient.attribute(
                input_data,
                target=pred_label_idx,
                additional_forward_args=additional_forward_args,
                attribute_to_layer_input=attribute_to_layer_input,
            )
        else:
            attributions = input_x_gradient.attribute(
                input_data,
                target=pred_label_idx,
                additional_forward_args=additional_forward_args,
            )
        validate_result(attributions=attributions)
        return attributions


class InputXGradientCVExplainer(BaseInputXGradientSHAPCVExplainer):
    """Input X Gradient algorithm explainer."""

    def create_explainer(
        self,
        model: torch.nn.Module,
        **kwargs,
    ) -> Union[InputXGradient, LayerGradientXActivation]:
        """Create explainer object.

        Args:
            model: The forward function of the model or any
                modification of it.

        Returns:
            Explainer object.
        """

        return InputXGradient(
            forward_func=model,
        )


class LayerInputXGradientCVExplainer(BaseInputXGradientSHAPCVExplainer):
    """Layer Input X Gradient algorithm explainer."""

    def create_explainer(
        self,
        model: torch.nn.Module,
        layer: Optional[torch.nn.Module] = None,
        multiply_by_inputs: bool = True,
        **kwargs,
    ) -> Union[InputXGradient, LayerGradientXActivation]:
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
                then this type of attribution method is also called local
                attribution. If it is, then that type of attribution
                method is called global.
                More detailed can be found here:
                https://arxiv.org/abs/1711.06104

                In case of layer gradient x activation, if `multiply_by_inputs`
                is set to True, final sensitivity scores are being multiplied by
                layer activations for inputs.

        Returns:
            Explainer object.

        Raises:
            ValueError: if model does not contain conv layers.
        """

        if layer is None:
            layer = get_last_conv_model_layer(model=model)

        return LayerGradientXActivation(
            forward_func=model,
            layer=layer,
            multiply_by_inputs=multiply_by_inputs,
        )
