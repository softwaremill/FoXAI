"""File with GradCAM algorithm explainer classes.

Based on https://github.com/pytorch/captum/blob/master/captum/attr/_core/guided_grad_cam.py
and https://github.com/pytorch/captum/blob/master/captum/attr/_core/layer/grad_cam.py.
"""

from abc import abstractmethod
from typing import Any, Optional, Union

import torch
from captum._utils.typing import TargetType
from captum.attr import GuidedGradCam, LayerGradCam

from autoxai.array_utils import validate_result
from autoxai.explainer.base_explainer import CVExplainer
from autoxai.explainer.model_utils import get_last_conv_model_layer, modify_modules


class BaseGradCAMCVExplainer(CVExplainer):
    """Base GradCAM algorithm explainer."""

    @abstractmethod
    def create_explainer(
        self, model: torch.nn.Module, layer: torch.nn.Module, **kwargs
    ) -> Union[GuidedGradCam, LayerGradCam]:
        """Create explainer object.

        Args:
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

    @abstractmethod
    def calculate_features(
        self,
        model: torch.nn.Module,
        input_data: torch.Tensor,
        pred_label_idx: TargetType = None,
        additional_forward_args: Any = None,
        attribute_to_layer_input: bool = False,
        **kwargs,
    ) -> torch.Tensor:
        """Generate features image with GradCAM algorithm explainer.

        Args:
            model: The forward function of the model or any
                modification of it.
            input_data: Input for which attributions
                are computed. If forward_func takes a single
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
                argument of a Tensor or arbitrary (non-tuple) type or a
                tuple containing multiple additional arguments including
                tensors or any arbitrary python types. These arguments
                are provided to forward_func in order following the
                arguments in inputs.
                Note that attributions are not computed with respect
                to these arguments.
                Default: None
            attribute_to_layer_input: Indicates whether to
                compute the attribution with respect to the layer input
                or output in `LayerGradCam`.
                If `attribute_to_layer_input` is set to True
                then the attributions will be computed with respect to
                layer inputs, otherwise it will be computed with respect
                to layer outputs.
                Note that currently it is assumed that either the input
                or the output of internal layer, depending on whether we
                attribute to the input or output, is a single tensor.
                Support for multiple tensors will be added later.
                Default: False

        Returns:
            Element-wise product of (upsampled) GradCAM
            and/or Guided Backprop attributions.
            If a single tensor is provided as inputs, a single tensor is
            returned. If a tuple is provided for inputs, a tuple of
            corresponding sized tensors is returned.
            Attributions will be the same size as the provided inputs,
            with each value providing the attribution of the
            corresponding input index.
            If the GradCAM attributions cannot be upsampled to the shape
            of a given input tensor, None is returned in the corresponding
            index position.
        """


class GuidedGradCAMCVExplainer(BaseGradCAMCVExplainer):
    """GuidedGradCAM algorithm explainer."""

    def create_explainer(
        self,
        model: torch.nn.Module,
        layer: torch.nn.Module,
        **kwargs,
    ) -> Union[GuidedGradCam, LayerGradCam]:
        """Create explainer object.

        Args:
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
        model = modify_modules(model)

        return GuidedGradCam(model=model, layer=layer)

    def calculate_features(
        self,
        model: torch.nn.Module,
        input_data: torch.Tensor,
        pred_label_idx: TargetType = None,
        additional_forward_args: Any = None,
        attribute_to_layer_input: bool = False,
        interpolate_mode: str = "nearest",
        **kwargs,
    ) -> torch.Tensor:
        """Generate model's attributes with GradCAM algorithm explainer.

        Args:
            model: The forward function of the model or any
                modification of it.
            input_data: Input for which attributions
                are computed. If forward_func takes a single
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
                argument of a Tensor or arbitrary (non-tuple) type or a
                tuple containing multiple additional arguments including
                tensors or any arbitrary python types. These arguments
                are provided to forward_func in order following the
                arguments in inputs.
                Note that attributions are not computed with respect
                to these arguments.
                Default: None
            interpolate_mode: Method for interpolation, which
                must be a valid input interpolation mode for
                torch.nn.functional. These methods are
                "nearest", "area", "linear" (3D-only), "bilinear"
                (4D-only), "bicubic" (4D-only), "trilinear" (5D-only)
                based on the number of dimensions of the chosen layer
                output (which must also match the number of
                dimensions for the input tensor). Note that
                the original GradCAM paper uses "bilinear"
                interpolation, but we default to "nearest" for
                applicability to any of 3D, 4D or 5D tensors.
                Default: "nearest"
            attribute_to_layer_input: Indicates whether to
                compute the attribution with respect to the layer input
                or output in `LayerGradCam`.
                If `attribute_to_layer_input` is set to True
                then the attributions will be computed with respect to
                layer inputs, otherwise it will be computed with respect
                to layer outputs.
                Note that currently it is assumed that either the input
                or the output of internal layer, depending on whether we
                attribute to the input or output, is a single tensor.
                Support for multiple tensors will be added later.
                Default: False

        Returns:
            Element-wise product of (upsampled) GradCAM
            and/or Guided Backprop attributions.
            If a single tensor is provided as inputs, a single tensor is
            returned. If a tuple is provided for inputs, a tuple of
            corresponding sized tensors is returned.
            Attributions will be the same size as the provided inputs,
            with each value providing the attribution of the
            corresponding input index.
            If the GradCAM attributions cannot be upsampled to the shape
            of a given input tensor, None is returned in the corresponding
            index position.

        Raises:
            ValueError: if model does not contain conv layers.
            RuntimeError: if attributions has shape (0)
        """
        layer: Optional[torch.nn.Module] = kwargs.get("layer", None)
        if layer is None:
            layer = get_last_conv_model_layer(model=model)

        guided_cam = self.create_explainer(model=model, layer=layer)

        if isinstance(guided_cam, GuidedGradCam):
            attributions = guided_cam.attribute(
                input_data,
                target=pred_label_idx,
                additional_forward_args=additional_forward_args,
                interpolate_mode=interpolate_mode,
                attribute_to_layer_input=attribute_to_layer_input,
            )
        validate_result(attributions=attributions)
        return attributions


class LayerGradCAMCVExplainer(BaseGradCAMCVExplainer):
    """Layer GradCAM algorithm explainer."""

    def create_explainer(
        self,
        model: torch.nn.Module,
        layer: torch.nn.Module,
        **kwargs,
    ) -> Union[GuidedGradCam, LayerGradCam]:
        """Create explainer object.

        Args:
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
        model = modify_modules(model)

        return LayerGradCam(forward_func=model, layer=layer)

    def calculate_features(
        self,
        model: torch.nn.Module,
        input_data: torch.Tensor,
        pred_label_idx: TargetType = None,
        additional_forward_args: Any = None,
        attribute_to_layer_input: bool = False,
        relu_attributions: bool = False,
        # attr_dim_summation: bool = True,
        **kwargs,
    ) -> torch.Tensor:
        """Generate features image with GradCAM algorithm explainer.

        Args:
            model: The forward function of the model or any
                modification of it.
            inputs_data: Input for which attributions
                are computed. If forward_func takes a single
                tensor as input, a single input tensor should be provided.
                If forward_func takes multiple tensors as input, a tuple
                of the input tensors should be provided. It is assumed
                that for all given input tensors, dimension 0 corresponds
                to the number of examples, and if multiple input tensors
                are provided, the examples must be aligned appropriately.
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
                argument of a Tensor or arbitrary (non-tuple) type or a
                tuple containing multiple additional arguments including
                tensors or any arbitrary python types. These arguments
                are provided to forward_func in order following the
                arguments in inputs.
                Note that attributions are not computed with respect
                to these arguments.
                Default: None
            attribute_to_layer_input: Indicates whether to
                compute the attribution with respect to the layer input
                or output in `LayerGradCam`.
                If `attribute_to_layer_input` is set to True
                then the attributions will be computed with respect to
                layer inputs, otherwise it will be computed with respect
                to layer outputs.
                Note that currently it is assumed that either the input
                or the output of internal layer, depending on whether we
                attribute to the input or output, is a single tensor.
                Support for multiple tensors will be added later.
                Default: False
            relu_attributions: Indicates whether to
                apply a ReLU operation on the final attribution,
                returning only non-negative attributions. Setting this
                flag to True matches the original GradCAM algorithm,
                otherwise, by default, both positive and negative
                attributions are returned.
                Default: False
            attr_dim_summation: Indicates whether to
                sum attributions along dimension 1 (usually channel).
                The default (True) means to sum along dimension 1.
                Default: True

        Returns:
            Element-wise product of (upsampled) GradCAM
            and/or Guided Backprop attributions.
            If a single tensor is provided as inputs, a single tensor is
            returned. If a tuple is provided for inputs, a tuple of
            corresponding sized tensors is returned.
            Attributions will be the same size as the provided inputs,
            with each value providing the attribution of the
            corresponding input index.
            If the GradCAM attributions cannot be upsampled to the shape
            of a given input tensor, None is returned in the corresponding
            index position.

        Raises:
            ValueError: if model does not contain conv layers.
            RuntimeError: if attributions has shape (0)
        """
        layer: Optional[torch.nn.Module] = kwargs.get("layer", None)
        if layer is None:
            layer = get_last_conv_model_layer(model=model)

        guided_cam = self.create_explainer(model=model, layer=layer)
        if isinstance(guided_cam, LayerGradCam):
            attributions = guided_cam.attribute(
                input_data,
                target=pred_label_idx,
                additional_forward_args=additional_forward_args,
                attribute_to_layer_input=attribute_to_layer_input,
                relu_attributions=relu_attributions,
                # attr_dim_summation=attr_dim_summation,
            )
        validate_result(attributions=attributions)
        return attributions
