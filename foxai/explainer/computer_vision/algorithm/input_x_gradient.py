"""File with Input X Gradient algorithm explainer classes.

Based on https://github.com/pytorch/captum/blob/master/captum/attr/_core/input_x_gradient.py
and https://github.com/pytorch/captum/blob/master/captum/attr/_core/layer/layer_gradient_x_activation.py.
"""
from abc import abstractmethod
from typing import Any, Optional, Tuple

import torch

from foxai.array_utils import validate_result
from foxai.explainer.base_explainer import Explainer
from foxai.explainer.computer_vision.algorithm.gradient_utils import (
    compute_gradients,
    compute_layer_gradients,
    require_grad,
)
from foxai.explainer.computer_vision.model_utils import get_last_conv_model_layer
from foxai.types import AttributionsType, LayerType, ModelType


class BaseInputXGradientSHAPCVExplainer(Explainer):
    """Base Input X Gradient algorithm explainer."""

    @abstractmethod
    def calculate_features(  # type: ignore[override]
        self,
        model: ModelType,
        input_data: torch.Tensor,
        pred_label_idx: Optional[int] = None,
        additional_forward_args: Optional[Tuple[Any]] = None,
        **kwargs,
    ) -> AttributionsType:
        pass


class InputXGradientCVExplainer(BaseInputXGradientSHAPCVExplainer):
    """Input X Gradient algorithm explainer."""

    def calculate_features(  # type: ignore[override]
        self,
        model: ModelType,
        input_data: torch.Tensor,
        pred_label_idx: Optional[int] = None,
        additional_forward_args: Optional[Tuple[Any]] = None,
        **kwargs,
    ) -> AttributionsType:
        """Generate model's attributes with Input X Gradient algorithm explainer.

        Args:
            model: The forward function of the model or any modification of it.
            input_data: Input for which attributions are computed.
            pred_label_idx: Output indices for
                which gradients are computed (for classification cases,
                this is usually the target class).
                If the network returns a scalar value per example,
                no target index is necessary.
                Targets can be either a single integer or a tensor containing a single
                integer, which is applied to all input examples.

                Default: None
            additional_forward_args: If the forward function
                requires additional arguments other than the inputs for
                which attributions should not be computed, this argument
                can be provided. It must be a tuple
                containing multiple additional arguments including tensors
                or any arbitrary python types. These arguments are provided to
                forward_func in order following the arguments in inputs.
                Note that attributions are not computed with respect
                to these arguments.
                Default: None

        Returns:
            The input x gradient with respect to each input feature or gradient
            and activation for each neuron in given layer output. Attributions
            will always be the same size as the provided input.

        Raises:
            RuntimeError: if attribution has shape (0).
        """
        with require_grad(input_data) as input_tensor:

            gradient = compute_gradients(
                model, input_tensor, pred_label_idx, additional_forward_args
            )

            attributions = input_data * gradient

        validate_result(attributions=attributions)
        return attributions


class LayerInputXGradientCVExplainer(BaseInputXGradientSHAPCVExplainer):
    """Layer Input X Gradient algorithm explainer."""

    def calculate_features(  # type: ignore[override]
        self,
        model: ModelType,
        input_data: torch.Tensor,
        pred_label_idx: Optional[int] = None,
        additional_forward_args: Optional[Tuple[Any]] = None,
        attribute_to_layer_input: bool = False,
        layer: Optional[LayerType] = None,
        **kwargs,
    ) -> AttributionsType:
        """Generate model's attributes with Input X Gradient algorithm explainer.

        Args:
            model: The forward function of the model or any modification of it.
            input_data: Input for which
                attributions are computed. If forward_func takes a single
                tensor as input, a single input tensor should be provided.
            pred_label_idx: Output indices for
                which gradients are computed (for classification cases,
                this is usually the target class).
                If the network returns a scalar value per example,
                no target index is necessary.
                Targets can be either a single integer or a tensor containing a single
                integer, which is applied to all input examples.

                Default: None
            additional_forward_args: If the forward function
                requires additional arguments other than the inputs for
                which attributions should not be computed, this argument
                can be provided. It must be a tuple
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
            layer: Layer for which attributions are computed.
                If None provided, last convolutional layer from the model
                is taken.
                Default: None

        Returns:
            The input x gradient with respect to each input feature or gradient
            and activation for each neuron in given layer output. Attributions
            will always be the same size as the provided input.

        Raises:
            RuntimeError: if attribution has shape (0).
        """

        if layer is None:
            layer = get_last_conv_model_layer(model=model)

        gradients, layer_evals = compute_layer_gradients(
            model,
            layer,
            input_data,
            pred_label_idx,
            additional_forward_args,
            attribute_to_layer_input=attribute_to_layer_input,
        )
        attributions = gradients * layer_evals

        validate_result(attributions=attributions)
        return attributions
