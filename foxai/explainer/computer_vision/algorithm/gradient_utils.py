from inspect import signature
from typing import Any, Callable, Optional, Tuple

import torch
from torch.utils.hooks import RemovableHandle

from foxai.types import LayerType, ModelType


def _run_forward(
    forward_func: Callable,
    input_tensor: torch.Tensor,
    target: Optional[int] = None,
    additional_forward_args: Optional[Tuple[Any]] = None,
) -> torch.Tensor:
    r"""Executes forward pass and returns result with respect to the target.

    Args:
        forward_func: forward function. This can be for example model's forward function.
        input_tensor:      Input to be passed to forward_fn.
        target: Index of the target class for which gradients must be computed (classification only).
        additional_forward_args: Additional input arguments that forward function requires.
    """
    if len(signature(forward_func).parameters) == 0:
        output = forward_func()
    elif additional_forward_args is not None:
        output = forward_func((input_tensor, *additional_forward_args))
    else:
        output = forward_func(input_tensor)

    if target is None:
        return output
    return output[:, target]


def compute_gradients(
    forward_fn: Callable,
    input_tensor: torch.Tensor,
    target_ind: Optional[int] = None,
    additional_forward_args: Optional[Tuple[Any]] = None,
) -> torch.Tensor:
    r"""
    Computes gradients of the output with respect to inputs for an arbitrary forward function.

    Args:
        forward_fn: forward function. This can be for example model's forward function.
        input_tensor:      Input at which gradients are evaluated, will be passed to forward_fn.
        target_ind: Index of the target class for which gradients must be computed (classification only).
        additional_forward_args: Additional input arguments that forward function requires.
    """
    with torch.autograd.set_grad_enabled(True):
        outputs = _run_forward(
            forward_fn, input_tensor, target_ind, additional_forward_args
        )
        assert outputs[0].numel() == 1, (
            "Target not provided when necessary, cannot"
            " take gradient with respect to multiple outputs."
        )
        grads = torch.autograd.grad(torch.unbind(outputs), input_tensor)
    return grads[0]


def _forward_layer_eval(
    forward_fn: ModelType,
    inputs: torch.Tensor,
    layer: LayerType,
    target_ind: Optional[int] = None,
    additional_forward_args: Any = None,
    attribute_to_layer_input: bool = False,
) -> Tuple[torch.Tensor, torch.Tensor]:
    r"""
    A helper function that allows to set a hook on model's `layer`, run the forward
    pass and return layer result and optionally also the output of the forward function
    depending on whether we set `attribute_to_layer_input` to True or False.

    NOTE: To properly handle inplace operations, a clone of the layer output
    is stored. This structure inhibits execution of a backward hook on the last
    module for the layer output when computing the gradient with respect to
    the input, since we store an intermediate clone, as opposed to the true module output.
    """
    activations: torch.Tensor

    def pre_hook_wrapper(_):
        def forward_pre_hook(_, input_tensor):
            nonlocal activations
            activations = input_tensor[0]
            return activations.clone()

        return forward_pre_hook

    def hook_wrapper(_):
        def forward_hook(_, input_tensor, output):  # pylint: disable=unused-argument
            nonlocal activations
            activations = output
            return activations.clone()

        return forward_hook

    hook: Optional[RemovableHandle] = None
    try:
        # forward_hook is used to get the final activation from the forward function of the model
        # and pre_forward_hook when the activation is to be with relation to the target layer
        if attribute_to_layer_input:
            hook = layer.register_forward_pre_hook(pre_hook_wrapper(layer))
        else:
            hook = layer.register_forward_hook(hook_wrapper(layer))
        output = _run_forward(
            forward_fn,
            inputs,
            target=target_ind,
            additional_forward_args=additional_forward_args,
        )
        return activations, output
    finally:
        if hook is not None:
            hook.remove()


def compute_layer_gradients(
    model: ModelType,
    layer: LayerType,
    input_tensor: torch.Tensor,
    target_ind: Optional[int] = None,
    additional_forward_args: Optional[Any] = None,
    attribute_to_layer_input: bool = False,
) -> Tuple[torch.Tensor, torch.Tensor]:
    r"""
    Computes gradients of the output with respect to a given layer as well
    as the output evaluation of the layer for an arbitrary forward function
    and given input.

    NOTE: To properly handle inplace operations, a clone of the layer output
    is stored. This structure inhibits execution of a backward hook on the last
    module for the layer output when computing the gradient with respect to
    the input, since we store an intermediate clone, as
    opposed to the true module output.

    Args:

        model: forward function. This can be for example model's
                    forward function.
        layer:      Layer for which gradients / output will be evaluated.
        input_tensor:     Input at which gradients are evaluated, will be passed to forward_fn.
        target_ind: Index of the target class for which gradients must be computed (classification only).
        additional_forward_args: Additional input arguments that forward function requires.
        attribute_to_layer_input: Indicates whether to compute the attribution with respect to the layer input
                or output. If `attribute_to_layer_input` is set to True then the attributions will be computed
                with respect to layer input, otherwise it will be computed with respect to layer output.
                Default: False

    Returns:
        tuple[**gradients**, **layer_outputs**]:
        - **gradients**:
            Gradients of output with respect to target layer output.
        - **layer_outputs**:
            Target layer output for given input.
    """
    with torch.autograd.set_grad_enabled(True):
        layer_outputs, output = _forward_layer_eval(
            model,
            input_tensor,
            layer,
            target_ind=target_ind,
            additional_forward_args=additional_forward_args,
            attribute_to_layer_input=attribute_to_layer_input,
        )
        assert output[0].numel() == 1, (
            "Target not provided when necessary, cannot"
            " take gradient with respect to multiple outputs."
        )

        grads = torch.autograd.grad(torch.unbind(output), layer_outputs)

    return grads[0], layer_outputs  # type: ignore
