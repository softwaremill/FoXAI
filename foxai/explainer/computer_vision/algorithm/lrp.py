"""File with LRP algorithm explainer classes.

Based on https://github.com/pytorch/captum/blob/master/captum/attr/_core/lrp.py
and https://github.com/pytorch/captum/blob/master/captum/attr/_core/layer/layer_lrp.py.
"""

from abc import abstractmethod
from contextlib import contextmanager
from typing import Any, Dict, Generator, List, Optional, Tuple

import torch
from torch import nn
from torch.nn import Module
from torch.utils.hooks import RemovableHandle

from foxai.array_utils import validate_result
from foxai.explainer.base_explainer import Explainer
from foxai.explainer.computer_vision.algorithm.gradient_utils import (
    _run_forward,
    compute_gradients,
    require_grad,
)
from foxai.explainer.computer_vision.algorithm.lrp_rules import (
    EpsilonRule,
    GammaRule,
    PropagationRule,
)
from foxai.explainer.computer_vision.model_utils import (
    get_last_conv_model_layer,
    modify_modules,
)
from foxai.types import AttributionsType, LayerType, ModelType


class BaseLRPCVExplainer(Explainer):
    """Base LRP algorithm explainer."""

    def __init__(self) -> None:
        super().__init__()
        self.removable_handles: List[RemovableHandle] = []
        self.layer_to_rule: Dict[Module, PropagationRule] = {}
        self.layer_to_initial_input: Dict[Module, torch.Tensor] = {}

    @abstractmethod
    def get_relevances(
        self,
        model: Module,
        gradients: torch.Tensor,
        **kwargs,
    ) -> torch.Tensor:
        pass

    def calculate_features(  # type: ignore[override]
        self,
        model: ModelType,
        input_data: torch.Tensor,
        pred_label_idx: Optional[int] = None,
        additional_forward_args: Any = None,
        verbose: bool = False,
        **kwargs,
    ) -> AttributionsType:
        """Generate model's attributes with LRP algorithm explainer.

        Args:
            model: The forward function of the model or any
                modification of it.
            input_data: Input for which relevance is
                propagated. If forward_func takes a single
                tensor as input, a single input tensor should be provided.
            pred_label_idx: Output indices for
                which gradients are computed (for classification cases,
                this is usually the target class).
                If the network returns a scalar value per example,
                no target index is necessary.

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
            verbose: Indicates whether information on application
                of rules is printed during propagation.
                Default: False

        Returns:
            Features matrix.

        Raises:
            RuntimeError: if attribution has shape (0)."""
        layers = list(m for m in model.modules() if len(list(m.children())) == 0)
        self.add_rules(modify_modules(model))

        with require_grad(input_data) as input_tensor:
            with self.safe_model_modify(model=model) as safe_model:
                # 1. Forward pass: Change weights of layers according to selected rules.
                output = self._apply_propagation_rules(
                    input_tensor,
                    safe_model,
                    pred_label_idx,
                    additional_forward_args,
                    layers,
                )
                # 2. Forward pass + backward pass: Register hooks to configure relevance
                # propagation and execute back-propagation.
                for layer in layers:
                    self.removable_handles.append(
                        layer.register_forward_pre_hook(
                            self._forward_pre_hook_activations
                        )
                    )
                    if type(layer) not in {nn.ReLU, nn.Dropout, nn.Tanh}:
                        self.removable_handles.append(
                            layer.register_forward_hook(self._forward_hook)
                        )
                        if verbose:
                            print(
                                f"Applied {self.layer_to_rule[layer]} on layer {layer}"
                            )
                gradients = compute_gradients(
                    model, input_data, pred_label_idx, additional_forward_args
                )
                relevances = self.get_relevances(model, gradients, **kwargs)
                # reshaping so that first dimension match relevances first dimension
                attributions = relevances * output.reshape(
                    (relevances.shape[0],) + (1,) * (relevances.dim() - 1)
                )

        validate_result(attributions=attributions)
        return attributions

    def _forward_pre_hook_activations(
        self, layer: Module, input_tensors: Tuple[torch.Tensor, ...]
    ) -> None:
        input_tensors[0].data = self.layer_to_initial_input[layer]

    def _forward_hook(
        self,
        layer: Module,
        input_tensors: Tuple[torch.Tensor, ...],
        outputs: torch.Tensor,
    ) -> torch.Tensor:
        self.removable_handles.append(
            input_tensors[0].register_hook(
                self.layer_to_rule[layer].create_backward_hook_input(
                    input_tensors[0].data
                )
            )
        )
        self.removable_handles.append(
            outputs.register_hook(
                self.layer_to_rule[layer].create_backward_hook_output(outputs.data)
            )
        )
        return outputs.clone()

    def add_rules(self, model: ModelType) -> None:
        """Add rules for the LRP explainer,
        according to https://arxiv.org/pdf/1910.09840.pdf.

        Args:
            model: The forward function of the model or any
                modification of it.

        Returns:
            Modified DNN object.
        """
        layers_number: int = len(list(model.modules()))
        for idx_layer, layer in enumerate(model.modules()):
            if idx_layer <= layers_number // 2:
                self.layer_to_rule[layer] = GammaRule()
            elif idx_layer != (layers_number - 1):
                self.layer_to_rule[layer] = EpsilonRule()
            else:
                self.layer_to_rule[layer] = EpsilonRule(epsilon=0)  # LRP-0

    @contextmanager
    def safe_model_modify(self, model: Module) -> Generator[Module, Module, None]:
        """
        Reverts model back to initial state after modification.
        """
        original_state_dict = model.state_dict()
        try:
            yield model
        finally:
            model.load_state_dict(original_state_dict)
            self._remove_handles()

    def _remove_handles(self) -> None:
        for removable_handle in self.removable_handles:
            removable_handle.remove()

    def _apply_propagation_rules(
        self,
        inputs: torch.Tensor,
        model: Module,
        target: Optional[int],
        additional_forward_args: Any,
        layers: List[Module],
    ) -> torch.Tensor:
        def modify_weights(layer: Module, inputs: Tuple[torch.Tensor, ...], _) -> None:
            self.layer_to_initial_input[layer] = inputs[0].data
            self.layer_to_rule[layer].modify_weights(layer, inputs[0])

        try:
            for layer in layers:
                self.removable_handles.append(
                    layer.register_forward_hook(modify_weights)
                )

            output = _run_forward(model, inputs, target, additional_forward_args)
        finally:
            self._remove_handles()
        return output


class LRPCVExplainer(BaseLRPCVExplainer):
    """LRP algorithm explainer."""

    def get_relevances(
        self,
        model: Module,
        gradients: torch.Tensor,
        **kwargs,
    ) -> torch.Tensor:
        return gradients


class LayerLRPCVExplainer(BaseLRPCVExplainer):
    """Layer LRP algorithm explainer."""

    def get_relevances(
        self,
        model: Module,
        gradients: torch.Tensor,
        attribute_to_layer_input: bool = False,
        layer: Optional[LayerType] = None,
        **kwargs,
    ) -> torch.Tensor:

        if layer is None:
            layer = get_last_conv_model_layer(model=model)

        if attribute_to_layer_input:
            relevance = self.layer_to_rule[layer].relevance_input
        else:
            relevance = self.layer_to_rule[layer].relevance_output

        if relevance is None:
            raise AttributeError(
                f"Relevance was not calculated properly for the given layer {layer}"
            )
        return relevance
