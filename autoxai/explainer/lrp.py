"""File with LRP algorithm explainer classes.

Based on https://github.com/pytorch/captum/blob/master/captum/attr/_core/lrp.py
and https://github.com/pytorch/captum/blob/master/captum/attr/_core/layer/layer_lrp.py.
"""

from abc import abstractmethod
from typing import Any, Optional, Union

import torch
from captum._utils.typing import TargetType
from captum.attr import LRP, LayerLRP
from captum.attr._utils.lrp_rules import EpsilonRule, GammaRule

from autoxai.array_utils import validate_result
from autoxai.explainer.base_explainer import CVExplainer
from autoxai.explainer.model_utils import get_last_conv_model_layer, modify_modules


class BaseLRPCVExplainer(CVExplainer):
    """Base LRP algorithm explainer."""

    @abstractmethod
    def create_explainer(
        self,
        model: torch.nn.Module,
        **kwargs,
    ) -> Union[LRP, LayerLRP]:
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
        verbose: bool = False,
        **kwargs,
    ) -> torch.Tensor:
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
                forward_func in order, following the arguments in inputs.
                Note that attributions are not computed with respect
                to these arguments.
                Default: None
            attribute_to_layer_input: Indicates whether to
                compute the attribution with respect to the layer input
                or output. If `attribute_to_layer_input` is set to True
                then the attributions will be computed with respect to
                layer input, otherwise it will be computed with respect
                to layer output.
            verbose: Indicates whether information on application
                of rules is printed during propagation.
                Default: False

        Returns:
            Features matrix.

        Raises:
            RuntimeError: if attribution has shape (0).
        """
        layer: Optional[torch.nn.Module] = kwargs.get("layer", None)

        lrp = self.create_explainer(model=model, layer=layer)

        if isinstance(lrp, LRP):
            attributions = lrp.attribute(
                input_data,
                target=pred_label_idx,
                additional_forward_args=additional_forward_args,
                return_convergence_delta=False,
                verbose=verbose,
            )
        else:
            attributions = lrp.attribute(
                input_data,
                target=pred_label_idx,
                additional_forward_args=additional_forward_args,
                return_convergence_delta=False,
                attribute_to_layer_input=attribute_to_layer_input,
                verbose=verbose,
            )
        validate_result(attributions=attributions)
        return attributions

    def add_rules(self, model: torch.nn.Module) -> torch.nn.Module:
        """Add rules for the LRP explainer,
        according to https://arxiv.org/pdf/1910.09840.pdf.

        Args:
            model: The forward function of the model or any
                modification of it.

        Returns:
            Modified DNN object.
        """
        layers_number: int = len(list(model.modules()))
        for idx_layer, module in enumerate(model.modules()):
            if idx_layer <= layers_number // 2:
                setattr(module, "rule", GammaRule())
            elif idx_layer != (layers_number - 1):
                setattr(module, "rule", EpsilonRule())
            else:
                setattr(module, "rule", EpsilonRule(epsilon=0))  # LRP-0

        return model


class LRPCVExplainer(BaseLRPCVExplainer):
    """LRP algorithm explainer."""

    def create_explainer(
        self,
        model: torch.nn.Module,
        **kwargs,
    ) -> Union[LRP, LayerLRP]:
        """Create explainer object.

        Args:
            model: The forward function of the model or any
                modification of it.

        Returns:
            Explainer object.
        """
        model = self.add_rules(modify_modules(model))

        return LRP(model=model)


class LayerLRPCVExplainer(BaseLRPCVExplainer):
    """Layer LRP algorithm explainer."""

    def create_explainer(
        self,
        model: torch.nn.Module,
        layer: Optional[torch.nn.Module] = None,
        **kwargs,
    ) -> Union[LRP, LayerLRP]:
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

        model = self.add_rules(modify_modules(model))

        return LayerLRP(model=model, layer=layer)
