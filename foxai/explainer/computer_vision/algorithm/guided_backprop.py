"""File with Guided Backpropagation algorithm explainer class.

Based on https://github.com/pytorch/captum/blob/master/captum/attr/_core/guided_backprop_deconvnet.py.
"""

from abc import abstractmethod
from typing import Any

import torch
from captum._utils.typing import TargetType
from captum.attr import GuidedBackprop

from foxai.array_utils import validate_result
from foxai.explainer.base_explainer import Explainer
from foxai.explainer.computer_vision.model_utils import modify_modules
from foxai.types import AttributionsType, ModelType


class BaseGuidedBackpropCVExplainer(Explainer):
    """Base Guided Backpropagation algorithm explainer."""

    @abstractmethod
    def create_explainer(
        self,
        model: ModelType,
        **kwargs,
    ) -> GuidedBackprop:
        """Create explainer object.

        Returns:
            Explainer object.
        """

    def calculate_features(
        self,
        model: ModelType,
        input_data: torch.Tensor,
        pred_label_idx: TargetType = None,
        additional_forward_args: Any = None,
        **kwargs,
    ) -> AttributionsType:
        """Generate model's attributes with Guided Backpropagation algorithm explainer.

        Args:
            model: The forward function of the model or any
                modification of it.
            input_data: Input for which
                attributions are computed. If forward_func takes a single
                tensor as input, a single input tensor should be provided.
                If forward_func takes multiple tensors as input, a tuple
                of the input tensors should be provided. It is assumed
                that for all given input tensors, dimension 0 corresponds
                to the number of examples (aka batch size), and if
                multiple input tensors are provided, the examples must
                be aligned appropriately.
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

        Returns:
            The guided backprop gradients with respect to each
            input feature. Attributions will always
            be the same size as the provided inputs, with each value
            providing the attribution of the corresponding input index.
            If a single tensor is provided as inputs, a single tensor is
            returned. If a tuple is provided for inputs, a tuple of
            corresponding sized tensors is returned.

        Raises:
            RuntimeError: if attribution has shape (0).
        """
        guided_backprop = self.create_explainer(model=model)
        attributions: AttributionsType = guided_backprop.attribute(
            input_data,
            target=pred_label_idx,
            additional_forward_args=additional_forward_args,
        )
        validate_result(attributions=attributions)
        return attributions


class GuidedBackpropCVExplainer(BaseGuidedBackpropCVExplainer):
    """Guided Backpropagation algorithm explainer."""

    def create_explainer(
        self,
        model: ModelType,
        **kwargs,
    ) -> GuidedBackprop:
        """Create explainer object.

        Returns:
            Explainer object.
        """

        return GuidedBackprop(model=modify_modules(model))
