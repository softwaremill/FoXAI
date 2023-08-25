"""File with Saliency algorithm explainer classes.

Based on https://github.com/pytorch/captum/blob/master/captum/attr/_core/saliency.py.
"""

from typing import Any, Optional

import torch

from foxai.array_utils import validate_result
from foxai.explainer.base_explainer import Explainer
from foxai.explainer.computer_vision.algorithm.gradient_utils import compute_gradients
from foxai.types import AttributionsType, ModelType


class SaliencyCVExplainer(Explainer):
    """Saliency algorithm explainer."""

    def calculate_features(  # type: ignore[override]
        self,
        model: ModelType,
        input_data: torch.Tensor,
        pred_label_idx: Optional[int] = None,
        abs_value: bool = True,
        additional_forward_args: Any = None,
        **kwargs,  # pylint: disable = (unused-argument)
    ) -> AttributionsType:
        """Generate model's attributes with Saliency algorithm explainer.

        Args:
            model: The forward function of the model or any
                modification of it.
            input_data: Input for which saliency
                is computed. If forward_func takes a single tensor
                as input, a single input tensor should be provided.
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
            abs_value: Returns absolute value of gradients if set
                to True, otherwise returns the (signed) gradients if
                False.
                Default: True
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

        Returns:
            The gradients with respect to each input feature.
            Attributions will always be
            the same size as the provided inputs, with each value
            providing the attribution of the corresponding input index.
            If a single tensor is provided as inputs, a single tensor is
            returned. If a tuple is provided for inputs, a tuple of
            corresponding sized tensors is returned.

        Raises:
            RuntimeError: if attribution has shape (0).
        """
        grad_required = input_data.requires_grad
        input_data.requires_grad_()

        gradients = compute_gradients(
            model, input_data, pred_label_idx, additional_forward_args
        )
        if abs_value:
            attributions = torch.abs(gradients)
        else:
            attributions = gradients

        if not grad_required:
            input_data.requires_grad_(False)

        validate_result(attributions=attributions)
        return attributions
