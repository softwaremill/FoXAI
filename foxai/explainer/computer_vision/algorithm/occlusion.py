"""File with Occulusion algorithm explainer classes.

Based on https://github.com/pytorch/captum/blob/master/captum/attr/_core/occlusion.py.
"""

from typing import Any, List, Optional, Tuple, Union

import torch
from captum._utils.typing import BaselineType
from captum.attr import Occlusion

from foxai.array_utils import validate_result
from foxai.explainer.base_explainer import Explainer
from foxai.explainer.computer_vision.model_utils import preprocess_baselines
from foxai.types import AttributionsType, ModelType


class OcclusionCVExplainer(Explainer):
    """Occlusion algorithm explainer."""

    def calculate_features(
        self,
        model: ModelType,
        input_data: torch.Tensor,
        pred_label_idx: Optional[int] = None,
        sliding_window_shapes: Union[Tuple[int, ...], Tuple[Tuple[int, ...], ...]] = (
            1,
            1,
            1,
        ),
        strides: Optional[
            Union[int, Tuple[int, ...], Tuple[Union[int, Tuple[int, ...]], ...]]
        ] = None,
        baselines: Optional[BaselineType] = None,
        additional_forward_args: Any = None,
        perturbations_per_eval: int = 1,
        show_progress: bool = False,
        **kwargs,
    ) -> AttributionsType:
        """Generate model's attributes with Occlusion algorithm explainer.

        Args:
            model: The forward function of the model or any
                modification of it.
            input_data: Input for which occlusion
                attributions are computed. If forward_func takes a single
                tensor as input, a single input tensor should be provided.
            pred_label_idx: Output indices for
                which difference is computed (for classification cases,
                this is usually the target class).
                If the network returns a scalar value per example,
                no target index is necessary.
                Default: None
            sliding_window_shapes: Shape of patch
                (hyperrectangle) to occlude each input. For a single
                input tensor, this must be a tuple of length equal to the
                number of dimensions of the input tensor - 1, defining
                the dimensions of the patch. If the input tensor is 1-d,
                this should be an empty tuple.
                Default: (1, 1, 1)
            strides:
                This defines the step by which the occlusion hyperrectangle
                should be shifted by in each direction for each iteration.
                For a single tensor input, this can be either a single
                integer, which is used as the step size in each direction,
                or a tuple of integers matching the number of dimensions
                in the occlusion shape, defining the step size in the
                corresponding dimension.
                To ensure that all inputs are covered by at least one
                sliding window, the stride for any dimension must be
                <= the corresponding sliding window dimension if the
                sliding window dimension is less than the input
                dimension.
                If None is provided, a stride of 1 is used for each
                dimension of each input tensor.
                Default: None
            baselines:
                Baselines define reference value which replaces each
                feature when occluded.
                Baselines can be provided as:

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

                In the cases when `baselines` is not provided, we internally
                use zero scalar corresponding to each input tensor.
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
                For a tensor, the first dimension of the tensor must
                correspond to the number of examples. For all other types,
                the given argument is used for all forward evaluations.
                Note that attributions are not computed with respect
                to these arguments.
                Default: None
            perturbations_per_eval: Allows multiple occlusions
                to be included in one batch (one call to forward_fn).
                By default, perturbations_per_eval is 1, so each occlusion
                is processed individually.
                Each forward pass will contain a maximum of
                perturbations_per_eval * #examples samples.
                For DataParallel models, each batch is split among the
                available devices, so evaluations on each available
                device contain at most
                (perturbations_per_eval * #examples) / num_devices
                samples.
                Default: 1
            show_progress: Displays the progress of computation.
                It will try to use tqdm if available for advanced features
                (e.g. time estimation). Otherwise, it will fallback to
                a simple output of progress.
                Default: False

        Returns:
            The attributions with respect to each input feature.
            Attributions will always be
            the same size as the provided inputs, with each value
            providing the attribution of the corresponding input index.
            If a single tensor is provided as inputs, a single tensor is
            returned. If a tuple is provided for inputs, a tuple of
            corresponding sized tensors is returned.

        Raises:
            RuntimeError: if attribution has shape (0).
        """
        attributions: AttributionsType
        occlusion = Occlusion(model)

        attributions_list: List[AttributionsType] = []
        baselines_list, aggregate_attributes = preprocess_baselines(
            baselines=baselines,
            input_data_shape=input_data.shape,
        )

        for baseline in baselines_list:
            attributions = occlusion.attribute(
                input_data,
                strides=strides,
                target=pred_label_idx,
                sliding_window_shapes=sliding_window_shapes,
                baselines=baseline,
                additional_forward_args=additional_forward_args,
                perturbations_per_eval=perturbations_per_eval,
                show_progress=show_progress,
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
