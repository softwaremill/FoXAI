"""File with XRAI algorithm explainer classes.

Paper: https://arxiv.org/abs/1906.02825
Based on https://github.com/PAIR-code/saliency/blob/master/saliency/core/xrai.py.
"""
from __future__ import absolute_import, division, print_function

import logging

import numpy as np
import torch
from captum._utils.typing import TargetType
from skimage import segmentation
from skimage.morphology import dilation, disk
from skimage.transform import resize

from foxai.array_utils import validate_result
from foxai.explainer.base_explainer import Explainer
from foxai.explainer.computer_vision.algorithm.integrated_gradients import (
    IntegratedGradientsCVExplainer,
)
from foxai.types import AttributionsType, ModelType

_logger = logging.getLogger(__name__)

_FELZENSZWALB_SCALE_VALUES = [50, 100, 150, 250, 500, 1200]
_FELZENSZWALB_SIGMA_VALUES = [0.8]
_FELZENSZWALB_IM_RESIZE = (224, 224)
_FELZENSZWALB_IM_VALUE_RANGE = [-1.0, 1.0]
_FELZENSZWALB_MIN_SEGMENT_SIZE = 150


def _normalize_image(im, value_range, resize_shape=None):
    """Normalize an image by resizing it and rescaling its values.

    Args:
        im: Input image.
        value_range: [min_value, max_value]
        resize_shape: New image shape. Defaults to None.

    Returns:
        Resized and rescaled image.
    """
    im_max = np.max(im)
    im_min = np.min(im)
    im = (im - im_min) / (im_max - im_min)
    im = im * (value_range[1] - value_range[0]) + value_range[0]
    if resize_shape is not None:
        im = resize(
            im,
            resize_shape,
            order=3,
            mode="constant",
            preserve_range=True,
            anti_aliasing=True,
        )
    return im


def _get_segments_felzenszwalb(im, resize_image=True, scale_range=None, dilation_rad=5):
    """Compute image segments based on Felzenszwalb's algorithm.

    Efficient graph-based image segmentation, Felzenszwalb, P.F.
    and Huttenlocher, D.P. International Journal of Computer Vision, 2004

    Args:
    im: Input image.
    resize_image: If True, the image is resized to 224,224 for the segmentation
                    purposes. The resulting segments are rescaled back to match
                    the original image size. It is done for consistency w.r.t.
                    segmentation parameter range. Defaults to True.
    scale_range:  Range of image values to use for segmentation algorithm.
                    Segmentation algorithm is sensitive to the input image
                    values, therefore we need to be consistent with the range
                    for all images. If None is passed, the range is scaled to
                    [-1.0, 1.0]. Defaults to None.
    dilation_rad: Sets how much each segment is dilated to include edges,
                    larger values cause more blobby segments, smaller values
                    get sharper areas. Defaults to 5.
    Returns:
        masks: A list of boolean masks as np.ndarrays if size HxW for im size of
                HxWxC.
    """

    # TODO (tolgab) Set this to default float range of 0.0 - 1.0 and tune
    # parameters for that
    if scale_range is None:
        scale_range = _FELZENSZWALB_IM_VALUE_RANGE
    # Normalize image value range and size
    original_shape = im.shape[:2]
    # TODO (tolgab) This resize is unnecessary with more intelligent param range
    # selection
    if resize_image:
        im = _normalize_image(im, scale_range, _FELZENSZWALB_IM_RESIZE)
    else:
        im = _normalize_image(im, scale_range)
    segs = []
    for scale in _FELZENSZWALB_SCALE_VALUES:
        for sigma in _FELZENSZWALB_SIGMA_VALUES:
            seg = segmentation.felzenszwalb(
                im, scale=scale, sigma=sigma, min_size=_FELZENSZWALB_MIN_SEGMENT_SIZE
            )
        if resize_image:
            seg = resize(
                seg,
                original_shape,
                order=0,
                preserve_range=True,
                mode="constant",
                anti_aliasing=False,
            ).astype(int)
        segs.append(seg)
    masks = _unpack_segs_to_masks(segs)
    if dilation_rad:
        footprint = disk(dilation_rad)
        masks = [dilation(mask, footprint=footprint) for mask in masks]
    return masks


def _attr_aggregation_max(attr, axis=-1):
    return attr.max(axis=axis)


def _gain_density(mask1, attr, mask2=None):
    # Compute the attr density over mask1. If mask2 is specified, compute density
    # for mask1 \ mask2
    if mask2 is None:
        added_mask = mask1
    else:
        added_mask = _get_diff_mask(mask1, mask2)
    if not np.any(added_mask):
        return -np.inf
    else:
        return attr[added_mask].mean()


def _get_diff_mask(add_mask, base_mask):
    return np.logical_and(add_mask, np.logical_not(base_mask))


def _get_diff_cnt(add_mask, base_mask):
    return np.sum(_get_diff_mask(add_mask, base_mask))


def _unpack_segs_to_masks(segs):
    masks = []
    for seg in segs:
        for l in range(seg.min(), seg.max() + 1):
            masks.append(seg == l)
    return masks


class XRAIParameters(object):
    """Dictionary of parameters to specify how to XRAI and return outputs."""

    def __init__(
        self,
        steps=100,
        area_threshold=1.0,
        return_ig_attributions=False,
        return_xrai_segments=False,
        flatten_xrai_segments=True,
        algorithm="full",
    ):
        # TODO(tolgab) add return_ig_for_every_step functionality

        # Number of steps to use for calculating the Integrated Gradients
        # attribution. The higher the number of steps the higher is the precision
        # but lower the performance. (see also XRAIOutput.error).
        self.steps = steps
        # The fraction of the image area that XRAI should calculate the segments
        # for. All segments that exceed that threshold will be merged into a single
        # segment. The parameter is used to accelerate the XRAI computation if the
        # caller is only interested in the top fraction of segments, e.g. 20%. The
        # value should be in the [0.0, 1.0] range, where 1.0 means that all segments
        # should be returned (slowest). Fast algorithm ignores this setting.
        self.area_threshold = area_threshold
        # self.return_baseline_predictions = return_baseline_predictions
        # If set to True, the XRAI output returns Integrated Gradients attributions
        # for every baseline. (see XraiOutput.ig_attribution)
        self.return_ig_attributions = return_ig_attributions
        # If set to True the XRAI output returns XRAI segments in the order of their
        # importance. This parameter works in conjunction with the
        # flatten_xrai_sements parameter. (see also XraiOutput.segments)
        self.return_xrai_segments = return_xrai_segments
        # If set to True, the XRAI segments are returned as an integer array with
        # the same dimensions as the input (excluding color channels). The elements
        # of the array are set to values from the [1,N] range, where 1 is the most
        # important segment and N is the least important segment. If
        # flatten_xrai_sements is set to False, the segments are returned as a
        # boolean array, where the first dimension has size N. The [0, ...] mask is
        # the most important and the [N-1, ...] mask is the least important. This
        # parameter has an effect only if return_xrai_segments is set to True.
        self.flatten_xrai_segments = flatten_xrai_segments
        # Specifies a flavor of the XRAI algorithm. full - executes slower but more
        # precise XRAI algorithm. fast - executes faster but less precise XRAI
        # algorithm.
        self.algorithm = algorithm
        # EXPERIMENTAL - Contains experimental parameters that may change in future.
        self.experimental_params = {"min_pixel_diff": 50}


class XRAIOutput(object):
    """Dictionary of outputs from a single run of XRAI.GetMaskWithDetails."""

    def __init__(self, attribution_mask):
        # The saliency mask of individual input features. For an [HxWx3] image, the
        # returned attribution is [H,W,1] float32 array. Where HxW are the
        # dimensions of the image.
        self.attribution_mask = attribution_mask
        # Baselines that were used for IG calculation. The shape is [B,H,W,C], where
        # B is the number of baselines, HxWxC are the image dimensions.
        self.baselines = None
        # TODO(tolgab) add return IG error functionality from XRAI API
        # The average error of the IG attributions as a percentage. The error can be
        # decreased by increasing the number of steps (see XraiParameters.steps).
        # self.error = None
        # TODO(tolgab) add return baseline predictions functionality from XRAI API
        # Predictions for the baselines that were used for the calculation of IG
        # attributions. The value is set only when
        # XraiParameters.return_baseline_predictions is set to True.
        # self.baseline_predictions = None
        # IG attributions for individual baselines. The value is set only when
        # XraiParameters.ig_attributions is set to True. For the dimensions of the
        # output see XraiParameters.return_ig_for_every _step.
        self.ig_attribution = None
        # The result of the XRAI segmentation. The value is set only when
        # XraiParameters.return_xrai_segments is set to True. For the dimensions of
        # the output see XraiParameters.flatten_xrai_segments.
        self.segments = None


class XRAI:
    """A CoreSaliency class that computes saliency masks using the XRAI method."""

    def __init__(self, forward_func):
        # Initialize integrated gradients.
        self.forward_func = forward_func

    def _get_integrated_gradients(
        self,
        im,
        pred_label_idx,
        call_model_function,
        call_model_args,
        baselines,
        steps,
        batch_size,  # pylint: disable = (unused-argument)
    ):
        """Takes mean of attributions from all baselines."""
        grads = []
        integrated_gradients = IntegratedGradientsCVExplainer()
        for baseline in baselines:
            grads.append(
                integrated_gradients.calculate_features(
                    model=call_model_function,
                    input_data=im,
                    baselines=baseline,
                    pred_label_idx=pred_label_idx,
                    n_steps=steps,
                    additional_forward_args=call_model_args,
                )
            )

        return grads

    def _make_baselines(self, x_value, x_baselines):
        # If baseline is not provided default to im min and max values
        x_value_np = x_value.detach().cpu().numpy()
        if x_baselines is None:
            x_baselines = []
            x_baselines.append(np.min(x_value_np) * np.ones_like(x_value_np))
            x_baselines.append(np.max(x_value_np) * np.ones_like(x_value_np))
        else:
            for baseline in x_baselines:
                if baseline.shape != x_value.shape:
                    raise ValueError(
                        f"Baseline size {baseline.shape} does not match input size {x_value.shape}"
                    )
        return x_baselines

    def _predict(self, x):
        raise NotImplementedError

    def get_mask(
        self,
        x_value,
        call_model_function,
        pred_label_idx,
        call_model_args=None,
        baselines=None,
        segments=None,
        base_attribution=None,
        batch_size=1,
        extra_parameters=None,
    ):
        """Applies XRAI method on an input image and returns the result saliency heatmap.


        Args:
            x_value: Input ndarray.
            call_model_function: A function that interfaces with a model to return
            specific data in a dictionary when given an input and other arguments.
            Expected function signature:
            - call_model_function(x_value_batch,
                                    call_model_args=None,
                                    expected_keys=None):
                x_value_batch - Input for the model, given as a batch (i.e.
                dimension 0 is the batch dimension, dimensions 1 through n
                represent a single input).
                call_model_args - Other arguments used to call and run the model.
                expected_keys - List of keys that are expected in the output. For
                this method (XRAI), the expected keys are
                INPUT_OUTPUT_GRADIENTS - Gradients of the output being
                    explained (the logit/softmax value) with respect to the input.
                    Shape should be the same shape as x_value_batch.
            call_model_args: The arguments that will be passed to the call model
            function, for every call of the model.
            baselines: a list of baselines to use for calculating
            Integrated Gradients attribution. Every baseline in
            the list should have the same dimensions as the
            input. If the value is not set then the algorithm
            will make the best effort to select default
            baselines. Defaults to None.
            segments: the list of precalculated image segments that should
            be passed to XRAI. Each element of the list is an
            [N,M] boolean array, where NxM are the image
            dimensions. Each elemeent on the list contains exactly the
            mask that corresponds to one segment. If the value is None,
            Felzenszwalb's segmentation algorithm will be applied.
            Defaults to None.
            base_attribution: an optional pre-calculated base attribution that XRAI
            should use. The shape of the parameter should match
            the shape of `x_value`. If the value is None, the
            method calculates Integrated Gradients attribution and
            uses it.
            batch_size: Maximum number of x inputs (steps along the integration
            path) that are passed to call_model_function as a batch.
            extra_parameters: an XRAIParameters object that specifies
            additional parameters for the XRAI saliency
            method. If it is None, an XRAIParameters object
            will be created with default parameters. See
            XRAIParameters for more details.

        Raises:
            ValueError: If algorithm type is unknown (not full or fast).
                        If the shape of `base_attribution` dosn't match the shape of
                        `x_value`.
                        If the shape of INPUT_OUTPUT_GRADIENTS doesn't match the
                        shape of x_value_batch.

        Returns:
            np.ndarray: A numpy array that contains the saliency heatmap.


        TODO(tolgab) Add output_selector functionality from XRAI API doc
        """
        results = self.get_mask_with_details(
            x_value,
            call_model_function,
            call_model_args=call_model_args,
            pred_label_idx=pred_label_idx,
            baselines=baselines,
            segments=segments,
            base_attribution=base_attribution,
            batch_size=batch_size,
            extra_parameters=extra_parameters,
        )
        return results.attribution_mask

    def get_mask_with_details(
        self,
        x_value,
        call_model_function,
        pred_label_idx,
        call_model_args=None,
        baselines=None,
        segments=None,
        base_attribution=None,
        batch_size=1,
        extra_parameters=None,
    ):
        """Applies XRAI method on an input image and returns detailed information.


        Args:
            x_value: Input ndarray.
            call_model_function: A function that interfaces with a model to return
            specific data in a dictionary when given an input and other arguments.
            Expected function signature:
            - call_model_function(x_value_batch,
                                    call_model_args=None,
                                    expected_keys=None):
                x_value_batch - Input for the model, given as a batch (i.e.
                dimension 0 is the batch dimension, dimensions 1 through n
                represent a single input).
                call_model_args - Other arguments used to call and run the model.
                expected_keys - List of keys that are expected in the output. For
                this method (XRAI), the expected keys are
                INPUT_OUTPUT_GRADIENTS - Gradients of the output being
                    explained (the logit/softmax value) with respect to the input.
                    Shape should be the same shape as x_value_batch.
            call_model_args: The arguments that will be passed to the call model
            function, for every call of the model.
            baselines: a list of baselines to use for calculating
            Integrated Gradients attribution. Every baseline in
            the list should have the same dimensions as the
            input. If the value is not set then the algorithm
            will make the best effort to select default
            baselines. Defaults to None.
            segments: the list of precalculated image segments that should
            be passed to XRAI. Each element of the list is an
            [N,M] boolean array, where NxM are the image
            dimensions. Each elemeent on the list contains exactly the
            mask that corresponds to one segment. If the value is None,
            Felzenszwalb's segmentation algorithm will be applied.
            Defaults to None.
            base_attribution: an optional pre-calculated base attribution that XRAI
            should use. The shape of the parameter should match
            the shape of `x_value`. If the value is None, the
            method calculates Integrated Gradients attribution and
            uses it.
            batch_size: Maximum number of x inputs (steps along the integration
            path) that are passed to call_model_function as a batch.
            extra_parameters: an XRAIParameters object that specifies
            additional parameters for the XRAI saliency
            method. If it is None, an XRAIParameters object
            will be created with default parameters. See
            XRAIParameters for more details.

        Raises:
            ValueError: If algorithm type is unknown (not full or fast).
                        If the shape of `base_attribution` dosn't match the shape of
                        `x_value`.
                        If the shape of INPUT_OUTPUT_GRADIENTS doesn't match the
                        shape of x_value_batch.

        Returns:
            XRAIOutput: an object that contains the output of the XRAI algorithm.

        TODO(tolgab) Add output_selector functionality from XRAI API doc
        """
        if extra_parameters is None:
            extra_parameters = XRAIParameters()

        # Check the shape of base_attribution.
        if base_attribution is not None:
            if not isinstance(base_attribution, np.ndarray):
                base_attribution = np.array(base_attribution)
            if base_attribution.shape != x_value.shape:
                raise ValueError(
                    "The base attribution shape should be the same as the shape of "
                    + f"`x_value`. Expected {x_value.shape}, got {base_attribution.shape}"
                )

        # Calculate IG attribution if not provided by the caller.
        _logger.info("Computing IG...")
        if baselines is not None:
            x_baselines = self._make_baselines(x_value, baselines)
            baselines = [torch.tensor(x).to(x_value.device) for x in x_baselines]
        else:
            baselines = torch.rand((2,) + x_value.shape, device=x_value.device)

        attrs = self._get_integrated_gradients(
            x_value,
            pred_label_idx,
            call_model_function,
            call_model_args=call_model_args,
            baselines=baselines,
            steps=extra_parameters.steps,
            batch_size=batch_size,
        )
        # Merge attributions from different baselines.
        attr = np.mean([a.detach().cpu().numpy() for a in attrs], axis=0)

        # Merge attribution channels for XRAI input
        if len(attr.shape) > 2:
            attr = _attr_aggregation_max(attr, axis=1)

        attr = attr[0]
        x_value = (
            x_value.reshape((x_value.shape[2], x_value.shape[3], x_value.shape[1]))
            .detach()
            .cpu()
            .numpy()
        )

        if baselines is not None:
            baselines = [
                b.reshape((b.shape[2], b.shape[3], b.shape[1]))
                if isinstance(b, torch.Tensor)
                else b
                for b in baselines
            ]

        _logger.info("Done with IG. Computing XRAI...")
        if segments is not None:
            segs = segments
        else:
            segs = _get_segments_felzenszwalb(x_value, dilation_rad=5)

        attr_map, attr_data = self._xrai(
            attr=attr,
            segs=segs,
            area_perc_th=extra_parameters.area_threshold,
            min_pixel_diff=extra_parameters.experimental_params["min_pixel_diff"],
            gain_fun=_gain_density,
            integer_segments=extra_parameters.flatten_xrai_segments,
        )

        results = XRAIOutput(attr_map)
        results.baselines = baselines
        if extra_parameters.return_xrai_segments:
            results.segments = attr_data
        if extra_parameters.return_ig_attributions:
            results.ig_attribution = attrs
        return results

    @staticmethod
    def _xrai(
        attr,
        segs,
        gain_fun=_gain_density,
        area_perc_th=1.0,
        min_pixel_diff=50,
        integer_segments=True,
    ):
        """Run XRAI saliency given attributions and segments.

        Args:
            attr: Source attributions for XRAI. XRAI attributions will be same size
            as the input attr.
            segs: Input segments as a list of boolean masks. XRAI uses these to
            compute attribution sums.
            gain_fun: The function that computes XRAI area attribution from source
            attributions. Defaults to _gain_density, which calculates the
            density of attributions in a mask.
            area_perc_th: The saliency map is computed to cover area_perc_th of
            the image. Lower values will run faster, but produce
            uncomputed areas in the image that will be filled to
            satisfy completeness. Defaults to 1.0.
            min_pixel_diff: Do not consider masks that have difference less than
            this number compared to the current mask. Set it to 1
            to remove masks that completely overlap with the
            current mask.
            integer_segments: See XRAIParameters. Defaults to True.

        Returns:
            tuple: saliency heatmap and list of masks or an integer image with
            area ranks depending on the parameter integer_segments.
        """
        output_attr = -np.inf * np.ones(shape=attr.shape, dtype=float)

        n_masks = len(segs)
        current_area_perc = 0.0
        current_mask = np.zeros(attr.shape, dtype=bool)

        masks_trace = []
        remaining_masks = dict(enumerate(segs))

        added_masks_cnt = 1
        # While the mask area is less than area_th and remaining_masks is not empty
        while current_area_perc <= area_perc_th:
            best_gain = -np.inf
            best_key = None
            remove_key_queue = []
            for mask_key in remaining_masks:
                mask = remaining_masks[mask_key]
                # If mask does not add more than min_pixel_diff to current mask, remove
                mask_pixel_diff = _get_diff_cnt(mask, current_mask)
                if mask_pixel_diff < min_pixel_diff:
                    remove_key_queue.append(mask_key)
                    if _logger.isEnabledFor(logging.DEBUG):
                        _logger.debug(
                            f"Skipping mask with pixel difference: {mask_pixel_diff}"
                        )
                    continue
                gain = gain_fun(mask, attr, mask2=current_mask)
                if gain > best_gain:
                    best_gain = gain
                    best_key = mask_key
            for key in remove_key_queue:
                del remaining_masks[key]
            if not remaining_masks:
                break
            added_mask = remaining_masks[best_key]
            mask_diff = _get_diff_mask(added_mask, current_mask)
            masks_trace.append((mask_diff, best_gain))

            current_mask = np.logical_or(current_mask, added_mask)
            current_area_perc = np.mean(current_mask)
            output_attr[mask_diff] = best_gain
            del remaining_masks[best_key]  # delete used key
            if _logger.isEnabledFor(logging.DEBUG):
                current_attr_sum = np.sum(attr[current_mask])
                _logger.debug(
                    f"{added_masks_cnt} of {n_masks} masks added, attr_sum: "
                    + f"{current_attr_sum}, area: {current_area_perc}/{area_perc_th}, "
                    + f"{len(remaining_masks)} remaining masks"
                )
            added_masks_cnt += 1

        uncomputed_mask = output_attr == -np.inf
        # Assign the uncomputed areas a value such that sum is same as ig
        output_attr[uncomputed_mask] = gain_fun(uncomputed_mask, attr)
        masks_trace = [v[0] for v in sorted(masks_trace, key=lambda x: -x[1])]
        if np.any(uncomputed_mask):
            masks_trace.append(uncomputed_mask)
        if integer_segments:
            attr_ranks = np.zeros(shape=attr.shape, dtype=int)
            for i, mask in enumerate(masks_trace):
                attr_ranks[mask] = i + 1
            return output_attr, attr_ranks
        else:
            return output_attr, masks_trace


class XRAICVExplainer(Explainer):
    """XRAI algorithm explainer."""

    def calculate_features(
        self,
        model: ModelType,
        input_data: torch.Tensor,
        pred_label_idx: TargetType = None,
        **kwargs,  # pylint: disable = (unused-argument)
    ) -> AttributionsType:
        """Generate model's attributes with XRAI algorithm explainer.

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
        xrai_explainer = XRAI(forward_func=model)

        attributions = xrai_explainer.get_mask(
            input_data,
            pred_label_idx=pred_label_idx,
            call_model_function=model,
            **kwargs,
        )
        attributions = torch.tensor(attributions).unsqueeze(0)
        validate_result(attributions=attributions)
        return attributions
