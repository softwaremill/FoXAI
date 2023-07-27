"""File with XRAI algorithm explainer classes.

Paper: https://arxiv.org/abs/1906.02825
Based on https://github.com/PAIR-code/saliency/blob/master/saliency/core/xrai.py.
"""
from __future__ import absolute_import, division, print_function

from typing import Any, Callable, Iterable, List, Optional, Tuple, Union

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


def _normalize_image(
    image_batch: np.ndarray,
    value_range: Tuple[float, float],
    resize_shape: Optional[Tuple[int, int]] = None,
) -> np.ndarray:
    """Normalize an image by resizing it and rescaling its values.

    Args:
        image: Input image of shape (B x H x W x C).
        value_range: [min_value, max_value]
        resize_shape: New image shape. Defaults to None.

    Returns:
        Resized and rescaled image.
    """
    image_max = np.max(image_batch)
    image_min = np.min(image_batch)
    image_batch = (image_batch - image_min) / (image_max - image_min)
    image_batch = image_batch * (value_range[1] - value_range[0]) + value_range[0]
    color_channels: int = image_batch.shape[-1]
    resized_image_list: List[np.ndarray] = []
    for image in image_batch:
        if resize_shape is not None:
            # resize can only process single image with shape (H x W x C)
            resized_image_list.append(
                resize(
                    image,
                    resize_shape + (color_channels,),
                    order=3,
                    mode="constant",
                    preserve_range=True,
                    anti_aliasing=True,
                )
            )
    # add artificial batch size dimension to each image
    batch_resized_image = np.vstack(
        [np.expand_dims(im, 0) for im in resized_image_list]
    )
    return batch_resized_image


def _get_segments_felzenszwalb(
    image_batch: np.ndarray,
    scale_range: Tuple[float, float],
    dilation_rad: int,
    image_resize: Optional[Tuple[int, int]],
    sigma_values: Iterable[float],
    scale_values: Iterable[int],
    min_segment_size: int,
) -> List[List[np.ndarray]]:
    """Compute image segments based on Felzenszwalb's algorithm.

    Efficient graph-based image segmentation, Felzenszwalb, P.F.
    and Huttenlocher, D.P. International Journal of Computer Vision, 2004

    Args:
        image: Input image in shape (B x H x W x C).
            resize_image:
        scale_range:  Range of image values to use for segmentation algorithm.
            Segmentation algorithm is sensitive to the input image
            values, therefore we need to be consistent with the range
            for all images. Defaults to [-1.0, 1.0].
        dilation_rad: Sets how much each segment is dilated to include edges,
            larger values cause more blobby segments, smaller values
            get sharper areas.
        image_resize: If not None, the image is resized to given shape HxW for
            the segmentation purposes. The resulting segments are rescaled back
            to match the original image size. It is done for consistency w.r.t.
            segmentation parameter range.
        sigma_values: Iterable of sigma values from Felzenszwalb algorithm. Sigma is
            width (standard deviation) of Gaussian kernel used in preprocessing.
        scale_values: Iterable of scale values from Felzenszwalb algorithm. It's free
            parameter. Higher means larger clusters.
        min_segment_size: Minimum segment size returned by Felzenszwalb algorithm.
    Returns:
        masks: A list of lists of boolean masks as np.ndarrays if size HxW for im size of
                HxWxC. First level of list has length of batch size.
    """
    # Normalize image value range and size
    original_shape = image_batch.shape[1:3]
    image_batch = _normalize_image(image_batch, scale_range, image_resize)

    if len(image_batch.shape) == 3:
        # add artificial batch size
        image_batch = np.expand_dims(image_batch, 0)

    batch_masks: List[List[np.ndarray]] = []
    for image in image_batch:
        segment_list: List[np.ndarray] = []
        for scale in scale_values:
            for sigma in sigma_values:
                segment = segmentation.felzenszwalb(
                    image,
                    scale=scale,
                    sigma=sigma,
                    min_size=min_segment_size,
                )
                if image_resize:
                    segment = resize(
                        segment,
                        original_shape,
                        order=0,
                        preserve_range=True,
                        mode="constant",
                        anti_aliasing=False,
                    ).astype(int)
                segment_list.append(segment)
        masks = _unpack_segs_to_masks(segment_list)
        if dilation_rad:
            footprint = disk(dilation_rad)
            masks = [dilation(mask, footprint=footprint) for mask in masks]

        batch_masks.append(masks)
    return batch_masks


def _gain_density(
    mask1: np.ndarray, attributes: np.ndarray, mask2: Optional[np.ndarray] = None
) -> float:
    """Calculate gain density as mean value in attributes segment.

    Args:
        mask1: Compute density for mask1.
        attributes: Attributes array.
        mask2: If mask2 is specified, compute density
            for mask1 \\ mask2. Defaults to None.

    Returns:
        Gain density score.
    """
    if mask2 is None:
        added_mask = mask1
    else:
        added_mask = _get_diff_mask(mask1, mask2)
    if not np.any(added_mask):
        return -np.inf
    else:
        return attributes[added_mask].mean()


def _get_diff_mask(array1: np.ndarray, array2: np.ndarray) -> np.ndarray:
    """Compute element-wise AND operation between two arrays.

    Args:
        array1: First array.
        array2: Second array.

    Returns:
        Array of booleans representing result of element-wise AND operation
            between arrays.
    """
    return np.logical_and(array1, np.logical_not(array2))


def _get_diff_cnt(array1: np.ndarray, array2: np.ndarray) -> int:
    """Compute sum of different elements in 2 arrays.

    Args:
        array1: First array.
        array2: Second array.

    Returns:
        Number of different elements between two arrays.
    """
    return np.sum(_get_diff_mask(array1, array2))


def _unpack_segs_to_masks(segment_list: List[np.ndarray]) -> List[np.ndarray]:
    """Create masks from segments.

    Args:
        segment_list: List of segments.

    Returns:
        List of masks.
    """
    masks: List[np.ndarray] = []
    for segment in segment_list:
        for index in range(segment.min(), segment.max() + 1):
            masks.append(segment == index)
    return masks


class XRAI:
    """A CoreSaliency class that computes saliency masks using the XRAI method."""

    def __init__(self, forward_func: ModelType):
        # Initialize integrated gradients.
        self.forward_func = forward_func

    def _get_integrated_gradients(
        self,
        input_data: torch.Tensor,
        pred_label_idx: TargetType,
        call_model_function: ModelType,
        call_model_args: Any,
        baselines: torch.Tensor,
        steps: int,
    ) -> List[AttributionsType]:
        """Get Integrated Gradients attributes.

        Args:
            input_data: Input data.
            pred_label_idx: Output indices for which gradients are
                computed (for classification cases, this is usually the
                target class).
            call_model_function: The forward function of the model or any
                modification of it
            call_model_args: Additional arguments to model forward pass.
            baselines: Baselines define the starting point from which integral
                is computed.
            steps: The number of steps used by the approximation
                method.

        Returns:
            List of attributions.
        """
        grads: List[AttributionsType] = []
        integrated_gradients = IntegratedGradientsCVExplainer()
        for baseline in baselines:
            grads.append(
                integrated_gradients.calculate_features(
                    model=call_model_function,
                    input_data=input_data,
                    baselines=baseline,
                    pred_label_idx=pred_label_idx,
                    n_steps=steps,
                    additional_forward_args=call_model_args,
                )
            )

        return grads

    def _validate_baselines(
        self,
        input_data: torch.Tensor,
        baselines: torch.Tensor,
    ) -> None:
        """Validate if baselines are correct.

        Args:
            input_data: Input data.
            baselines: Baselines define the starting point from which integral
                is computed.

        Raises:
            ValueError: If baselines and input data shapes differ.
        """
        for baseline in baselines:
            if baseline.shape != input_data.shape:
                raise ValueError(
                    f"Baseline size {baseline.shape} does not match input size {input_data.shape}"
                )

    def get_attribution_mask(
        self,
        input_data: torch.Tensor,
        pred_label_idx: TargetType,
        call_model_function: ModelType,
        call_model_args: Any = None,
        baselines: Optional[torch.Tensor] = None,
        segments: Optional[List[List[np.ndarray]]] = None,
        steps: int = 100,
        area_threshold: float = 1.0,
        flatten_xrai_segments: bool = True,
        min_pixel_diff: int = 50,
        dilation_rad: int = 5,
        scale_range: Tuple[float, float] = (-1.0, 1.0),
        image_resize: Optional[Tuple[int, int]] = (224, 224),
        sigma_values: Iterable[float] = (0.8,),
        scale_values: Iterable[int] = (
            50,
            100,
            150,
            250,
            500,
            1200,
        ),
        min_segment_size: int = 150,
    ) -> np.ndarray:
        """Applies XRAI method on an input data and returns the result saliency heatmap.

        Args:
            input_data: Input data in shape (B x C x H x W).
            call_model_function: A function that interfaces with a model to return
                specific data in a dictionary when given an input and other arguments.
                The forward function of the model or any modification of it.
            call_model_args: The arguments that will be passed to the call model
                function, for every call of the model.
            baselines: A list of baselines to use for calculating
                Integrated Gradients attribution. Every baseline in
                the list should have the same dimensions as the
                input. If the value is not set then the algorithm
                will make the best effort to select default
                baselines. Defaults to None.
            segments: The list of lists of precalculated image segments that should
                be passed to XRAI. Each element of the list is an
                [B,N,M] boolean array, where NxM are the image
                dimensions, B is the batch size. Each elemeent on the list contains
                exactly the mask that corresponds to one segment. If the value is None,
                Felzenszwalb's segmentation algorithm will be applied.
                Defaults to None.
            steps: Number of steps to use for calculating the Integrated Gradients
                attribution. The higher the number of steps the higher is the precision
                but lower the performance.
            area_threshold: The fraction of the image area that XRAI should calculate the
                segments for. All segments that exceed that threshold will be merged into
                a single segment. The parameter is used to accelerate the XRAI computation
                if the caller is only interested in the top fraction of segments, e.g. 20%.
                The value should be in the [0.0, 1.0] range, where 1.0 means that all
                segments should be returned.
            flatten_xrai_segments: If set to True, the XRAI segments are returned as an
                integer array with the same dimensions as the input (excluding color channels).
                The elements of the array are set to values from the [1,N] range, where 1
                is the most important segment and N is the least important segment. If
                flatten_xrai_sements is set to False, the segments are returned as a
                boolean array, where the first dimension has size N. The [0, ...] mask is
                the most important and the [N-1, ...] mask is the least important.
            min_pixel_diff: Do not consider masks that have difference less than
                this number compared to the current mask. Set it to 1 to remove masks
                that completely overlap with the current mask. Defaults to 50.
            image_resize: If not None, the image is resized to given shape HxW for
                the segmentation purposes. The resulting segments are rescaled back
                to match the original image size. It is done for consistency w.r.t.
                segmentation parameter range. Defaults to (224, 224).
            sigma_values: Iterable of sigma values from Felzenszwalb algorithm. Sigma is
                width (standard deviation) of Gaussian kernel used in preprocessing.
                Defaults to (0.8,).
            scale_values: Iterable of scale values from Felzenszwalb algorithm. It's free
                parameter. Higher means larger clusters.
                Defaults to (50, 100, 150, 250, 500, 1200,).
            min_segment_size: Minimum segment size returned by Felzenszwalb algorithm.
                Defaults to 150.

        Returns:
            A numpy array that contains the saliency heatmap.
        """
        if baselines is not None:
            self._validate_baselines(input_data, baselines)
            baselines = torch.vstack(
                [torch.tensor(x).to(input_data.device) for x in baselines]
            )
        else:
            baselines = torch.rand((2,) + input_data.shape, device=input_data.device)

        attrs = self._get_integrated_gradients(
            input_data=input_data,
            pred_label_idx=pred_label_idx,
            call_model_function=call_model_function,
            call_model_args=call_model_args,
            baselines=baselines,
            steps=steps,
        )
        # Merge attributions from different baselines.
        attr = np.mean([a.detach().cpu().numpy() for a in attrs], axis=0)

        # Merge attribution channels for XRAI input
        if len(attr.shape) > 2:
            attr = attr.max(axis=1)

        x_value_np: np.ndarray = (
            input_data.reshape(
                (
                    input_data.shape[0],
                    input_data.shape[2],
                    input_data.shape[3],
                    input_data.shape[1],
                )
            )
            .detach()
            .cpu()
            .numpy()
        )

        if segments is not None:
            segs = segments
        else:
            segs = _get_segments_felzenszwalb(
                image_batch=x_value_np,
                dilation_rad=dilation_rad,
                scale_range=scale_range,
                image_resize=image_resize,
                sigma_values=sigma_values,
                scale_values=scale_values,
                min_segment_size=min_segment_size,
            )

        attr_map, _ = self._xrai(
            attributes=attr,
            segment_list=segs,
            area_perc_th=area_threshold,
            min_pixel_diff=min_pixel_diff,
            gain_fun=_gain_density,
            integer_segments=flatten_xrai_segments,
        )

        return attr_map

    @staticmethod
    def _xrai(
        attributes: np.ndarray,
        segment_list: List[List[np.ndarray]],
        gain_fun: Callable[
            [np.ndarray, np.ndarray, Optional[np.ndarray]], float
        ] = _gain_density,
        area_perc_th: float = 1.0,
        min_pixel_diff: int = 50,
        integer_segments: bool = True,
    ) -> Tuple[np.ndarray, Union[np.ndarray, List[np.ndarray]]]:
        """Run XRAI saliency given attributions and segments.

        Args:
            attributes: Source attributions for XRAI. XRAI attributions will be same size
                as the input attr.
            segment_list: Input segments as a list of boolean masks. XRAI uses these to
                compute attribution sums. Fist level list is for batch dimension.
            gain_fun: The function that computes XRAI area attribution from source
                attributions. Defaults to _gain_density, which calculates the
                density of attributions in a mask.
            area_perc_th: The saliency map is computed to cover area_perc_th of
                the image. Lower values will run faster, but produce
                uncomputed areas in the image that will be filled to
            min_pixel_diff: Do not consider masks that have difference less than
                this number compared to the current mask. Set it to 1
                to remove masks that completely overlap with the
                current mask.
            integer_segments: If set to True, the XRAI segments are returned as an
                integer array with the same dimensions as the input (excluding color
                channels). The elements of the array are set to values from the
                [1,N] range, where 1 is the most important segment and N is the
                least important segment. If integer_segments is set to False,
                the segments are returned as a boolean array, where the first
                dimension has size N. The [0, ...] mask is the most important and the
                [N-1, ...] mask is the least important. Defaults to True.

        Returns:
            Saliency heatmap np.ndarray and masks or an integer image with
                area ranks depending on the parameter integer_segments.
        """
        batch_attr_maps: List[np.ndarray] = []
        batch_final_mask_trace: List[Union[np.ndarray, List[np.ndarray]]] = []
        for attribute_sample, segment in zip(attributes, segment_list):
            output_attr: np.ndarray = -np.inf * np.ones(
                shape=attribute_sample.shape, dtype=float
            )
            current_area_perc: float = 0.0
            current_mask = np.zeros(attribute_sample.shape, dtype=bool)

            masks_trace: List[Tuple[np.ndarray, float]] = []
            remaining_masks = dict(enumerate(segment))

            added_masks_cnt: int = 1
            # While the mask area is less than area_th and remaining_masks is not empty
            while current_area_perc <= area_perc_th:
                best_gain: float = -np.inf
                best_key: int
                remove_key_queue: List[int] = []
                for mask_key in remaining_masks:
                    mask: np.ndarray = remaining_masks[mask_key]
                    # If mask does not add more than min_pixel_diff to current mask, remove
                    mask_pixel_diff = _get_diff_cnt(mask, current_mask)
                    if mask_pixel_diff < min_pixel_diff:
                        remove_key_queue.append(mask_key)
                        continue
                    gain: float = gain_fun(mask, attribute_sample, current_mask)
                    if gain > best_gain:
                        best_gain = gain
                        best_key = mask_key
                for key in remove_key_queue:
                    del remaining_masks[key]

                # if dictionary is empty end processing
                if not remaining_masks:
                    break

                added_mask = remaining_masks[best_key]
                mask_diff = _get_diff_mask(added_mask, current_mask)
                masks_trace.append((mask_diff, best_gain))

                current_mask = np.logical_or(current_mask, added_mask)
                current_area_perc = np.mean(current_mask)
                output_attr[mask_diff] = best_gain
                del remaining_masks[best_key]  # delete used key
                added_masks_cnt += 1

            uncomputed_mask: np.ndarray = output_attr == -np.inf
            # Assign the uncomputed areas a value such that sum is same as ig
            output_attr[uncomputed_mask] = gain_fun(
                uncomputed_mask, attribute_sample, None
            )
            masks_trace_array_list = [
                v[0] for v in sorted(masks_trace, key=lambda x: -x[1])
            ]
            if np.any(uncomputed_mask):
                masks_trace_array_list.append(uncomputed_mask)

            final_mask_trace: List[np.ndarray] = masks_trace_array_list
            if integer_segments:
                integer_segment_mask = np.zeros(shape=attribute_sample.shape, dtype=int)
                for i, mask in enumerate(masks_trace_array_list):
                    integer_segment_mask[mask] = i + 1

                final_mask_trace = [integer_segment_mask]

            # add 2 artificial dimensions
            # first for artificial batch size which will be stacked
            # second for artificial 1D channel
            batch_attr_maps.append(output_attr[None, None, ...])
            batch_final_mask_trace.append(
                [im[None, None, ...] for im in final_mask_trace]
            )

        final_batch_attr_maps = np.vstack(batch_attr_maps)
        final_batch_final_mask_trace = np.vstack(batch_final_mask_trace)

        return final_batch_attr_maps, final_batch_final_mask_trace


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

        attributions_np = xrai_explainer.get_attribution_mask(
            input_data=input_data,
            pred_label_idx=pred_label_idx,
            call_model_function=model,
            **kwargs,
        )
        attributions: AttributionsType = torch.tensor(attributions_np)
        validate_result(attributions=attributions)
        return attributions
