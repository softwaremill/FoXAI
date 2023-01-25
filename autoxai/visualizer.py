from typing import Tuple

import matplotlib
import numpy as np
import torch

from autoxai.array_utils import (
    convert_float_to_uint8,
    normalize_attributes,
    resize_attributes,
    retain_only_positive,
    transpose_array,
)


def generate_figure(
    attributions: np.ndarray,
    transformed_img: np.ndarray,
    title: str = "",
    figsize: Tuple[int, int] = (8, 8),
    alpha: float = 0.5,
) -> matplotlib.pyplot.Figure:
    """Create figure from image and heatmap.

    Args:
        attributions: Heatmap.
        transformed_img: Image in shape (H x W x C).
        title: Title of the figure. Defaults to "".
        figsize: Tuple with size of figure. Defaults to (8, 8).
        alpha: Opacity level. Defaults to 0.5,

    Returns:
        Heatmap of single channel applied on original image.
    """
    figure = matplotlib.figure.Figure(figsize=figsize)
    axis = figure.subplots()
    axis.imshow(transformed_img)
    heatmap_plot = axis.imshow(
        attributions, cmap=matplotlib.cm.jet, vmin=0, vmax=1, alpha=alpha
    )

    figure.colorbar(heatmap_plot, label="Pixel relevance")
    axis.get_xaxis().set_visible(False)
    axis.get_yaxis().set_visible(False)
    axis.set_title(title)

    return figure


def _preprocess_img_and_attributes(
    attributes_matrix: np.ndarray,
    transformed_img_np: np.ndarray,
    only_positive_attr: bool,
) -> Tuple[np.ndarray, np.ndarray]:
    single_channel_attributes: np.ndarray = normalize_attributes(
        attributes=attributes_matrix,
    )

    if only_positive_attr:
        single_channel_attributes = retain_only_positive(
            array=single_channel_attributes
        )

    resized_attributes: np.ndarray = resize_attributes(
        attributes=single_channel_attributes,
        dest_height=transformed_img_np.shape[1],
        dest_width=transformed_img_np.shape[2],
    )

    # standardize attributes to uint8 type and back-scale them to range 0-1
    grayscale_attributes = convert_float_to_uint8(resized_attributes) / 255

    # transpoze image from (C x H x W) shape to (H x W x C) to matplotlib imshow
    normalized_transformed_img = transpose_array(
        convert_float_to_uint8(transformed_img_np)
    )
    return grayscale_attributes, normalized_transformed_img


def mean_channels_visualization(
    attributions: torch.Tensor,
    transformed_img: torch.Tensor,
    title: str = "",
    figsize: Tuple[int, int] = (8, 8),
    alpha: float = 0.5,
    only_positive_attr: bool = True,
) -> matplotlib.pyplot.Figure:
    """Create image with calculated heatmap.

    Args:
        attributions: Features.
        transformed_img: Image in shape (C x H x W) or (H x W).
        title: Title of the figure. Defaults to "".
        figsize: Tuple with size of figure. Defaults to (8, 8).
        alpha: Opacity level. Defaults to 0.5,
        only_positive_attr: Whether to display only positive or all attributes.
            Defaults to True.

    Returns:
        Heatmap of mean channel values applied on original image.
    """
    attributes_matrix: np.ndarray = attributions.detach().cpu().numpy()
    transformed_img_np: np.ndarray = transformed_img.detach().cpu().numpy()

    grayscale_attributes, normalized_transformed_img = _preprocess_img_and_attributes(
        attributes_matrix=attributes_matrix,
        transformed_img_np=transformed_img_np,
        only_positive_attr=only_positive_attr,
    )

    return generate_figure(
        attributions=grayscale_attributes,
        transformed_img=normalized_transformed_img,
        title=title,
        figsize=figsize,
        alpha=alpha,
    )


def single_channel_visualization(
    attributions: torch.Tensor,
    transformed_img: torch.Tensor,
    selected_channel: int,
    title: str = "",
    figsize: Tuple[int, int] = (8, 8),
    alpha: float = 0.5,
    only_positive_attr: bool = True,
) -> matplotlib.pyplot.Figure:
    """Create image with calculated heatmap.

    Args:
        attributions: Features.
        transformed_img: Image in shape (C x H x W) or (H x W).
        selected_channel: Single color channel to visualize.
        title: Title of the figure. Defaults to "".
        figsize: Tuple with size of figure. Defaults to (8, 8).
        alpha: Opacity level. Defaults to 0.5,
        only_positive_attr: Whether to display only positive or all attributes.
            Defaults to True.

    Returns:
        Heatmap of single channel applied on original image.
    """
    assert 0 <= selected_channel <= attributions.shape[0]

    attributes_matrix: np.ndarray = attributions.detach().cpu().numpy()
    transformed_img_np: np.ndarray = transformed_img.detach().cpu().numpy()
    attributes_matrix = attributes_matrix[selected_channel]

    grayscale_attributes, normalized_transformed_img = _preprocess_img_and_attributes(
        attributes_matrix=attributes_matrix,
        transformed_img_np=transformed_img_np,
        only_positive_attr=only_positive_attr,
    )

    single_channel_attributes: np.ndarray = normalize_attributes(
        attributes=attributes_matrix,
    )

    if only_positive_attr:
        single_channel_attributes = retain_only_positive(
            array=single_channel_attributes
        )

    resized_attributes: np.ndarray = resize_attributes(
        attributes=single_channel_attributes,
        dest_height=transformed_img_np.shape[1],
        dest_width=transformed_img_np.shape[2],
    )

    # standardize attributes to uint8 type and back-scale them to range 0-1
    grayscale_attributes = convert_float_to_uint8(resized_attributes) / 255

    # transpoze image from (C x H x W) shape to (H x W x C) to matplotlib imshow
    normalized_transformed_img = transpose_array(
        convert_float_to_uint8(transformed_img_np)
    )

    return generate_figure(
        attributions=grayscale_attributes,
        transformed_img=normalized_transformed_img,
        title=title,
        figsize=figsize,
        alpha=alpha,
    )
