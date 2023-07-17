from typing import List, Optional, Tuple

import cv2
import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import torch
import torchvision
from matplotlib.pyplot import Axes, Figure

from foxai.array_utils import (
    convert_standardized_float_to_uint8,
    normalize_attributes,
    resize_attributes,
    retain_only_positive,
    standardize_array,
    transpose_array,
)
from foxai.explainer.computer_vision.object_detection.types import ObjectDetectionOutput


def draw_image(
    image: torch.Tensor,
    title: str = "",
    ax: Optional[Axes] = None,
    figsize: Tuple[int, int] = (8, 8),
) -> Figure:
    """Draw picture.

    Args:
        image: Image in shape (C x H x W).
        title: Title for the plot.
        ax: Axis class used to draw image. If not provided, new figure will be created.
        figsize: Tuple with size of figure. Defaults to (8, 8).

    Returns:
        Axes with the plot rendered.
    """
    if ax is None:
        figure = Figure(figsize=figsize)
        ax = figure.subplots()

    # change image shape from (C X H X W) to (H X W X C) where C stands for colour, X is height and W is width dimension
    sample_np = image.permute((1, 2, 0)).detach().cpu().numpy().astype(float)

    ax.set_title(title)
    # disable visualizing X and Y axes
    ax.get_xaxis().set_visible(False)
    ax.get_yaxis().set_visible(False)

    # convert image from float to uint8 and display it
    ax.imshow(
        convert_standardized_float_to_uint8(standardize_array(sample_np.astype(float)))
    )
    return ax.get_figure()


def generate_figure(
    attributions: np.ndarray,
    transformed_img: np.ndarray,
    title: str = "",
    figsize: Tuple[int, int] = (8, 8),
    alpha: float = 0.5,
    ax: Optional[Axes] = None,
) -> Figure:
    """Create figure from image and heatmap.

    Args:
        attributions: Heatmap.
        transformed_img: Image in shape (H x W x C).
        title: Title of the figure. Defaults to "".
        figsize: Tuple with size of figure. Defaults to (8, 8).
        alpha: Opacity level. Defaults to 0.5,
        ax: Axis class used to draw image. If not provided, new figure will be created.

    Returns:
        Heatmap of single channel applied on original image.
    """
    if ax is None:
        ax = Figure(figsize=figsize).subplots()
    ax.imshow(transformed_img)
    heatmap_plot = ax.imshow(
        attributions, cmap=matplotlib.cm.jet, vmin=0, vmax=1, alpha=alpha
    )

    figure = ax.get_figure()
    ax.figure.colorbar(heatmap_plot, label="Pixel relevance")
    ax.get_xaxis().set_visible(False)
    ax.get_yaxis().set_visible(False)
    ax.set_title(title)

    return figure


def _preprocess_img_and_attributes(
    attributes_matrix: np.ndarray,
    transformed_img_np: np.ndarray,
    only_positive_attr: bool,
) -> Tuple[np.ndarray, np.ndarray]:
    """Pre-process image and attributes matrices.

    Pre-processing consists of:
        - squash color dimensions by mean over all colors in attributes matrix
        - optional retension of only positive attributes
        - resizing attributes heatmap to match the size of an image
        - standardization to value range [0-1]
        - transpoze image matrix from (C x H x W) to (H x W x C)

    Args:
        attributions: Features.
        transformed_img: Image in shape (C x H x W) or (H x W).
        only_positive_attr: Whether to display only positive or all attributes.
            Defaults to True.

    Returns:
        Tuple of pre-processed attributes and image matrices.
    """
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
    grayscale_attributes = standardize_array(resized_attributes)

    # standardize image
    standardized_img = standardize_array(transformed_img_np.astype(np.dtype(float)))

    # transpoze image from (C x H x W) shape to (H x W x C) to matplotlib imshow
    normalized_transformed_img = transpose_array(
        convert_standardized_float_to_uint8(standardized_img),
    )
    return grayscale_attributes, normalized_transformed_img


def mean_channels_visualization(
    attributions: torch.Tensor,
    transformed_img: torch.Tensor,
    title: str = "",
    figsize: Tuple[int, int] = (8, 8),
    alpha: float = 0.5,
    only_positive_attr: bool = True,
    ax: Optional[Axes] = None,
) -> Figure:
    """Create image with calculated heatmap.

    Args:
        attributions: Features.
        transformed_img: Image in shape (C x H x W) or (H x W).
        title: Title of the figure. Defaults to "".
        figsize: Tuple with size of figure. Defaults to (8, 8).
        alpha: Opacity level. Defaults to 0.5,
        only_positive_attr: Whether to display only positive or all attributes.
            Defaults to True.
        ax: Axis class used to draw image. If not provided, new figure will be created.

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
        ax=ax,
    )


def single_channel_visualization(
    attributions: torch.Tensor,
    transformed_img: torch.Tensor,
    selected_channel: int,
    title: str = "",
    figsize: Tuple[int, int] = (8, 8),
    alpha: float = 0.5,
    only_positive_attr: bool = True,
    ax: Optional[Axes] = None,
) -> Figure:
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
        ax: Axis class used to draw image. If not provided, new figure will be created.

    Returns:
        Heatmap of single channel applied on original image.

    Raises:
        ValueError: if selected channel is negative number or exceed dimension
            of color channels of attributes.
    """
    condition: bool = 0 <= selected_channel < attributions.shape[0]
    if not condition:
        raise ValueError(
            f"The selected channel exceeds color dimension. Selected channel: {selected_channel}",
        )

    attributes_matrix: np.ndarray = attributions.detach().cpu().numpy()
    transformed_img_np: np.ndarray = transformed_img.detach().cpu().numpy()
    attributes_matrix = attributes_matrix[selected_channel]

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
        ax=ax,
    )


def visualize_metric(
    importance_lst: List[np.ndarray],
    metric_result: float,
    metric_type: str = "Deletion",
):
    """
    Visualize graph for Insertion or deletion metric based on which area under the curve is caluclated.
    """
    plt.ylim((0, 1))
    plt.xlim((0, len(importance_lst)))
    plt.plot(np.arange(len(importance_lst)), importance_lst)
    plt.title(f"{metric_type}: {metric_result}")
    plt.show()


def preprocess_object_detection_image(input_image: torch.Tensor) -> np.ndarray:
    """Process input image to display.

    Args:
        input_image: Original image of type float in range [0-1].

    Returns:
        Converted image as np.ndarray in (C x H x W).
    """
    return (
        input_image.squeeze(0)
        .mul(255)
        .clamp_(0, 255)
        .permute(1, 2, 0)
        .detach()
        .cpu()
        .numpy()
    )


def get_heatmap_bbox(
    heatmap: np.ndarray,
    bbox: List[int],
    mask_value: int = 0,
) -> np.ndarray:
    """_summary_

    Code based on https://github.com/pooya-mohammadi/deep_utils/blob/main/deep_utils/utils/box_utils/boxes.py.

    Args:
        heatmap: Heatmap to visualize.
        bbox: Bounding box of detection.
        mask_value: Masking value . Defaults to 0.

    Returns:
        Numpy array with heatmap only present in area of given bounding box.
    """
    # fill the outer area of the selected box
    mask = np.ones_like(heatmap, dtype=np.uint8) * mask_value
    mask[bbox[0] : bbox[2], bbox[1] : bbox[3]] = 1
    masked_heatmap = cv2.multiply(heatmap, mask)
    return masked_heatmap


def draw_heatmap_in_bbox(
    bbox: List[int],
    heatmap: torch.Tensor,
    img: np.ndarray,
) -> np.ndarray:
    """Draw heatmap in bounding box on image.

    Args:
        bbox: List of coordinates for bounding box.
        heatmap: Heatmap to display.
        img: Original image.

    Returns:
        Image with displayed heatmap in bounding box area.
    """
    heatmap_np = preprocess_object_detection_image(heatmap).astype(np.uint8)
    heatmap_np = cv2.applyColorMap(heatmap_np, cv2.COLORMAP_JET)

    masked_heatmap = get_heatmap_bbox(heatmap=heatmap_np, bbox=bbox).astype(np.float32)

    img = cv2.add(img, masked_heatmap)
    img = img / img.max()
    img = (img * 255).astype(np.uint8)
    return img


def concat_images(images: List[np.ndarray]) -> np.ndarray:
    """Concatenate images into one.

    Args:
        images: List of images to merge.

    Returns:
        Final image.
    """
    w, h = images[0].shape[:2]
    width = w
    height = h * len(images)
    base_img = np.zeros((width, height, 3), dtype=np.uint8)
    for i, img in enumerate(images):
        base_img[:, h * i : h * (i + 1), ...] = img

    return base_img


def object_detection_visualization(
    detections: ObjectDetectionOutput,
    input_image: torch.Tensor,
) -> np.ndarray:
    """Create array with detection heatmaps.

    Args:
        detections: Object detection data class.
        input_image: Image in shape (C x H x W) or (H x W).

    Returns:
        Array of series of images with heatmap displayed on detection bounding boxes.
    """
    masks = detections.saliency_maps
    boxes = [pred.bbox for pred in detections.predictions]
    class_names = [pred.class_name for pred in detections.predictions]
    img_to_display = preprocess_object_detection_image(input_image)
    img_to_display = img_to_display[..., ::-1]  # convert to bgr
    images = [img_to_display]

    for i, mask in enumerate(masks):
        res_img = img_to_display.copy()
        bbox, cls_name = boxes[i], class_names[i]
        bbox = [int(val) for val in bbox]
        res_img = draw_heatmap_in_bbox(bbox, mask, res_img)

        # convert to (C x H x W)
        res_img_tensor = torch.tensor(res_img).transpose(0, 2).transpose(1, 2)
        bbox = [bbox[1], bbox[0], bbox[3], bbox[2]]
        res_img_tensor = torchvision.utils.draw_bounding_boxes(
            image=res_img_tensor,
            boxes=torch.tensor([bbox]),
            labels=[cls_name],
        )
        # convert to (H x W x C)
        res_img = res_img_tensor.transpose(1, 2).transpose(0, 2).numpy()
        images.append(res_img)

    final_image = concat_images(images)
    return final_image
