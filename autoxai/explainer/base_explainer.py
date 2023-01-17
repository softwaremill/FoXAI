"""Abstract Explainer class."""
import logging
from abc import ABC, abstractmethod
from typing import Optional, Tuple, TypeVar

import cv2
import matplotlib
import numpy as np
import torch

from autoxai.array_utils import convert_float_to_uint8
from autoxai.logger import create_logger

_LOGGER: Optional[logging.Logger] = None


def log() -> logging.Logger:
    """Get or create logger."""
    # pylint: disable = global-statement
    global _LOGGER
    if _LOGGER is None:
        _LOGGER = create_logger(__name__)
    return _LOGGER


class CVExplainer(ABC):
    """Abstract explainer class."""

    # TODO: add support in explainer for multiple input models
    @abstractmethod
    def calculate_features(
        self,
        model: torch.nn.Module,
        input_data: torch.Tensor,  # TODO: add more generic way of passing model inputs # pylint: disable = (fixme)
        pred_label_idx: int,
        **kwargs,
    ) -> torch.Tensor:
        """Calculate features of given explainer.

        Args:
            model: Neural network model You want to explain.
            input_data: Input image.
            pred_label_idx: Predicted label.

        Returns:
            Tensor of attributes.
        """

    @property
    def algorithm_name(self) -> str:
        """Get algorithm name.

        Returns:
            str: Name of algorithm.
        """
        return type(self).__name__

    @classmethod
    def visualize(
        cls,
        attributions: torch.Tensor,
        transformed_img: torch.Tensor,
        title: str = "",
        figsize: Tuple[int, int] = (8, 8),
        alpha: float = 0.5,
        only_positive_attr: bool = True,
    ) -> matplotlib.pyplot.Figure:
        """Create image with calculated features.

        Args:
            attributions: Features.
            transformed_img: Image in shape (C x H x W) or (H x W).
            title: Title of the figure. Defaults to "".
            figsize: Tuple with size of figure. Defaults to (8, 8).
            alpha: Opacity level. Defaults to 0.5,
            only_positive_attr: Whether to display only positive or all attributes.
                Defaults to True.

        Returns:
            Image with paired figures: original image and features heatmap.
        """
        attributes_matrix: np.ndarray = attributions.detach().cpu().numpy()
        transformed_img_np: np.ndarray = transformed_img.detach().cpu().numpy()

        if len(attributes_matrix.shape) == 3:
            # if we have attributes with shape (C x H x W)
            # where C is color, W is width and H is height dimension
            # calculate mean over attributes for all colors
            attributes_matrix = np.mean(attributes_matrix, axis=0)
        else:
            raise ValueError(
                f"Incorrect shape of attributions: {attributes_matrix.shape}"
            )

        # discard negative attributes
        if only_positive_attr:
            attributes_matrix[attributes_matrix < 0] = 0

        # resize attributes matrix to match input image
        single_channel_attributes: np.ndarray = np.array(
            cv2.resize(
                attributes_matrix,
                (transformed_img.shape[2], transformed_img.shape[1]),
            )
        )
        # standardize attributes to uint8 type
        grayscale_attributes = convert_float_to_uint8(single_channel_attributes)

        # transpoze image from (C x H x W) shape to (H x W x C) to matplotlib imshow
        normalized_transformed_img = np.transpose(
            convert_float_to_uint8(transformed_img_np), (1, 2, 0)
        )

        figure = matplotlib.figure.Figure(figsize=figsize)
        axis = figure.subplots()
        axis.imshow(normalized_transformed_img)
        heatmap_plot = axis.imshow(
            grayscale_attributes, cmap=matplotlib.cm.jet, vmin=0, vmax=255, alpha=alpha
        )

        figure.colorbar(heatmap_plot, label="Pixel relevance")
        axis.get_xaxis().set_visible(False)
        axis.get_yaxis().set_visible(False)
        axis.set_title(title)

        return figure


CVExplainerT = TypeVar("CVExplainerT", bound=CVExplainer)
"""CVExplainer subclass type."""
