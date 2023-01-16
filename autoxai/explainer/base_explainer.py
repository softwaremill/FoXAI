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
    ) -> matplotlib.pyplot.Figure:
        """Create image with calculated features.

        Args:
            attributions: Features.
            transformed_img: Image.
            title: Title of the figure.
            figsize: Tuple with size of figure.
            alpha: Opacity level.

        Returns:
            Image with paired figures: original image and features heatmap.
        """
        if len(attributions.shape) == 4:
            # if we have attributes with shape (B x C x W x H)
            # where B is batch size, C is color, W is width and H is height dimension
            attributes_matrix = attributions[0][0].numpy()
        elif len(attributions.shape) == 3:
            # if we have attributes with shape (B x W x H)
            # where B is batch size, W is width and H is height dimension
            attributes_matrix = attributions[0].numpy()
        else:
            raise ValueError(f"Incorrect shape of attributions: {attributions.shape}")

        transformed_img_np: np.ndarray = transformed_img.detach().cpu().numpy()
        (width, height) = (transformed_img.shape[2], transformed_img.shape[1])

        resized_attributes = cv2.resize(attributes_matrix, (width, height))
        single_channel_attributes = resized_attributes.reshape(
            1,
            resized_attributes.shape[0],
            resized_attributes.shape[1],
        )

        grayscale_attributes = np.transpose(
            convert_float_to_uint8(single_channel_attributes), (1, 2, 0)
        ).astype(np.uint8)
        normalized_transformed_img = np.transpose(
            convert_float_to_uint8(transformed_img_np), (1, 2, 0)
        )

        figure = matplotlib.figure.Figure(figsize=figsize)
        axis = figure.subplots()
        axis.imshow(np.mean(normalized_transformed_img, axis=2), cmap="gray")
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
