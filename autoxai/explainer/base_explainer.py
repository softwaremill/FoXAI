"""Abstract Explainer class."""
import logging
from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import List, Optional, Tuple, TypeVar

import cv2
import matplotlib
import numpy as np
import torch
from captum.attr import visualization as viz

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


@dataclass
class ExplanationMethods:
    """Holder for the explainer attributes visualization method."""

    method: viz.ImageVisualizationMethod
    sign: viz.VisualizeSign
    title: str


def determine_visualization_methods(
    attributions_np: np.ndarray,
) -> List[ExplanationMethods]:
    """Determine which visualization methods to use,
    based on presents of positive and negative values
    in the explained image.

    Args:
        attributions_np: Attributes of the explained image.

    Returns:
        List of visualization methods to be used,
        for the given image attributes.
    """
    explanation_methods: List[ExplanationMethods] = [
        ExplanationMethods(
            method=viz.ImageVisualizationMethod.original_image,
            sign=viz.VisualizeSign.all,
            title="Original image",
        )
    ]

    # check whether we can explain model with positive and negative attributes
    if np.any(attributions_np > 0):
        explanation_methods.append(
            ExplanationMethods(
                method=viz.ImageVisualizationMethod.masked_image,
                sign=viz.VisualizeSign.positive,
                title="Positive attributes",
            )
        )
    else:
        log().info(msg="No positive attributes in the explained model.")

    if np.any(attributions_np < 0):
        explanation_methods.append(
            ExplanationMethods(
                method=viz.ImageVisualizationMethod.masked_image,
                sign=viz.VisualizeSign.negative,
                title="Negative attributes",
            )
        )
    else:
        log().info(msg="No negative attributes in the explained model.")

    if np.any(attributions_np != 0):
        explanation_methods.append(
            ExplanationMethods(
                method=viz.ImageVisualizationMethod.masked_image,
                sign=viz.VisualizeSign.positive,
                title="All attributes",
            )
        )

    return explanation_methods


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
        transformed_img_np: np.ndarray = transformed_img.detach().cpu().numpy()
        (w, h) = (transformed_img.shape[2], transformed_img.shape[1])
        value = attributions[0][0].numpy()
        value = cv2.resize(value, (w, h))
        value = torch.tensor(value.reshape(1, value.shape[0], value.shape[1]))

        grayscale_attributes = np.transpose(
            convert_float_to_uint8(value.detach().cpu().numpy()), (1, 2, 0)
        ).astype(np.uint8)
        # create figure from attributes and original image
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
