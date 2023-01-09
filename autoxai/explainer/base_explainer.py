"""Abstract Explainer class."""
import logging
from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import List, Optional

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
                method=viz.ImageVisualizationMethod.heat_map,
                sign=viz.VisualizeSign.positive,
                title="Positive attributes",
            )
        )
    else:
        log().info(msg="No positive attributes in the explained model.")

    if np.any(attributions_np < 0):
        explanation_methods.append(
            ExplanationMethods(
                method=viz.ImageVisualizationMethod.heat_map,
                sign=viz.VisualizeSign.negative,
                title="Negative attributes",
            )
        )
    else:
        log().info(msg="No negative attributes in the explained model.")

    if np.any(attributions_np != 0):
        explanation_methods.append(
            ExplanationMethods(
                method=viz.ImageVisualizationMethod.heat_map,
                sign=viz.VisualizeSign.all,
                title="All attributes",
            )
        )

    return explanation_methods


class CVExplainer(ABC):
    """Abstract explainer class."""

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
        cls, attributions: torch.Tensor, transformed_img: torch.Tensor
    ) -> matplotlib.pyplot.Figure:
        """Create image with calculated features.

        Args:
            attributions: Features.
            transformed_img: Image.

        Returns:
            Image with paired figures: original image and features heatmap.
        """
        # for single color, e.g. MNIST data copy one colour channel 3 times to simulate RGB
        if len(attributions.shape) == 4 and attributions.shape[1] == 1:
            attributions = attributions.expand(
                1, 3, attributions.shape[2], attributions.shape[3]
            )
        elif len(attributions.shape) == 3 and attributions.shape[0] == 1:
            attributions = attributions.expand(
                3, attributions.shape[1], attributions.shape[2]
            )
        if len(transformed_img.shape) == 4 and transformed_img.shape[1] == 1:
            transformed_img = transformed_img.expand(
                1, 3, transformed_img.shape[2], transformed_img.shape[3]
            )
        elif len(transformed_img.shape) == 3 and transformed_img.shape[0] == 1:
            transformed_img = transformed_img.expand(
                3, transformed_img.shape[1], transformed_img.shape[2]
            )

        # change dimension from (C x H x W) to (H x W x C)
        # where C is colour dimension, H and W are height and width dimensions
        attributions_np: np.ndarray = attributions.squeeze().detach().cpu().numpy()
        transformed_img_np: np.ndarray = (
            transformed_img.squeeze().detach().cpu().numpy()
        )
        if len(attributions.shape) >= 3:
            attributions_np = np.transpose(attributions_np, (1, 2, 0))
        if len(transformed_img.shape) >= 3:
            transformed_img_np = np.transpose(transformed_img_np, (1, 2, 0))

        explanation_methods: List[ExplanationMethods] = determine_visualization_methods(
            attributions_np=attributions_np
        )

        figure, _ = viz.visualize_image_attr_multiple(
            attr=attributions_np,
            original_image=convert_float_to_uint8(array=transformed_img_np),
            methods=[explanation.method.name for explanation in explanation_methods],
            signs=[explanation.sign.name for explanation in explanation_methods],
            titles=[explanation.title for explanation in explanation_methods],
            show_colorbar=True,
            use_pyplot=False,
        )
        return figure
