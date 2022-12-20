"""Abstract Explainer class."""
from abc import ABC, abstractmethod

import matplotlib
import numpy as np
import torch
from captum.attr import visualization as viz
from torch import fx


class CVExplainer(ABC):
    """Abstract explainer class."""

    @abstractmethod
    def calculate_features(
        self,
        model: fx.GraphModule,
        input_data: torch.Tensor,
        pred_label_idx: int,
        **kwargs,
    ) -> torch.Tensor:
        """Calculate features of given explainer.

        Args:
            model: Any DNN model You want to use.
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

        figure, _ = viz.visualize_image_attr_multiple(
            attr=attributions_np,
            original_image=transformed_img_np,
            methods=["original_image", "heat_map", "heat_map", "heat_map"],
            signs=["all", "positive", "negative", "all"],
            titles=[
                "Original image",
                "Positive attributes",
                "Negative attributes",
                "All attributes",
            ],
            show_colorbar=True,
            use_pyplot=False,
        )

        return figure
