"""File with functions to manipulate figures."""

import matplotlib
import numpy as np
import torch
from matplotlib.backends.backend_agg import FigureCanvasAgg as FigureCanvas

from autoxai.explainer.base_explainer import CVExplainer


def convert_figure_to_numpy(figure: matplotlib.pyplot.Figure) -> np.ndarray:
    """Convert matplotlib figure to numpy array.

    Args:
        figure: Matplotlib figure.

    Returns:
        Numpy array.
    """
    canvas = FigureCanvas(figure)
    canvas.draw()
    buf = canvas.buffer_rgba()
    image = np.asarray(buf)
    return image


def create_figure(
    attribute_pickle_path: str, transformed_img: torch.Tensor
) -> matplotlib.pyplot.Figure:
    """Create figure from pickled attributed and original image.

    Args:
        attribute_pickle_path: Path to pickled attributes.
        transformed_img: Original image tensor.

    Returns:
        Visualization figure.
    """
    attributions = torch.Tensor(np.load(attribute_pickle_path))
    return CVExplainer.visualize(attributions, transformed_img)
