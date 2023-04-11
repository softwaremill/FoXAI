# pylint: disable = missing-class-docstring

from typing import List

import numpy as np
import pytest
import torch
from matplotlib.pyplot import Figure

from foxai.visualizer import get_heatmap_bbox, single_channel_visualization


def test_single_channel_visualization_selected_channel_should_raise_exception() -> None:
    """Test if function raises ValueError when passing selected_channel
    not in range of color dimension of attributions matrix."""
    selected_channel: int = 4
    attributions: torch.Tensor = torch.zeros((3, 2, 2), dtype=torch.float)
    transformed_img: torch.Tensor = torch.zeros((3, 2, 2), dtype=torch.float)
    with pytest.raises(ValueError):
        _ = single_channel_visualization(
            attributions=attributions,
            transformed_img=transformed_img,
            selected_channel=selected_channel,
        )


def test_single_channel_visualization_selected_channel_should_pass() -> None:
    """Test if function returns Figure object when passing selected_channel
    in range of color dimension of attributions matrix."""
    selected_channel: int = 0
    attributions: torch.Tensor = torch.zeros((3, 2, 2), dtype=torch.float)
    transformed_img: torch.Tensor = torch.zeros((3, 2, 2), dtype=torch.float)
    result = single_channel_visualization(
        attributions=attributions,
        transformed_img=transformed_img,
        selected_channel=selected_channel,
    )

    assert isinstance(result, Figure)


def test_get_heatmap_bbox_should_return_valid_mask():
    heatmap: np.ndarray = np.ones((5, 5), dtype=np.uint8)
    bbox: List[int] = [1, 1, 4, 4]
    expected = np.asarray(
        [
            [0, 0, 0, 0, 0],
            [0, 1, 1, 1, 0],
            [0, 1, 1, 1, 0],
            [0, 1, 1, 1, 0],
            [0, 0, 0, 0, 0],
        ],
        dtype=np.uint8,
    )

    result = get_heatmap_bbox(
        heatmap=heatmap,
        bbox=bbox,
    )

    np.testing.assert_almost_equal(expected, result)
