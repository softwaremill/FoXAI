# pylint: disable = missing-class-docstring

import pytest
import torch
from matplotlib.pyplot import Figure

from foxai.visualizer import single_channel_visualization


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
