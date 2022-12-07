"""Explainer manager class."""
from typing import Any

import torch

from src.data_transformer import DataTransformer
from src.model_utils import get_prediction


class ExplainerManager:  # pylint: disable = (too-few-public-methods)
    """Explainer manager class."""

    def predict(  # pylint: disable = (too-many-arguments)
        self,
        model: Any,
        data: torch.Tensor,
        transformer: DataTransformer,
    ) -> torch.Tensor:
        """Make explanations for all provided algorithms.

        Args:
            model: Explained model.
            data: Data to explain.
            transformer: Class to be applied to transform data.

        Returns:
            Predicted label tensor.
        """
        return get_prediction(model, transformer.preprocess(data=data))
