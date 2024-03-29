"""Abstract Explainer class."""
from abc import ABC, abstractmethod
from typing import Optional, TypeVar

import torch

from foxai.types import AttributionsType, ModelType


class Explainer(ABC):
    """Abstract explainer class."""

    # TODO: add support in explainer for multiple input models
    @abstractmethod
    def calculate_features(
        self,
        model: ModelType,
        input_data: torch.Tensor,  # TODO: add more generic way of passing model inputs # pylint: disable = (fixme)
        pred_label_idx: Optional[int] = None,
        **kwargs,
    ) -> AttributionsType:
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


CVExplainerT = TypeVar("CVExplainerT", bound=Explainer)
"""CVExplainer subclass type."""
