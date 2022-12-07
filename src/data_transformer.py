"""Data transforming classes."""
from abc import ABC

import torch
from torchvision import transforms


class DataTransformer(ABC):  # pylint: disable = (too-few-public-methods)
    """Abstract class performing input data transformations."""

    def preprocess(
        self, data: torch.Tensor  # pylint: disable=unused-argument
    ) -> torch.Tensor:
        """Preprocess image.

        Args:
            data: Tensor to be preprocessed.

        Returns:
            Preprocessed image tensor.
        """


class ExplainerCVTransformer(
    DataTransformer
):  # pylint: disable = (too-few-public-methods)
    """Computer vision tasks input data transformation class."""

    def preprocess(
        self, data: torch.Tensor  # pylint: disable=unused-argument
    ) -> torch.Tensor:
        """Preprocess image.

        Args:
            data: Image tensor to be preprocessed.

        Returns:
            Preprocessed image tensor.
        """
        resized_img = self.resize(data=data)
        centered_img = self.center(data=resized_img)
        return self.transform(data=centered_img)

    def transform(self, data: torch.Tensor) -> torch.Tensor:
        """Transform image to desired

        Args:
            data: Image tensor to transform.

        Returns:
            Resized image and normalized image tensor.
        """
        transform_normalize = transforms.Normalize(
            mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]
        )
        input_data = transform_normalize(data)
        input_data = input_data.unsqueeze(0)
        return input_data

    def resize(self, data: torch.Tensor) -> torch.Tensor:
        """Resize image.

        Args:
            data: Image tensor to be resized.

        Returns:
            Resized image tensor.
        """
        return transforms.Resize(256)(data)

    def center(self, data: torch.Tensor) -> torch.Tensor:
        """Center image.

        Args:
            data: Image tensor to be centered.

        Returns:
            Centered image tensor.
        """
        return transforms.CenterCrop(224)(data)
