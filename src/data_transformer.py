"""Data transforming classes."""
from abc import ABC

import torch
from torchvision import transforms


class CVTransformer(ABC):
    """Abstract class performing input data transformations."""

    def resize_and_center(
        self, img: torch.Tensor  # pylint: disable=unused-argument
    ) -> torch.Tensor:
        """Resize and ceneter image.

        Args:
            img: Image tensor to be resized and centered.

        Returns:
            Resized and centered image tensor.
        """

    def transform(
        self, img: torch.Tensor
    ) -> torch.Tensor:  # pylint: disable=unused-argument
        """Transform image to desired format.

        Args:
            img: Image to transform.

        Returns:
            Return tensor.
        """


class ExplainerCVTransformer(CVTransformer):
    """Computer vision tasks input data transformation class."""

    def resize_and_center(self, img: torch.Tensor) -> torch.Tensor:
        """Resize and ceneter image.

        Args:
            img: Image tensor to be resized and centered.

        Returns:
            Resized and centered image tensor.
        """
        transform = transforms.Compose(
            [transforms.Resize(256), transforms.CenterCrop(224)]
        )
        return transform(img)

    def transform(self, img: torch.Tensor) -> torch.Tensor:
        """Transform image to desired

        Args:
            img: Image tensor to transform.

        Returns:
            Resized image and normalized image tensor.
        """
        transform_normalize = transforms.Normalize(
            mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]
        )
        input_data = transform_normalize(img)
        input_data = input_data.unsqueeze(0)
        return input_data
