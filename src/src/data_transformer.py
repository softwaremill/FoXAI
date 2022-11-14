from abc import ABC
import PIL
from typing import Tuple, Any
import torch
from torchvision import transforms


class CVTransformer(ABC):
    """Abstract class performing input data transformations."""

    def resize_and_center(self, img: PIL.Image) -> PIL.Image:  # pylint: disable=unused-argument
        """Resize and ceneter image.

        Args:
            img (PIL.Image): Image to be resized and centered.

        Returns:
            PIL.Image: Resized and centered image.
        """
        ...

    def transform(self, img: PIL.Image) -> Any:  # pylint: disable=unused-argument
        """Transform image to desired format.

        Args:
            img (PIL.Image): Image to transform.

        Returns:
            Any: Return any type You want to use.
        """
        ...


class ExplainerCVTransformer(CVTransformer):
    """Computer vision tasks input data transformation class."""

    def resize_and_center(self, img: PIL.Image) -> PIL.Image:
        """Resize and ceneter image.

        Args:
            img (PIL.Image): Image to be resized and centered.

        Returns:
            PIL.Image: Resized and centered image.
        """
        transform = transforms.Compose([transforms.Resize(256), transforms.CenterCrop(224), transforms.ToTensor()])
        return transform(img)

    def transform(self, img: PIL.Image) -> Tuple[torch.Tensor, torch.Tensor]:
        """Transform image to desired

        Args:
            img (PIL.Image): Image to transform.

        Returns:
            Tuple[torch.Tensor, torch.Tensor]: Tuple of resized image, resized and normalized image.
        """
        transform_normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        input_data = transform_normalize(img)
        input_data = input_data.unsqueeze(0)

        return input_data
