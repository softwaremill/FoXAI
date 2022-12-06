"""Function to load image from storage."""
import torch
from PIL import Image
from torchvision import transforms


def load_image(img_path: str) -> torch.Tensor:
    """Load image tensor from path.

    Args:
        img_path: Path to image location.

    Returns:
        Image tensor.
    """
    img = Image.open(img_path)
    converter = transforms.ToTensor()
    return converter(img)
