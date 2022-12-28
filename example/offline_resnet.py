"""Entry point for CLI."""
import datetime
import os
import uuid
from argparse import ArgumentParser, Namespace
from pathlib import Path
from typing import Dict

import torch
import torch.nn.functional as F
import torchvision
from PIL import Image

from autoxai.cache_manager import LocalDirCacheManager
from autoxai.path_manager import ExperimentDataClass


class DataTransformer:  # pylint: disable = (too-few-public-methods)
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
        transform_normalize = torchvision.transforms.Normalize(
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
        return torchvision.transforms.Resize(256)(data)

    def center(self, data: torch.Tensor) -> torch.Tensor:
        """Center image.

        Args:
            data: Image tensor to be centered.

        Returns:
            Centered image tensor.
        """
        return torchvision.transforms.CenterCrop(224)(data)


def load_image(img_path: str) -> torch.Tensor:
    """Load image tensor from path.

    Args:
        img_path: Path to image location.

    Returns:
        Image tensor.
    """
    img = Image.open(img_path)
    converter = torchvision.transforms.ToTensor()
    return converter(img)


def load_model() -> torchvision.models.ResNet:
    """Load model to explain.

    Returns:
        Any model type.
    """
    model = torchvision.models.resnet18(pretrained=True)
    model = model.eval()

    return model


def load_model_idx_to_label() -> Dict[int, str]:
    """Load index to label mapping for model.

    Returns:
        Dictionary containgin index to label mapping.
    """
    weights = torchvision.models.ResNet18_Weights.DEFAULT
    return {  # pylint: disable = (unnecessary-comprehension)
        i: val for i, val in enumerate(weights.meta["categories"])
    }


def get_prediction(
    model: torchvision.models.ResNet, input_data: torch.Tensor
) -> torch.Tensor:
    """Get predicted label from model.

    Args:
        model: ResNet model.
        input_data: Input data tensor.

    Returns:
        Tensor with predicted label.
    """
    output = model(input_data)
    output = F.softmax(output, dim=1)
    _, pred_label_idx = torch.topk(output, 1)  # pylint: disable = (no-member)

    pred_label_idx.squeeze_()
    return pred_label_idx


def parse_args() -> Namespace:
    """Parse CLI arguments.

    Returns:
        Namespace with parsed arguments.
    """
    parser = ArgumentParser(description="AutoXAI - Explainable AI")
    parser.add_argument(
        "--cache",
        type=str,
        default="autoxai_cache/",
        help="Path to AutoXAI cache directory",
    )
    parser.add_argument("--img_path", type=str, required=True, help="Path to image")
    return parser.parse_args()


def main():
    """Entry point for application."""
    args = parse_args()

    experiment = ExperimentDataClass(
        base_path=args.cache,
        date=str(datetime.date.today()),
        uuid=str(uuid.uuid1()),
    )
    model = load_model()
    idx_to_label = load_model_idx_to_label()

    transformer = DataTransformer()
    cache_manager = LocalDirCacheManager()
    cache_manager.save_artifact(
        os.path.join(experiment.path, "labels", "idx_to_label.json"),
        idx_to_label,
    )

    filepath = args.img_path
    image_name = Path(filepath).name
    img = load_image(filepath)

    input_data = transformer.preprocess(data=img)
    cache_manager.save_artifact(
        os.path.join(experiment.path_to_data, image_name),
        input_data,
    )

    path: str = os.path.join(experiment.path, "training", "0", "model.pt")

    if not os.path.exists(path):
        os.makedirs(Path(path).parent)

    torch.save(model.state_dict(), path)


if __name__ == "__main__":
    main()
