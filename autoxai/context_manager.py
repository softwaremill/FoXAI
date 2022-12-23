"""
Run xai alongside with inference.

Example:
    with torch.no_grad():
        with AutoXaiExplainer(model=model) as xai_model:
            output, xai_explanations = xai_model(input_data)
"""
from enum import Enum
from typing import Any, Dict, List, Tuple

import torch
from PIL import Image
from PIL.PngImagePlugin import PngImageFile
from torch.nn import functional as F
from torchvision import transforms

from autoxai import explainer
from autoxai.cache_manager import LocalDirCacheManager

# pylint: disable = no-name-in-module
from autoxai.explainer.base_explainer import CVExplainerT  # type: ignore


class Explainers(Enum):
    """Enum of supported explainers types."""

    CV_OCCLUSION_EXPLAINER: str = "OcculusionCVExplainer"
    CV_INTEGRATED_GRADIENTS_EXPLAINER: str = "IntegratedGradientsCVExplainer"
    CV_NOISE_TUNNEL_EXPLAINER: str = "NoiseTunnelCVExplainer"
    CV_GRADIENT_SHAP_EXPLAINER: str = "GradientSHAPCVExplainer"
    CV_LRP_EXPLAINER: str = "LRPCVExplainer"
    CV_GUIDEDGRADCAM_EXPLAINER: str = "GuidedGradCAMCVExplainer"
    CV_LAYER_INTEGRATED_GRADIENTS_EXPLAINER: str = "LayerIntegratedGradientsCVExplainer"
    CV_LAYER_NOISE_TUNNEL_EXPLAINER: str = "LayerNoiseTunnelCVExplainer"
    CV_LAYER_GRADIENT_SHAP_EXPLAINER: str = "LayerGradientSHAPCVExplainer"
    CV_LAYER_LRP_EXPLAINER: str = "LayerLRPCVExplainer"
    CV_LAYER_GRADCAM_EXPLAINER: str = "LayerGradCAMCVExplainer"


def load_input_data(path: str) -> torch.Tensor:
    """Load preprocessed input data sample from given path.

    Args:
        cache_path: Path to cache directory.
        selected_date: Selected date.
        experiment_hash: Selected experiment hash.

    Returns:
        Preprocessed data sample.
    """
    cache_manager = LocalDirCacheManager()
    input_data = cache_manager.load_artifact(path)

    return input_data


class AutoXaiExplainer:
    """Context menager for AutoXAI explanation.

    Example:
        with torch.no_grad():
            with AutoXaiExplainer(model=model) as xai_model:
                output, xai_explanations = xai_model(input_data)

    Raises:
        ValueError: if no explainer provided
    """

    def __init__(
        self,
        model: torch.nn.Module,
        explainers: List[Explainers],
        target: int = 0,
    ) -> None:
        """
        Args:
            model: the torch model to exavluate with CV explainer
            explainers: explainers names list, to use for model evaluation.
            target: predicted target index. For which class to generate xai.
        """

        self.model: torch.nn.Module = model
        if not explainers:
            raise ValueError("At leas one explainer should be defined.")

        self.explainer_map: Dict[str, CVExplainerT] = {
            explainer_name.name: getattr(explainer, explainer_name.value)()
            for explainer_name in explainers
        }

        self.target: int = target

    def __enter__(self) -> "AutoXaiExplainer":
        """Verify if model is in eval() mode.

        Raises:
            ValueError: if the model is in trainin mode.

        Return:
            the autoxai class instance.
        """

        if self.model.training:
            raise ValueError(
                "The autoxai should be run on model, which is in eval mode."
            )
        return self

    def __exit__(self, exc_type: Any, exc_value: Any, traceback: Any) -> None:
        pass

    def __call__(self, *args, **kwargs) -> Tuple[Any, Dict[str, torch.Tensor]]:
        """Run model prediction and explain the model with given explainers.

        Return:
            the model output and explanations for each requested explainer.
        """
        model_output: Any = self.model(*args, **kwargs)

        explanations: Dict[str, torch.Tensor] = {}
        for explainer_name in self.explainer_map:
            explanations[explainer_name] = self.explainer_map[
                explainer_name
            ].calculate_features(
                model=self.model,
                input_data=args,
                pred_label_idx=self.target,
            )

        return model_output, explanations


if __name__ == "__main__":

    class SampleModel(torch.nn.Module):
        """Sample pytorch model for experiment."""

        def __init__(
            self,
            in_channels: int = 1,
            resolution: int = 224,
        ):
            super().__init__()
            self.stride: int = 16
            self.out_channels: int = 16
            self.conv1 = torch.nn.Conv2d(
                in_channels=in_channels,
                out_channels=self.out_channels,
                kernel_size=5,
                stride=16,
            )

            output_channels: int = (
                (resolution // self.stride) ** 2
            ) * self.out_channels
            self.cls = torch.nn.Sequential(
                torch.nn.Dropout(),
                torch.nn.Linear(in_features=output_channels, out_features=1, bias=True),
            )
            self.sigmoid = torch.nn.Sigmoid()
            self.name = "SampleModel"

        def forward(self, x_tensor: torch.Tensor) -> torch.Tensor:
            """Forward methid for the module."""
            x_tensor = F.relu(self.conv1(x_tensor))
            x_tensor = x_tensor.view(x_tensor.size(0), -1)
            x_tensor = self.cls(x_tensor)
            x_tensor = self.sigmoid(x_tensor)
            return x_tensor

    classifier: torch.nn.Module = SampleModel(in_channels=1).eval()
    input_image: PngImageFile = Image.open("./example/images/pikachu_image.png")
    transform = transforms.Compose(
        [
            transforms.Grayscale(),
            transforms.Resize(size=224),
            transforms.CenterCrop(size=224),
            transforms.ToTensor(),
        ]
    )
    img_tensor: torch.Tensor = transform(input_image).unsqueeze(0)

    with torch.no_grad():
        with AutoXaiExplainer(
            model=classifier,
            explainers=[Explainers.CV_NOISE_TUNNEL_EXPLAINER],
        ) as xai_model:
            output, xai_explanations = xai_model(img_tensor)
