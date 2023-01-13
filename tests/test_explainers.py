# pylint: disable = missing-class-docstring
import inspect
import logging
from typing import Any, Callable, Dict, List, Optional  # , TypeAlias

import pytest
import torch
from torchvision import transforms

from autoxai import explainer
from autoxai.context_manager import AutoXaiExplainer, Explainers, ExplainerWithParams
from autoxai.explainer.base_explainer import CVExplainerT
from autoxai.logger import create_logger
from tests.pickachu_image import pikachu_image
from tests.sample_model import SampleModel

GetExplainerKwargsT = Callable[[Explainers, torch.nn.Module], Dict[str, Any]]

_LOGGER: Optional[logging.Logger] = None


def log() -> logging.Logger:
    """Get or create logger."""
    # pylint: disable = global-statement
    global _LOGGER
    if _LOGGER is None:
        _LOGGER = create_logger(__name__)
    return _LOGGER


class TestExplainers:
    """Test all explainers, if they run witout error."""

    @pytest.fixture
    def classifier(self) -> SampleModel:
        """Sample model to run AutoXaiExplainer on."""
        return SampleModel()

    @pytest.fixture
    def explainer_function_kwargs(self) -> GetExplainerKwargsT:
        def get_function_kwargs(
            explainer_name: Explainers, classifier: torch.nn.Module
        ) -> Dict[str, Any]:
            # create parameters for explainers, that require custom parameters
            function_kwargs: Dict[str, Any] = {}
            explainer_class: CVExplainerT = getattr(explainer, explainer_name.value)
            # check whether class contains 'create_explainer' function
            class_function: Optional[Callable] = getattr(
                explainer_class, "create_explainer", None
            )
            if callable(class_function):
                # if class contans 'create_explainer' function, check what type of arguments
                # does it require
                function_signature = inspect.signature(class_function)
                for function_params in function_signature.parameters.values():
                    if function_params.name == "layer":
                        # list all model convolution layers
                        conv_layers: List[torch.nn.Module] = []
                        for module in classifier.modules():
                            if isinstance(module, torch.nn.Conv2d):
                                conv_layers.append(module)
                        # pick the last convolution layer for explanation
                        function_kwargs[function_params.name] = conv_layers[-1]
            return function_kwargs

        return get_function_kwargs

    @pytest.fixture
    def check_currect_cuda_installed(self) -> bool:
        """Test whether CUDA accelerated torch is installed.

        Code copied from `torch/cuda/__init__.py` file, function `_check_cubins()`.
        """
        if torch.version.cuda is None:  # on ROCm we don't want this check
            return False

        arch_list = torch.cuda.get_arch_list()
        if len(arch_list) == 0:
            return False

        correct_cuda_installed: bool = False
        supported_sm = [int(arch.split("_")[1]) for arch in arch_list if "sm_" in arch]
        for idx in range(torch.cuda.device_count()):
            cap_major, _ = torch.cuda.get_device_capability(idx)
            # NVIDIA GPU compute architectures are backward compatible within major version
            supported = any(sm // 10 == cap_major for sm in supported_sm)
            if supported:
                correct_cuda_installed = True
                break

        return correct_cuda_installed

    def test_explainers_cpu(
        self, classifier: SampleModel, explainer_function_kwargs: GetExplainerKwargsT
    ):
        """Test all available explainers on a simple classifier model using cpu."""
        classifier.train()
        transform = transforms.Compose(
            [
                transforms.ToTensor(),
                transforms.Grayscale(),
                transforms.Resize(size=224),
                transforms.CenterCrop(size=224),
            ]
        )
        img_tensor: torch.Tensor = transform(pikachu_image).unsqueeze(0)

        for explainer_name in Explainers:
            function_kwargs: Dict[str, Any] = explainer_function_kwargs(
                explainer_name=explainer_name, classifier=classifier
            )

            with AutoXaiExplainer(
                model=classifier,
                explainers=[
                    ExplainerWithParams(
                        explainer_name=explainer_name,
                        **function_kwargs,
                    ),
                ],
            ) as xai_model:
                _, _ = xai_model(img_tensor)

    def test_explainers_gpu(
        self,
        classifier: SampleModel,
        explainer_function_kwargs: GetExplainerKwargsT,
        check_currect_cuda_installed: bool,
    ):
        """Test all available explainers on a simple classifier model using gpu."""

        if not check_currect_cuda_installed:
            log().warning("CUDA accelerated torch is not installed. Skipping GPU test.")
            return
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        if device != torch.device("cuda"):
            log().warning("GPU not detected. Skiping GPU test.")
            return
        else:
            log().info("GPU detected. Runing GPU tests...")

        classifier.train().to(device)
        transform = transforms.Compose(
            [
                transforms.ToTensor(),
                transforms.Grayscale(),
                transforms.Resize(size=224),
                transforms.CenterCrop(size=224),
            ]
        )
        img_tensor: torch.Tensor = transform(pikachu_image).unsqueeze(0).to(device)

        for explainer_name in Explainers:
            function_kwargs: Dict[str, Any] = explainer_function_kwargs(
                explainer_name=explainer_name, classifier=classifier
            )

            with AutoXaiExplainer(
                model=classifier,
                explainers=[
                    ExplainerWithParams(
                        explainer_name=explainer_name,
                        **function_kwargs,
                    ),
                ],
            ) as xai_model:
                _, _ = xai_model(img_tensor)
