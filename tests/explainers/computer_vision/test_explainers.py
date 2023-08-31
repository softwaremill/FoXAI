# pylint: disable = missing-class-docstring
import inspect
import logging
from typing import Any, Callable, Dict, List, Optional  # , TypeAlias
from unittest.mock import MagicMock, patch

import pytest
import torch
from torchvision import transforms

from foxai import explainer
from foxai.context_manager import (
    CVClassificationExplainers,
    ExplainerWithParams,
    FoXaiExplainer,
)
from foxai.explainer.base_explainer import CVExplainerT
from foxai.logger import create_logger
from tests.pickachu_image import pikachu_image
from tests.sample_model import SampleModel

GetExplainerKwargsT = Callable[
    [CVClassificationExplainers, torch.nn.Module], Dict[str, Any]
]

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
        """Sample model to run FoXaiExplainer on."""
        return SampleModel()

    @pytest.fixture
    def explainer_function_kwargs(self) -> GetExplainerKwargsT:
        def get_function_kwargs(
            explainer_class: CVClassificationExplainers,
            classifier: torch.nn.Module,
        ) -> Dict[str, Any]:
            # create parameters for explainers, that require custom parameters
            function_kwargs: Dict[str, Any] = {}
            explainer_class: CVExplainerT = getattr(explainer, explainer_class.value)
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

    @pytest.mark.parametrize("explainer_class", list(CVClassificationExplainers))
    @pytest.mark.parametrize("batch_size", [1, 3])
    def test_explainers_cpu_for_batch_data(
        self,
        batch_size: int,
        classifier: SampleModel,
        explainer_function_kwargs: GetExplainerKwargsT,
        explainer_class: CVClassificationExplainers,
    ):
        """Test all available explainers on a simple classifier model using cpu for batch input data."""
        classifier.train()
        transform = transforms.Compose(
            [
                transforms.ToTensor(),
                transforms.Grayscale(),
                transforms.Resize(size=224),
                transforms.CenterCrop(size=224),
            ]
        )
        batch_img_tensor: torch.Tensor = torch.vstack(
            [transform(pikachu_image).unsqueeze(0) for _ in range(0, batch_size)]
        )

        function_kwargs: Dict[str, Any] = explainer_function_kwargs(
            explainer_class=explainer_class,
            classifier=classifier,
        )
        attribute_key: str = f"{explainer_class.name}_0"
        baselines: Optional[torch.Tensor] = None
        if "SHAP" in explainer_class.name:
            baselines = torch.randn(
                (2 * batch_size,) + tuple(batch_img_tensor.shape[1:])
            )

        with FoXaiExplainer(
            model=classifier,
            explainers=[
                ExplainerWithParams(
                    explainer_name=explainer_class,
                    baselines=baselines,
                    **function_kwargs,
                ),
            ],
        ) as xai_model:
            _, explanations = xai_model(batch_img_tensor)

        assert len(explanations[attribute_key].shape) == len(batch_img_tensor.shape)
        assert explanations[attribute_key].shape[0] == batch_img_tensor.shape[0]

    @pytest.mark.parametrize("explainer_class", list(CVClassificationExplainers))
    @pytest.mark.parametrize("batch_size", [1, 3])
    def test_explainers_gpu_for_batch_data(
        self,
        batch_size: int,
        classifier: SampleModel,
        explainer_function_kwargs: GetExplainerKwargsT,
        explainer_class: CVClassificationExplainers,
    ):
        """Test all available explainers on a simple classifier model using gpu for batch input data."""

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
        batch_img_tensor: torch.Tensor = torch.vstack(
            [transform(pikachu_image).unsqueeze(0) for _ in range(0, batch_size)]
        ).to(device=device)

        function_kwargs: Dict[str, Any] = explainer_function_kwargs(
            explainer_class=explainer_class,
            classifier=classifier,
        )
        attribute_key: str = f"{explainer_class.name}_0"
        baselines: Optional[torch.Tensor] = None
        if "SHAP" in explainer_class.name:
            baselines = torch.randn(
                (2 * batch_size,) + tuple(batch_img_tensor.shape[1:]), device=device
            )

        with FoXaiExplainer(
            model=classifier,
            explainers=[
                ExplainerWithParams(
                    explainer_name=explainer_class,
                    **function_kwargs,
                    baselines=baselines,
                ),
            ],
        ) as xai_model:
            _, explanations = xai_model(batch_img_tensor)

        assert len(explanations[attribute_key].shape) == len(batch_img_tensor.shape)
        assert explanations[attribute_key].shape[0] == batch_img_tensor.shape[0]

    @pytest.mark.parametrize("explainer_class", list(CVClassificationExplainers))
    @pytest.mark.parametrize("batch_size", [1, 3])
    def test_explainers_gpu_for_batch_data_and_batch_baselines(
        self,
        batch_size: int,
        classifier: SampleModel,
        explainer_function_kwargs: GetExplainerKwargsT,
        explainer_class: CVClassificationExplainers,
    ):
        """Test all available explainers on a simple classifier model using gpu for batch input data."""

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
        batch_img_tensor: torch.Tensor = torch.vstack(
            [transform(pikachu_image).unsqueeze(0) for _ in range(0, batch_size)]
        ).to(device=device)

        function_kwargs: Dict[str, Any] = explainer_function_kwargs(
            explainer_class=explainer_class,
            classifier=classifier,
        )
        attribute_key: str = f"{explainer_class.name}_0"
        baselines: Optional[torch.Tensor] = None
        if "SHAP" in explainer_class.name:
            baselines = torch.randn(
                (
                    10,
                    2 * batch_size,
                )
                + tuple(batch_img_tensor.shape[1:]),
                device=device,
            )

        with FoXaiExplainer(
            model=classifier,
            explainers=[
                ExplainerWithParams(
                    explainer_name=explainer_class,
                    **function_kwargs,
                    baselines=baselines,
                ),
            ],
        ) as xai_model:
            _, explanations = xai_model(batch_img_tensor)

        assert len(explanations[attribute_key].shape) == len(batch_img_tensor.shape)
        assert explanations[attribute_key].shape[0] == batch_img_tensor.shape[0]


@patch(
    "foxai.explainer.computer_vision.algorithm.conductance.LayerConductance.attribute"
)
def test_conductance_raises_error_if_attributes_are_empty(
    fake_attribute: MagicMock,
) -> None:
    """Test if function raises RuntimeError when empty tensor returned from attribute method."""
    model = SampleModel()
    fake_attribute.return_value = torch.Tensor()
    explainer_alg = explainer.LayerConductanceCVExplainer()
    with pytest.raises(RuntimeError):
        _ = explainer_alg.calculate_features(
            model=model,
            input_data=torch.zeros((1, 1, 28, 28)),
            pred_label_idx=0,
            layer=model.conv1,
        )


@patch("foxai.explainer.computer_vision.algorithm.deconv.Deconvolution.attribute")
def test_deconvolution_raises_error_if_attributes_are_empty(
    fake_attribute: MagicMock,
) -> None:
    """Test if function raises RuntimeError when empty tensor returned from attribute method."""
    model = SampleModel()
    fake_attribute.return_value = torch.Tensor()
    explainer_alg = explainer.DeconvolutionCVExplainer()
    with pytest.raises(RuntimeError):
        _ = explainer_alg.calculate_features(
            model=model,
            input_data=torch.zeros((1, 1, 28, 28)),
            pred_label_idx=0,
        )


@patch("foxai.explainer.computer_vision.algorithm.deeplift_shap.DeepLiftShap.attribute")
def test_deepliftshap_raises_error_if_attributes_are_empty(
    fake_attribute: MagicMock,
) -> None:
    """Test if function raises RuntimeError when empty tensor returned from attribute method."""
    model = SampleModel()
    fake_attribute.return_value = torch.Tensor()
    explainer_alg = explainer.DeepLIFTSHAPCVExplainer()
    with pytest.raises(RuntimeError):
        _ = explainer_alg.calculate_features(
            model=model,
            input_data=torch.zeros((1, 1, 28, 28)),
            baselines=torch.zeros((1, 1, 28, 28)),
            pred_label_idx=0,
        )


@patch(
    "foxai.explainer.computer_vision.algorithm.deeplift_shap.LayerDeepLiftShap.attribute"
)
def test_layer_deepliftshap_raises_error_if_attributes_are_empty(
    fake_attribute: MagicMock,
) -> None:
    """Test if function raises RuntimeError when empty tensor returned from attribute method."""
    model = SampleModel()
    fake_attribute.return_value = torch.Tensor()
    explainer_alg = explainer.LayerDeepLIFTSHAPCVExplainer()
    with pytest.raises(RuntimeError):
        _ = explainer_alg.calculate_features(
            model=model,
            input_data=torch.zeros((1, 1, 28, 28)),
            baselines=torch.zeros((1, 1, 28, 28)),
            pred_label_idx=0,
        )


@patch("foxai.explainer.computer_vision.algorithm.deeplift.DeepLift.attribute")
def test_deeplift_raises_error_if_attributes_are_empty(
    fake_attribute: MagicMock,
) -> None:
    """Test if function raises RuntimeError when empty tensor returned from attribute method."""
    model = SampleModel()
    fake_attribute.return_value = torch.Tensor()
    explainer_alg = explainer.DeepLIFTCVExplainer()
    with pytest.raises(RuntimeError):
        _ = explainer_alg.calculate_features(
            model=model,
            input_data=torch.zeros((1, 1, 28, 28)),
            pred_label_idx=0,
        )


@patch("foxai.explainer.computer_vision.algorithm.deeplift.LayerDeepLift.attribute")
def test_layer_deeplift_raises_error_if_attributes_are_empty(
    fake_attribute: MagicMock,
) -> None:
    """Test if function raises RuntimeError when empty tensor returned from attribute method."""
    model = SampleModel()
    fake_attribute.return_value = torch.Tensor()
    explainer_alg = explainer.LayerDeepLIFTCVExplainer()
    with pytest.raises(RuntimeError):
        _ = explainer_alg.calculate_features(
            model=model,
            input_data=torch.zeros((1, 1, 28, 28)),
            pred_label_idx=0,
        )


@patch("foxai.explainer.computer_vision.algorithm.gradcam.GuidedGradCam.attribute")
def test_gradcam_raises_error_if_attributes_are_empty(
    fake_attribute: MagicMock,
) -> None:
    """Test if function raises RuntimeError when empty tensor returned from attribute method."""
    model = SampleModel()
    fake_attribute.return_value = torch.Tensor()
    explainer_alg = explainer.GuidedGradCAMCVExplainer()
    with pytest.raises(RuntimeError):
        _ = explainer_alg.calculate_features(
            model=model,
            input_data=torch.zeros((1, 1, 28, 28)),
            pred_label_idx=0,
            layer=model.conv1,
        )


@patch("foxai.explainer.computer_vision.algorithm.gradcam.LayerBaseGradCAM.forward")
def test_layer_gradcam_raises_error_if_attributes_are_empty(
    fake_attribute: MagicMock,
) -> None:
    """Test if function raises RuntimeError when empty tensor returned from attribute method."""
    model = SampleModel()
    fake_attribute.return_value = torch.Tensor()
    explainer_alg = explainer.LayerGradCAMCVExplainer()
    with pytest.raises(RuntimeError):
        _ = explainer_alg.calculate_features(
            model=model,
            input_data=torch.zeros((1, 1, 28, 28)),
            pred_label_idx=0,
            layer=model.conv1,
        )


@patch("foxai.explainer.computer_vision.algorithm.gradient_shap.GradientShap.attribute")
def test_gradient_shap_raises_error_if_attributes_are_empty(
    fake_attribute: MagicMock,
) -> None:
    """Test if function raises RuntimeError when empty tensor returned from attribute method."""
    model = SampleModel()
    fake_attribute.return_value = torch.Tensor()
    explainer_alg = explainer.GradientSHAPCVExplainer()
    with pytest.raises(RuntimeError):
        _ = explainer_alg.calculate_features(
            model=model,
            input_data=torch.zeros((1, 1, 28, 28)),
            pred_label_idx=0,
        )


@patch(
    "foxai.explainer.computer_vision.algorithm.gradient_shap.LayerGradientShap.attribute"
)
def test_layer_gradient_shap_raises_error_if_attributes_are_empty(
    fake_attribute: MagicMock,
) -> None:
    """Test if function raises RuntimeError when empty tensor returned from attribute method."""
    model = SampleModel()
    fake_attribute.return_value = torch.Tensor()
    explainer_alg = explainer.LayerGradientSHAPCVExplainer()
    with pytest.raises(RuntimeError):
        _ = explainer_alg.calculate_features(
            model=model,
            input_data=torch.zeros((1, 1, 28, 28)),
            pred_label_idx=0,
        )


@patch("foxai.explainer.computer_vision.algorithm.gradient_utils.compute_gradients")
def test_input_x_gradient_raises_error_if_attributes_are_empty(
    fake_attribute: MagicMock,
) -> None:
    """Test if function raises RuntimeError when empty tensor returned from attribute method."""
    model = SampleModel()
    fake_attribute.return_value = torch.Tensor()
    explainer_alg = explainer.InputXGradientCVExplainer()
    with pytest.raises(RuntimeError):
        _ = explainer_alg.calculate_features(
            model=model,
            input_data=torch.zeros((1, 1, 28, 28)),
            pred_label_idx=0,
        )


@patch(
    "foxai.explainer.computer_vision.algorithm.gradient_utils.compute_layer_gradients"
)
def test_layer_input_x_gradient_raises_error_if_attributes_are_empty(
    fake_attribute: MagicMock,
) -> None:
    """Test if function raises RuntimeError when empty tensor returned from attribute method."""
    model = SampleModel()
    fake_attribute.return_value = torch.Tensor()
    explainer_alg = explainer.LayerInputXGradientCVExplainer()
    with pytest.raises(RuntimeError):
        _ = explainer_alg.calculate_features(
            model=model,
            input_data=torch.zeros((1, 1, 28, 28)),
            pred_label_idx=0,
        )


@patch("foxai.explainer.computer_vision.algorithm.lrp.LRP.attribute")
def test_lrp_raises_error_if_attributes_are_empty(
    fake_attribute: MagicMock,
) -> None:
    """Test if function raises RuntimeError when empty tensor returned from attribute method."""
    model = SampleModel()
    fake_attribute.return_value = torch.Tensor()
    explainer_alg = explainer.LRPCVExplainer()
    with pytest.raises(RuntimeError):
        _ = explainer_alg.calculate_features(
            model=model,
            input_data=torch.zeros((1, 1, 28, 28)),
            pred_label_idx=0,
        )


@patch("foxai.explainer.computer_vision.algorithm.lrp.LayerLRP.attribute")
def test_layer_lrp_raises_error_if_attributes_are_empty(
    fake_attribute: MagicMock,
) -> None:
    """Test if function raises RuntimeError when empty tensor returned from attribute method."""
    model = SampleModel()
    fake_attribute.return_value = torch.Tensor()
    explainer_alg = explainer.LayerLRPCVExplainer()
    with pytest.raises(RuntimeError):
        _ = explainer_alg.calculate_features(
            model=model,
            input_data=torch.zeros((1, 1, 28, 28)),
            pred_label_idx=0,
        )


@patch(
    "foxai.explainer.computer_vision.algorithm.integrated_gradients.IntegratedGradients.attribute"
)
def test_integrated_gradients_raises_error_if_attributes_are_empty(
    fake_attribute: MagicMock,
) -> None:
    """Test if function raises RuntimeError when empty tensor returned from attribute method."""
    model = SampleModel()
    fake_attribute.return_value = torch.Tensor()
    explainer_alg = explainer.IntegratedGradientsCVExplainer()
    with pytest.raises(RuntimeError):
        _ = explainer_alg.calculate_features(
            model=model,
            input_data=torch.zeros((1, 1, 28, 28)),
            pred_label_idx=0,
        )


@patch(
    "foxai.explainer.computer_vision.algorithm.integrated_gradients.LayerIntegratedGradients.attribute"
)
def test_layer_integrated_gradients_raises_error_if_attributes_are_empty(
    fake_attribute: MagicMock,
) -> None:
    """Test if function raises RuntimeError when empty tensor returned from attribute method."""
    model = SampleModel()
    fake_attribute.return_value = torch.Tensor()
    explainer_alg = explainer.LayerIntegratedGradientsCVExplainer()
    with pytest.raises(RuntimeError):
        _ = explainer_alg.calculate_features(
            model=model,
            input_data=torch.zeros((1, 1, 28, 28)),
            pred_label_idx=0,
        )


@patch("foxai.explainer.computer_vision.algorithm.noise_tunnel.NoiseTunnel.attribute")
def test_noise_tunnel_raises_error_if_attributes_are_empty(
    fake_attribute: MagicMock,
) -> None:
    """Test if function raises RuntimeError when empty tensor returned from attribute method."""
    model = SampleModel()
    fake_attribute.return_value = torch.Tensor()
    explainer_alg = explainer.NoiseTunnelCVExplainer()
    with pytest.raises(RuntimeError):
        _ = explainer_alg.calculate_features(
            model=model,
            input_data=torch.zeros((1, 1, 28, 28)),
            pred_label_idx=0,
        )


@patch("foxai.explainer.computer_vision.algorithm.noise_tunnel.NoiseTunnel.attribute")
def test_layer_noise_tunnel_raises_error_if_attributes_are_empty(
    fake_attribute: MagicMock,
) -> None:
    """Test if function raises RuntimeError when empty tensor returned from attribute method."""
    model = SampleModel()
    fake_attribute.return_value = torch.Tensor()
    explainer_alg = explainer.LayerNoiseTunnelCVExplainer()
    with pytest.raises(RuntimeError):
        _ = explainer_alg.calculate_features(
            model=model,
            input_data=torch.zeros((1, 1, 28, 28)),
            pred_label_idx=0,
        )


@patch("foxai.explainer.computer_vision.algorithm.occlusion.Occlusion.attribute")
def test_occulusion_raises_error_if_attributes_are_empty(
    fake_attribute: MagicMock,
) -> None:
    """Test if function raises RuntimeError when empty tensor returned from attribute method."""
    model = SampleModel()
    fake_attribute.return_value = torch.Tensor()
    explainer_alg = explainer.OcclusionCVExplainer()
    with pytest.raises(RuntimeError):
        _ = explainer_alg.calculate_features(
            model=model,
            input_data=torch.zeros((1, 1, 28, 28)),
            pred_label_idx=0,
        )


@patch("foxai.explainer.computer_vision.algorithm.gradient_utils.compute_gradients")
def test_saliency_raises_error_if_attributes_are_empty(
    fake_attribute: MagicMock,
) -> None:
    """Test if function raises RuntimeError when empty tensor returned from attribute method."""
    model = SampleModel()
    fake_attribute.return_value = torch.Tensor()
    explainer_alg = explainer.SaliencyCVExplainer()
    with pytest.raises(RuntimeError):
        _ = explainer_alg.calculate_features(
            model=model,
            input_data=torch.zeros((1, 1, 28, 28)),
            pred_label_idx=0,
        )
