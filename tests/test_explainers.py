# pylint: disable = missing-class-docstring
import inspect
import logging
from typing import Any, Callable, Dict, List, Optional  # , TypeAlias
from unittest.mock import MagicMock, patch

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

    @pytest.mark.parametrize("explainer_name", list(Explainers))
    def test_explainers_cpu(
        self,
        classifier: SampleModel,
        explainer_function_kwargs: GetExplainerKwargsT,
        explainer_name: str,
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

    @pytest.mark.parametrize("explainer_name", list(Explainers))
    def test_explainers_gpu(
        self,
        classifier: SampleModel,
        explainer_function_kwargs: GetExplainerKwargsT,
        explainer_name: str,
    ):
        """Test all available explainers on a simple classifier model using gpu."""

        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        if device != torch.device("cuda"):
            pytest.fail("GPU not detected. Skiping GPU test.")
            # log().warning("GPU not detected. Skiping GPU test.")
            # return
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


@patch("autoxai.explainer.conductance.LayerConductance.attribute")
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


@patch("autoxai.explainer.deconv.Deconvolution.attribute")
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


@patch("autoxai.explainer.deeplift_shap.DeepLiftShap.attribute")
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
            pred_label_idx=0,
        )


@patch("autoxai.explainer.deeplift_shap.LayerDeepLiftShap.attribute")
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
            pred_label_idx=0,
        )


@patch("autoxai.explainer.deeplift.DeepLift.attribute")
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


@patch("autoxai.explainer.deeplift.LayerDeepLift.attribute")
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


@patch("autoxai.explainer.gradcam.GuidedGradCam.attribute")
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


@patch("autoxai.explainer.gradcam.LayerGradCam.attribute")
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


@patch("autoxai.explainer.gradient_shap.GradientShap.attribute")
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


@patch("autoxai.explainer.gradient_shap.LayerGradientShap.attribute")
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


@patch("autoxai.explainer.input_x_gradient.InputXGradient.attribute")
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


@patch("autoxai.explainer.input_x_gradient.LayerGradientXActivation.attribute")
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


@patch("autoxai.explainer.lrp.LRP.attribute")
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


@patch("autoxai.explainer.lrp.LayerLRP.attribute")
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


@patch("autoxai.explainer.integrated_gradients.IntegratedGradients.attribute")
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


@patch("autoxai.explainer.integrated_gradients.LayerIntegratedGradients.attribute")
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


@patch("autoxai.explainer.noise_tunnel.NoiseTunnel.attribute")
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


@patch("autoxai.explainer.noise_tunnel.NoiseTunnel.attribute")
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


@patch("autoxai.explainer.occlusion.Occlusion.attribute")
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


@patch("autoxai.explainer.saliency.Saliency.attribute")
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
