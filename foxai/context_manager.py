"""
Run xai alongside with inference.

Example:
    with FoXaiExplainer(
        model=classifier,
        explainers=[
            ExplainerWithParams(
                explainer_name=Explainers.CV_GRADIENT_SHAP_EXPLAINER,
                n_samples=100,
                stdevs=0.0005,
            ),
        ],
        target=pred_label_idx,
    ) as xai_model:
        output, xai_explanations = xai_model(img_tensor)
"""
import logging
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Dict, Generic, List, Optional, Tuple, Union, cast

import torch

from foxai import explainer
from foxai.explainer import (
    DeconvolutionCVExplainer,
    DeepLIFTCVExplainer,
    DeepLIFTSHAPCVExplainer,
    FullGradientCVExplainer,
    GradientSHAPCVExplainer,
    GuidedBackpropCVExplainer,
    GuidedGradCAMCVExplainer,
    InputXGradientCVExplainer,
    IntegratedGradientsCVExplainer,
    LayerConductanceCVExplainer,
    LayerDeepLIFTCVExplainer,
    LayerDeepLIFTSHAPCVExplainer,
    LayerGradCAMCVExplainer,
    LayerGradCAMObjectDetectionExplainer,
    LayerGradientSHAPCVExplainer,
    LayerInputXGradientCVExplainer,
    LayerIntegratedGradientsCVExplainer,
    LayerLRPCVExplainer,
    LayerNoiseTunnelCVExplainer,
    LRPCVExplainer,
    NoiseTunnelCVExplainer,
    OcclusionCVExplainer,
    SaliencyCVExplainer,
)
from foxai.explainer.base_explainer import CVExplainerT
from foxai.explainer.computer_vision.types import ModelType
from foxai.logger import create_logger

_LOGGER: Optional[logging.Logger] = None


def log() -> logging.Logger:
    """Get or create logger."""
    # pylint: disable = global-statement
    global _LOGGER
    if _LOGGER is None:
        _LOGGER = create_logger(__name__)
    return _LOGGER


class CVClassificationExplainers(Enum):
    """Enum of supported computer vision classification explainers types."""

    CV_OCCLUSION_EXPLAINER: str = OcclusionCVExplainer.__name__
    CV_INTEGRATED_GRADIENTS_EXPLAINER: str = IntegratedGradientsCVExplainer.__name__
    CV_NOISE_TUNNEL_EXPLAINER: str = NoiseTunnelCVExplainer.__name__
    CV_GRADIENT_SHAP_EXPLAINER: str = GradientSHAPCVExplainer.__name__
    CV_LRP_EXPLAINER: str = LRPCVExplainer.__name__
    CV_FULL_GRADIENTS_EXPLAINER: str = FullGradientCVExplainer.__name__
    CV_GUIDEDGRADCAM_EXPLAINER: str = GuidedGradCAMCVExplainer.__name__
    CV_LAYER_INTEGRATED_GRADIENTS_EXPLAINER: str = (
        LayerIntegratedGradientsCVExplainer.__name__
    )
    CV_LAYER_NOISE_TUNNEL_EXPLAINER: str = LayerNoiseTunnelCVExplainer.__name__
    CV_LAYER_GRADIENT_SHAP_EXPLAINER: str = LayerGradientSHAPCVExplainer.__name__
    CV_LAYER_LRP_EXPLAINER: str = LayerLRPCVExplainer.__name__
    CV_LAYER_GRADCAM_EXPLAINER: str = LayerGradCAMCVExplainer.__name__
    CV_INPUT_X_GRADIENT_EXPLAINER: str = InputXGradientCVExplainer.__name__
    CV_LAYER_INPUT_X_GRADIENT_EXPLAINER: str = LayerInputXGradientCVExplainer.__name__
    CV_DEEPLIFT_EXPLAINER: str = DeepLIFTCVExplainer.__name__
    CV_LAYER_DEEPLIFT_EXPLAINER: str = LayerDeepLIFTCVExplainer.__name__
    CV_DEEPLIFT_SHAP_EXPLAINER: str = DeepLIFTSHAPCVExplainer.__name__
    CV_LAYER_DEEPLIFT_SHAP_EXPLAINER: str = LayerDeepLIFTSHAPCVExplainer.__name__
    CV_DECONVOLUTION_EXPLAINER: str = DeconvolutionCVExplainer.__name__
    CV_LAYER_CONDUCTANCE_EXPLAINER: str = LayerConductanceCVExplainer.__name__
    CV_SALIENCY_EXPLAINER: str = SaliencyCVExplainer.__name__
    CV_GUIDED_BACKPOPAGATION_EXPLAINER: str = GuidedBackpropCVExplainer.__name__


class CVObjectDetectionExplainers(Enum):
    """Enum of supported computer vision object detection explainers types."""

    CV_LAYER_GRADCAM_OBJECT_DETECTION_EXPLAINER: str = (
        LayerGradCAMObjectDetectionExplainer.__name__
    )


@dataclass
class ExplainerWithParams:
    """Holder for explainer name (class name) and it's params"""

    explainer_name: Union[CVClassificationExplainers, CVObjectDetectionExplainers]
    kwargs: Dict[str, Any] = field(default_factory=dict)

    def __init__(
        self,
        explainer_name: Union[CVClassificationExplainers, CVObjectDetectionExplainers],
        **kwargs
    ) -> None:
        self.explainer_name = explainer_name
        if kwargs:
            self.kwargs = kwargs
        else:
            self.kwargs = {}


@dataclass
class ExplainerClassWithParams(Generic[CVExplainerT]):
    """Holder for explainer class and it's params"""

    explainer_class: CVExplainerT
    kwargs: Dict[str, Any] = field(default_factory=dict)

    def __init__(self, explainer_class: CVExplainerT, **kwargs) -> None:
        self.explainer_class = explainer_class
        if kwargs:
            self.kwargs = kwargs
        else:
            self.kwargs = {}


class FoXaiExplainer(Generic[CVExplainerT]):
    """Context manager for FoXAI explanation.

    Example:
        with FoXaiExplainer(
            model=classifier,
            explainers=[
                ExplainerWithParams(
                    explainer_name=Explainers.CV_GRADIENT_SHAP_EXPLAINER,
                    n_samples=100,
                    stdevs=0.0005,
                ),
            ],
            target=pred_label_idx,
        ) as xai_model:
            output, xai_explanations = xai_model(img_tensor)

    Raises:
        ValueError: if no explainer provided
    """

    def __init__(
        self,
        model: ModelType,
        explainers: List[ExplainerWithParams],
        target: int = 0,
    ) -> None:
        """
        Args:
            model: the torch model to exavluate with CV explainer
            explainers: explainers names list, to use for model evaluation.
            target: predicted target index. For which class to generate xai.
        """

        if not explainers:
            raise ValueError("At leas one explainer should be defined.")

        self.model: ModelType = model
        self.prev_model_training_state: bool = self.model.training

        self.explainer_map: Dict[str, ExplainerClassWithParams] = {
            explainer_with_params.explainer_name.name: ExplainerClassWithParams(
                explainer_class=getattr(
                    explainer, explainer_with_params.explainer_name.value
                )(),
                **explainer_with_params.kwargs,
            )
            for explainer_with_params in explainers
        }

        self.target: int = target

    def __enter__(self) -> "FoXaiExplainer":
        """Verify if model is in eval() mode.

        Raises:
            ValueError: if the model is in training mode.

        Returns:
            the foxai class instance.
        """
        self.prev_torch_grad = torch.is_grad_enabled()
        if not self.prev_torch_grad:
            log_msg: str = (
                "Torch model explainer can be called only with enabled "
                + "gradients, as it depends on gradients computations. The model is going "
                + "to be toggled to gradients enabled. For the "
                + "model prediction, the gradient is temporary turned off."
            )
            log().warning(log_msg)
            torch.set_grad_enabled(True)

        if self.model.training:
            self.model.eval()
            log().warning(
                "The model should be in the eval model. Toggling it to eval mode right now."
            )

        return self

    def __exit__(self, exc_type: Any, exc_value: Any, traceback: Any) -> None:
        """If the torch was not recording gradient, before entering in the
        context manager mode, switch it back to no gradient recording mode.

        If the torch was recording gradient before entering in the context
        manager modes, nothings changes.

        Setup model to previous state: `eval` or `training` to match initial
        state.
        """
        torch.set_grad_enabled(self.prev_torch_grad)
        self.model.train(self.prev_model_training_state)

    def __call__(self, *args, **kwargs) -> Tuple[Any, Dict[str, torch.Tensor]]:
        """Run model prediction and explain the model with given explainers.

        Explainers and model are defined as the class parameter.

        Args:
            list of arguments for the torch.nn.Module forward method.

        Returns:
            the model output and explanations for each requested explainer.
        """

        with torch.no_grad():
            model_output: Any = self.model(*args, **kwargs)

        if len(args) != 1:
            # TODO: add support in explainer for multiple input models
            raise NotImplementedError(
                "calculate_features() functions "
                + "in explainers does not support multiple inputs to the model."
            )
        input_tensor: torch.Tensor = cast(torch.Tensor, args)[0]
        # cashe tensor requires grad state
        prev_requires_grad: bool = input_tensor.requires_grad
        # turn on requires grad for the input tensor
        input_tensor.requires_grad = True

        explanations: Dict[str, torch.Tensor] = {}
        for explainer_name in self.explainer_map:
            # zero the previous gradient for the model
            self.model.zero_grad()
            # run explainer
            explainer_kwargs: Dict[str, Any] = self.explainer_map[explainer_name].kwargs
            explainer_class: CVExplainerT = self.explainer_map[
                explainer_name
            ].explainer_class

            explanations[explainer_name] = (
                explainer_class.calculate_features(
                    model=self.model,
                    input_data=input_tensor,
                    pred_label_idx=self.target,
                    **explainer_kwargs,
                )
                .detach()
                .cpu()
            )
            input_tensor.grad = None

        # restore tensor requires grad state
        input_tensor.requires_grad = prev_requires_grad

        return model_output, explanations
