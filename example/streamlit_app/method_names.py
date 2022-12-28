"""File with XAI algorithm names Enum class."""

from enum import Enum
from typing import Optional

from autoxai.explainer.conductance import LayerConductanceCVExplainer
from autoxai.explainer.deconv import DeconvolutionCVExplainer
from autoxai.explainer.deeplift import DeepLIFTCVExplainer, LayerDeepLIFTCVExplainer
from autoxai.explainer.deeplift_shap import (
    DeepLIFTSHAPCVExplainer,
    LayerDeepLIFTSHAPCVExplainer,
)
from autoxai.explainer.gradcam import GuidedGradCAMCVExplainer, LayerGradCAMCVExplainer
from autoxai.explainer.gradient_shap import (
    GradientSHAPCVExplainer,
    LayerGradientSHAPCVExplainer,
)
from autoxai.explainer.input_x_gradient import (
    InputXGradientCVExplainer,
    LayerInputXGradientCVExplainer,
)
from autoxai.explainer.integrated_gradients import (
    IntegratedGradientsCVExplainer,
    LayerIntegratedGradientsCVExplainer,
)
from autoxai.explainer.lrp import LayerLRPCVExplainer, LRPCVExplainer
from autoxai.explainer.noise_tunnel import (
    LayerNoiseTunnelCVExplainer,
    NoiseTunnelCVExplainer,
)
from autoxai.explainer.occulusion import OcculusionCVExplainer
from autoxai.explainer.saliency import SaliencyCVExplainer


class MethodName(Enum):
    """XAI algorithms names."""

    OCCULUSION = OcculusionCVExplainer().algorithm_name
    NOISE_TUNNEL = NoiseTunnelCVExplainer().algorithm_name
    LAYER_NOISE_TUNNEL = LayerNoiseTunnelCVExplainer().algorithm_name
    GRADIENT_SHAP = GradientSHAPCVExplainer().algorithm_name
    LAYER_GRADIENT_SHAP = LayerGradientSHAPCVExplainer().algorithm_name
    LRP = LRPCVExplainer().algorithm_name
    LAYER_LRP = LayerLRPCVExplainer().algorithm_name
    GUIDED_GRAD_CAM = GuidedGradCAMCVExplainer().algorithm_name
    LAYER_GRAD_CAM = LayerGradCAMCVExplainer().algorithm_name
    INTEGRATED_GRADIENTS = IntegratedGradientsCVExplainer().algorithm_name
    LAYER_INTEGRATED_GRADIENTS = LayerIntegratedGradientsCVExplainer().algorithm_name
    SALIENCY = SaliencyCVExplainer().algorithm_name
    DEEP_LIFT = DeepLIFTCVExplainer().algorithm_name
    LAYER_DEEP_LIFT = LayerDeepLIFTCVExplainer().algorithm_name
    DEEP_LIFT_SHAP = DeepLIFTSHAPCVExplainer().algorithm_name
    LAYER_DEEP_LIFT_SHAP = LayerDeepLIFTSHAPCVExplainer().algorithm_name
    DECONVOLUTION = DeconvolutionCVExplainer().algorithm_name
    INPUT_X_GRADIENT = InputXGradientCVExplainer().algorithm_name
    LAYER_INPUT_X_GRADIENT = LayerInputXGradientCVExplainer().algorithm_name
    LAYER_CONDUCTANCE = LayerConductanceCVExplainer().algorithm_name

    @classmethod
    def from_string(cls, name: str) -> "MethodName":
        """Return MethodName object from string.

        Args:
            name: Value of enum.

        Returns:
            MethodName object.

        Raises:
            ValueError if argument of the function is not a value of Enum.
        """
        obj: Optional[MethodName] = None
        for entry in cls:
            if entry.value == name:
                obj = entry
                break

        if obj is None:
            raise ValueError(f"Unrecognized Enum value: {name}")

        return obj
