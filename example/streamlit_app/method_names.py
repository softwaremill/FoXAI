"""File with XAI algorithm names Enum class."""

from enum import Enum
from typing import Optional

from src.explainer.gradcam import GuidedGradCAMCVExplainer, LayerGradCAMCVExplainer
from src.explainer.gradient_shap import (
    GradientSHAPCVExplainer,
    LayerGradientSHAPCVExplainer,
)
from src.explainer.integrated_gradients import (
    IntegratedGradientsCVExplainer,
    LayerIntegratedGradientsCVExplainer,
)
from src.explainer.lrp import LayerLRPCVExplainer, LRPCVExplainer
from src.explainer.noise_tunnel import (
    LayerNoiseTunnelCVExplainer,
    NoiseTunnelCVExplainer,
)
from src.explainer.occulusion import OcculusionCVExplainer


class MethodName(Enum):
    """XAI algorithm names."""

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
