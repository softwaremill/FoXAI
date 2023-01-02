"""File contains helper to get explainer class based on algorithm name."""

from autoxai.explainer.base_explainer import CVExplainer
from autoxai.explainer.gradcam import GuidedGradCAMCVExplainer, LayerGradCAMCVExplainer
from autoxai.explainer.gradient_shap import (
    GradientSHAPCVExplainer,
    LayerGradientSHAPCVExplainer,
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


def get_explainer_class(algorithm_name: str) -> CVExplainer:
    """Get Explainer class by name.

    Args:
        algorithm_name: Algorithm name of a explainer.

    Returns:
        Explainer class.

    Raises:
        RuntimeError if provided name is not recognized algorithm name.
    """
    retval: CVExplainer
    if algorithm_name == GradientSHAPCVExplainer().algorithm_name:
        retval = GradientSHAPCVExplainer()
    elif algorithm_name == LayerGradientSHAPCVExplainer().algorithm_name:
        retval = LayerGradientSHAPCVExplainer()
    elif algorithm_name == GuidedGradCAMCVExplainer().algorithm_name:
        retval = GuidedGradCAMCVExplainer()
    elif algorithm_name == LayerGradCAMCVExplainer().algorithm_name:
        retval = LayerGradCAMCVExplainer()
    elif algorithm_name == IntegratedGradientsCVExplainer().algorithm_name:
        retval = IntegratedGradientsCVExplainer()
    elif algorithm_name == LayerIntegratedGradientsCVExplainer().algorithm_name:
        retval = LayerIntegratedGradientsCVExplainer()
    elif algorithm_name == LRPCVExplainer().algorithm_name:
        retval = LRPCVExplainer()
    elif algorithm_name == LayerLRPCVExplainer().algorithm_name:
        retval = LayerLRPCVExplainer()
    elif algorithm_name == NoiseTunnelCVExplainer().algorithm_name:
        retval = NoiseTunnelCVExplainer()
    elif algorithm_name == LayerNoiseTunnelCVExplainer().algorithm_name:
        retval = LayerNoiseTunnelCVExplainer()
    elif algorithm_name == OcculusionCVExplainer().algorithm_name:
        retval = OcculusionCVExplainer()
    else:
        raise RuntimeError("Unrecognized algorithm name.")

    return retval
