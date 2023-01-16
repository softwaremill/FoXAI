"""File with LRP algorithm explainer classes."""

from abc import abstractmethod
from typing import Optional, Union

import torch
from captum.attr import LRP, LayerLRP
from captum.attr._utils.lrp_rules import EpsilonRule, GammaRule

from autoxai.explainer.base_explainer import CVExplainer
from autoxai.explainer.model_utils import get_last_conv_model_layer, modify_modules


class BaseLRPCVExplainer(CVExplainer):
    """Base LRP algorithm explainer."""

    @abstractmethod
    def create_explainer(
        self,
        model: torch.nn.Module,
        **kwargs,
    ) -> Union[LRP, LayerLRP]:
        """Create explainer object.

        Returns:
            Explainer object.
        """

    def calculate_features(
        self,
        model: torch.nn.Module,
        input_data: torch.Tensor,
        pred_label_idx: int,
        **kwargs,
    ) -> torch.Tensor:
        """Generate features image with LRP algorithm explainer.

        Args:
            model: Any DNN model You want to use.
            input_data: Input image.
            pred_label_idx: Predicted label.

        Returns:
            Features matrix.
        """
        layer: Optional[torch.nn.Module] = kwargs.get("layer", None)

        lrp = self.create_explainer(model=model, layer=layer)

        attributions = lrp.attribute(input_data, target=pred_label_idx)
        if attributions.shape[0] == 0:
            raise RuntimeError(
                "Error occured during attribution calculation. "
                + "Make sure You are applying this method to CNN network.",
            )
        return attributions

    def add_rules(self, model: torch.nn.Module) -> torch.nn.Module:
        """Add rules for the LRP explainer,
        according to https://arxiv.org/pdf/1910.09840.pdf.

        Args:
            model: DNN object to be modified.

        Returns:
            Modified DNN object.
        """
        layers_number: int = len(list(model.modules()))
        for idx_layer, module in enumerate(model.modules()):
            if idx_layer <= layers_number // 2:
                setattr(module, "rule", GammaRule())
            elif idx_layer != (layers_number - 1):
                setattr(module, "rule", EpsilonRule())
            else:
                setattr(module, "rule", EpsilonRule(epsilon=0))  # LRP-0

        return model


class LRPCVExplainer(BaseLRPCVExplainer):
    """LRP algorithm explainer."""

    def create_explainer(
        self,
        model: torch.nn.Module,
        **kwargs,
    ) -> Union[LRP, LayerLRP]:
        """Create explainer object.

        Returns:
            Explainer object.
        """
        model = self.add_rules(modify_modules(model))

        return LRP(model=model)


class LayerLRPCVExplainer(BaseLRPCVExplainer):
    """Layer LRP algorithm explainer."""

    def create_explainer(
        self,
        model: torch.nn.Module,
        layer: Optional[torch.nn.Module] = None,
        **kwargs,
    ) -> Union[LRP, LayerLRP]:
        """Create explainer object.

        Uses parameter `layer` from `kwargs`. If not provided function will call
        `get_last_conv_model_layer` function to obtain last `torch.nn.Conv2d` layer
        from provided model.

        Returns:
            Explainer object.

        Raises:
            ValueError: if model does not contain conv layers
        """
        if layer is None:
            layer = get_last_conv_model_layer(model=model)

        model = self.add_rules(modify_modules(model))

        return LayerLRP(model=model, layer=layer)
