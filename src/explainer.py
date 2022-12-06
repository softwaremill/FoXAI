"""Supported explainable algorithms classes."""
from abc import ABC
from typing import Any

import matplotlib
import numpy as np
import torch
from captum.attr import (
    LRP,
    GradientShap,
    GuidedGradCam,
    IntegratedGradients,
    NoiseTunnel,
    Occlusion,
)
from captum.attr import visualization as viz


class CVExplainer(ABC):
    """Abstract explainer class."""

    algorithm_name = ""

    def calculate_features(
        self,
        model: Any,  # pylint: disable=unused-argument
        input_data: torch.Tensor,  # pylint: disable=unused-argument
        pred_label_idx: torch.Tensor,  # pylint: disable=unused-argument
        *args,  # pylint: disable=unused-argument
        **kwargs,  # pylint: disable=unused-argument
    ) -> torch.Tensor:
        """Calculate features of given explainer.

        Args:
            model: Any DNN model You want to use.
            input_data: Input image.
            pred_label_idx: Predicted label.

        Returns:
            Tensor of attributes.
        """

    def visualize(
        self, attributions: torch.Tensor, transformed_img: torch.Tensor
    ) -> matplotlib.pyplot.Figure:
        """Create image with calculated features.

        Args:
            attributions: Features.
            transformed_img: Image.

        Returns:
            Image with paired figures: original image and features heatmap.
        """
        print(attributions.shape)
        print(transformed_img.shape)
        figure, _ = viz.visualize_image_attr_multiple(
            np.transpose(attributions.squeeze().cpu().detach().numpy(), (1, 2, 0)),
            np.transpose(transformed_img.squeeze().cpu().detach().numpy(), (1, 2, 0)),
            ["original_image", "heat_map"],
            ["all", "positive"],
            show_colorbar=True,
            outlier_perc=2,
            use_pyplot=False,
        )

        return figure


class IntegratedGradientsCVExplainer(CVExplainer):
    """Integrated Gradients algorithm explainer."""

    algorithm_name = "integrated_gradient"

    def calculate_features(
        self,
        model: Any,
        input_data: torch.Tensor,
        pred_label_idx: torch.Tensor,
        *args,  # pylint: disable=unused-argument
        **kwargs,  # pylint: disable=unused-argument
    ) -> torch.Tensor:
        """Generate features image with integrated gradients algorithm explainer.

        Args:
            model: Any DNN model You want to use.
            input_data: Input image.
            pred_label_idx: Predicted label.

        Returns:
            Features matrix.
        """
        n_steps = kwargs.get("n_steps", 100)

        integrated_gradients = IntegratedGradients(model)
        attributions = integrated_gradients.attribute(
            input_data, target=pred_label_idx, n_steps=n_steps
        )
        return attributions


class NoiseTunnelCVExplainer(CVExplainer):
    """Noise Tunnel algorithm explainer."""

    algorithm_name = "noise_tunnel"

    def calculate_features(
        self,
        model: Any,
        input_data: torch.Tensor,
        pred_label_idx: torch.Tensor,
        *args,  # pylint: disable=unused-argument
        **kwargs,  # pylint: disable=unused-argument
    ) -> torch.Tensor:
        """Generate features image with noise tunnel algorithm explainer.

        Args:
            model: Any DNN model You want to use.
            input_data: Input image.
            pred_label_idx: Predicted label.

        Returns:
            Features matrix.
        """
        integrated_gradients = IntegratedGradients(model)
        noise_tunnel = NoiseTunnel(integrated_gradients)

        attributions = noise_tunnel.attribute(
            input_data, nt_samples=10, nt_type="smoothgrad_sq", target=pred_label_idx
        )
        return attributions


class GradientSHAPCVExplainer(CVExplainer):
    """Gradient SHAP algorithm explainer."""

    algorithm_name = "gradient_shap"

    def calculate_features(
        self,
        model: Any,
        input_data: torch.Tensor,
        pred_label_idx: torch.Tensor,
        *args,  # pylint: disable=unused-argument
        **kwargs,  # pylint: disable=unused-argument
    ) -> torch.Tensor:
        """Generate features image with gradient SHAP algorithm explainer.

        Args:
            model: Any DNN model You want to use.
            input_data: Input image.
            pred_label_idx: Predicted label.

        Returns:
            Features matrix.
        """
        torch.manual_seed(0)
        np.random.seed(0)

        gradient_shap = GradientShap(model)

        # Defining baseline distribution of images
        rand_img_dist = torch.cat(  # pylint: disable = (no-member)
            [input_data * 0, input_data * 1]
        )

        attributions = gradient_shap.attribute(
            input_data,
            n_samples=50,
            stdevs=0.0001,
            baselines=rand_img_dist,
            target=pred_label_idx,
        )
        return attributions


class OcculusionCVExplainer(CVExplainer):
    """Occulusion algorithm explainer."""

    algorithm_name = "occulusion"

    def calculate_features(
        self,
        model: Any,
        input_data: torch.Tensor,
        pred_label_idx: torch.Tensor,
        *args,  # pylint: disable=unused-argument
        **kwargs,  # pylint: disable=unused-argument
    ) -> torch.Tensor:
        """Generate features image with occulusion algorithm explainer.

        Args:
            model: Any DNN model You want to use.
            input_data: Input image.
            pred_label_idx: Predicted label.

        Returns:
            Features matrix.
        """
        stride = kwargs.get("stride", (3, 8, 8))
        sliding_window_shapes = kwargs.get("sliding_window_shapes", (3, 15, 15))
        occlusion = Occlusion(model)

        attributions = occlusion.attribute(
            input_data,
            strides=stride,
            target=pred_label_idx,
            sliding_window_shapes=sliding_window_shapes,
            baselines=0,
        )
        return attributions


class LRPCVExplainer(CVExplainer):
    """LRP algorithm explainer."""

    algorithm_name = "lrp"

    def calculate_features(
        self,
        model: Any,
        input_data: torch.Tensor,
        pred_label_idx: torch.Tensor,
        *args,  # pylint: disable=unused-argument
        **kwargs,  # pylint: disable=unused-argument
    ) -> torch.Tensor:
        """Generate features image with occulusion algorithm explainer.

        Args:
            model: Any DNN model You want to use.
            input_data: Input image.
            pred_label_idx: Predicted label.

        Returns:
            Features matrix.
        """
        lrp = LRP(model)

        for module in model.modules():
            if isinstance(module, torch.nn.ReLU):
                module.inplace = False

        attributions = lrp.attribute(
            input_data,
            target=pred_label_idx,
        )
        return attributions


class GuidedGradCamCVExplainer(CVExplainer):
    """Guided GradCAM algorithm explainer."""

    algorithm_name = "guided_gradcam"

    def calculate_features(
        self,
        model: Any,
        input_data: torch.Tensor,
        pred_label_idx: torch.Tensor,
        *args,  # pylint: disable=unused-argument
        **kwargs,  # pylint: disable=unused-argument
    ) -> torch.Tensor:
        """Generate features image with occulusion algorithm explainer.

        Args:
            model: Any DNN model You want to use.
            input_data: Input image.
            pred_label_idx: Predicted label.

        Returns:
            Features matrix.
        """
        selected_layer = kwargs.get("selected_layer", -1)
        conv_layer_list = []
        for module in model.modules():
            if isinstance(module, torch.nn.ReLU):
                module.inplace = False

            if isinstance(module, torch.nn.Conv2d):
                conv_layer_list.append(module)

        guided_cam = GuidedGradCam(model, layer=conv_layer_list[selected_layer])
        attributions = guided_cam.attribute(
            input_data,
            target=pred_label_idx,
        )

        return attributions
