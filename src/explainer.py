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
        # for single color, e.g. MNIST data copy one colour channel 3 times to simulate RGB
        if attributions.shape[1] == 1:
            attributions = attributions.expand(
                1, 3, attributions.shape[2], attributions.shape[3]
            )
        if transformed_img.shape[1] == 1:
            transformed_img = transformed_img.expand(
                1, 3, transformed_img.shape[2], transformed_img.shape[3]
            )

        # change dimension from (C x H x W) to (H x W x C)
        # where C is colour dimension, H and W are height and width dimensions
        attributions_np: np.ndarray = attributions.squeeze().cpu().detach().numpy()
        transformed_img_np: np.ndarray = (
            transformed_img.squeeze().cpu().detach().numpy()
        )
        if len(attributions.shape) >= 3:
            attributions_np = np.transpose(attributions_np, (1, 2, 0))
        if len(transformed_img.shape) >= 3:
            transformed_img_np = np.transpose(transformed_img_np, (1, 2, 0))

        figure, _ = viz.visualize_image_attr_multiple(
            attributions_np,
            transformed_img_np,
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
        **kwargs,
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
        **kwargs,
    ) -> torch.Tensor:
        """Generate features image with noise tunnel algorithm explainer.

        Args:
            model: Any DNN model You want to use.
            input_data: Input image.
            pred_label_idx: Predicted label.

        Returns:
            Features matrix.
        """
        nt_samples: int = kwargs.get("nt_samples", 10)
        nt_type: str = kwargs.get("nt_type", "smoothgrad_sq")

        integrated_gradients = IntegratedGradients(model)
        noise_tunnel = NoiseTunnel(integrated_gradients)

        attributions = noise_tunnel.attribute(
            input_data, nt_samples=nt_samples, nt_type=nt_type, target=pred_label_idx
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
        **kwargs,
    ) -> torch.Tensor:
        """Generate features image with gradient SHAP algorithm explainer.

        Args:
            model: Any DNN model You want to use.
            input_data: Input image.
            pred_label_idx: Predicted label.

        Returns:
            Features matrix.
        """
        stdevs: float = kwargs.get("stdevs", 0.0001)
        n_samples: int = kwargs.get("n_samples", 50)

        gradient_shap = GradientShap(model)

        # Defining baseline distribution of images
        rand_img_dist = torch.cat(  # pylint: disable = (no-member)
            [input_data * 0, input_data * 1]
        )

        attributions = gradient_shap.attribute(
            input_data,
            n_samples=n_samples,
            stdevs=stdevs,
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
        **kwargs,
    ) -> torch.Tensor:
        """Generate features image with occulusion algorithm explainer.

        Args:
            model: Any DNN model You want to use.
            input_data: Input image.
            pred_label_idx: Predicted label.

        Returns:
            Features matrix.
        """
        stride_value = kwargs.get("stride_value", 3)
        window_value = kwargs.get("window_value", 3)

        stride = (input_data.shape[1], stride_value, stride_value)
        sliding_window_shapes = (input_data.shape[1], window_value, window_value)
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
        **kwargs,
    ) -> torch.Tensor:
        """Generate features image with occulusion algorithm explainer.

        Args:
            model: Any DNN model You want to use.
            input_data: Input image.
            pred_label_idx: Predicted label.

        Returns:
            Features matrix.
        """
        layer_type = kwargs.get("layer_type", torch.nn.Conv2d)
        selected_layer = kwargs.get("selected_layer", -1)
        conv_layer_list = []
        for module in model.modules():
            if isinstance(module, torch.nn.ReLU):
                print("relu")
                module.inplace = False

            if isinstance(module, layer_type):
                conv_layer_list.append(module)

        guided_cam = GuidedGradCam(model, layer=conv_layer_list[selected_layer])
        attributions = guided_cam.attribute(
            input_data,
            target=pred_label_idx,
        )

        return attributions
