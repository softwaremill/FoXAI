"""File with GradCAM algorithm explainer classes.

Based on https://github.com/pytorch/captum/blob/master/captum/attr/_core/guided_grad_cam.py
and https://github.com/pytorch/captum/blob/master/captum/attr/_core/layer/grad_cam.py.
"""

from abc import abstractmethod
from typing import Any, Dict, List, Optional, Union

import torch
import torch.nn.functional as F
from captum._utils.typing import TargetType
from captum.attr import GuidedGradCam

from foxai.array_utils import validate_result
from foxai.explainer.base_explainer import Explainer
from foxai.explainer.computer_vision.model_utils import (
    get_last_conv_model_layer,
    modify_modules,
)
from foxai.explainer.computer_vision.object_detection.base_object_detector import (
    BaseObjectDetector,
)
from foxai.explainer.computer_vision.object_detection.types import ObjectDetectionOutput
from foxai.types import AttributionsType, LayerType, ModelType


class LayerBaseGradCAM:
    """Layer GradCAM for object detection task."""

    def __init__(
        self,
        model: ModelType,
        target_layer: LayerType,
    ):
        self.model = model
        self._gradients: Dict[str, torch.Tensor] = {}
        self._activations: Dict[str, torch.Tensor] = {}
        self.target_layer = target_layer

        def backward_hook(
            module,  # pylint: disable = (unused-argument)
            grad_input,  # pylint: disable = (unused-argument)
            grad_output,
        ):
            self._gradients["value"] = grad_output[0]

        def forward_hook(
            module,  # pylint: disable = (unused-argument)
            input,  # pylint: disable = (unused-argument,redefined-builtin)
            output,
        ):
            self._activations["value"] = output

        self.target_layer.register_forward_hook(forward_hook)
        self.target_layer.register_backward_hook(backward_hook)

    @property
    def activations(self) -> torch.Tensor:
        return self._activations["value"]

    @property
    def gradients(self) -> torch.Tensor:
        return self._gradients["value"]

    def forward(
        self,
        input_img: torch.Tensor,
    ) -> Union[torch.Tensor, ObjectDetectionOutput]:
        """Forward pass of GradCAM algorithm.

        Args:
            input_img: Input image with shape of (B, C, H, W).

        Returns:
            ObjectDetectionOutput object for object detection and tensor with saliency
            map for classification.
        """
        saliency_maps: List[torch.Tensor] = []
        _, _, height, width = input_img.size()

        result_list: List[torch.Tensor] = self.model.forward(input_img)

        result = torch.vstack(tensors=tuple(result_list))
        score = result.max()

        # clear gradients
        self.model.zero_grad()

        # calculate gradients
        score.backward(retain_graph=True)

        saliency_maps.append(
            self.get_saliency_map(
                height=height,
                width=width,
                gradients=self.gradients,
                activations=self.activations,
            )
        )
        return torch.cat(saliency_maps)

    def get_saliency_map(
        self,
        height: int,
        width: int,
        gradients: torch.Tensor,
        activations: torch.Tensor,
    ) -> torch.Tensor:
        """Generate saliency map.

        Args:
            height: Original image height.
            width: Original image width.
            gradients: Layer gradients.
            activations: Layer activations.

        Returns:
            Saliency map.
        """
        b, k, _, _ = gradients.size()
        alpha = gradients.view(b, k, -1).mean(2)
        weights = alpha.view(b, k, 1, 1)
        saliency_map = (weights * activations).sum(1, keepdim=True)
        saliency_map = F.relu(saliency_map)
        saliency_map = F.upsample(
            saliency_map, size=(height, width), mode="bilinear", align_corners=False
        )
        saliency_map_min, saliency_map_max = saliency_map.min(), saliency_map.max()
        saliency_map = (
            (saliency_map - saliency_map_min)
            .div(saliency_map_max - saliency_map_min)
            .data
        )
        return saliency_map

    def __call__(
        self,
        input_img: torch.Tensor,
    ) -> Union[torch.Tensor, ObjectDetectionOutput]:
        return self.forward(input_img)


class BaseGradCAMCVExplainer(Explainer):
    """Base GradCAM algorithm explainer."""

    @abstractmethod
    def create_explainer(
        self, model: ModelType, layer: LayerType, **kwargs
    ) -> Union[GuidedGradCam, LayerBaseGradCAM]:
        """Create explainer object.

        Args:
            model: The forward function of the model or any
                modification of it.
            layer: Layer for which attributions are computed.
                Output size of attribute matches this layer's input or
                output dimensions, depending on whether we attribute to
                the inputs or outputs of the layer, corresponding to
                attribution of each neuron in the input or output of
                this layer.

        Returns:
            Explainer object.
        """

    @abstractmethod
    def calculate_features(
        self,
        model: ModelType,
        input_data: torch.Tensor,
        pred_label_idx: TargetType = None,
        additional_forward_args: Any = None,
        attribute_to_layer_input: bool = False,
        **kwargs,
    ) -> AttributionsType:
        """Generate features image with GradCAM algorithm explainer.

        Args:
            model: The forward function of the model or any
                modification of it.
            input_data: Input for which attributions
                are computed. If forward_func takes a single
                tensor as input, a single input tensor should be provided.
            pred_label_idx: Output indices for
                which gradients are computed (for classification cases,
                this is usually the target class).
                If the network returns a scalar value per example,
                no target index is necessary.
                For general 2D outputs, targets can be either:

                - a single integer or a tensor containing a single
                    integer, which is applied to all input examples

                - a list of integers or a 1D tensor, with length matching
                    the number of examples in inputs (dim 0). Each integer
                    is applied as the target for the corresponding example.

                For outputs with > 2 dimensions, targets can be either:

                - A single tuple, which contains #output_dims - 1
                    elements. This target index is applied to all examples.

                - A list of tuples with length equal to the number of
                    examples in inputs (dim 0), and each tuple containing
                    #output_dims - 1 elements. Each tuple is applied as the
                    target for the corresponding example.
                Default: None
            additional_forward_args: If the forward function
                requires additional arguments other than the inputs for
                which attributions should not be computed, this argument
                can be provided. It must be either a single additional
                argument of a Tensor or arbitrary (non-tuple) type or a
                tuple containing multiple additional arguments including
                tensors or any arbitrary python types. These arguments
                are provided to forward_func in order following the
                arguments in inputs.
                Note that attributions are not computed with respect
                to these arguments.
                Default: None
            attribute_to_layer_input: Indicates whether to
                compute the attribution with respect to the layer input
                or output in `LayerGradCam`.
                If `attribute_to_layer_input` is set to True
                then the attributions will be computed with respect to
                layer inputs, otherwise it will be computed with respect
                to layer outputs.
                Note that currently it is assumed that either the input
                or the output of internal layer, depending on whether we
                attribute to the input or output, is a single tensor.
                Support for multiple tensors will be added later.
                Default: False

        Returns:
            Element-wise product of (upsampled) GradCAM
            and/or Guided Backprop attributions.
            If a single tensor is provided as inputs, a single tensor is
            returned. If a tuple is provided for inputs, a tuple of
            corresponding sized tensors is returned.
            Attributions will be the same size as the provided inputs,
            with each value providing the attribution of the
            corresponding input index.
            If the GradCAM attributions cannot be upsampled to the shape
            of a given input tensor, None is returned in the corresponding
            index position.
        """


class GuidedGradCAMCVExplainer(BaseGradCAMCVExplainer):
    """GuidedGradCAM algorithm explainer."""

    def create_explainer(
        self,
        model: ModelType,
        layer: LayerType,
        **kwargs,
    ) -> Union[GuidedGradCam, LayerBaseGradCAM]:
        """Create explainer object.

        Args:
            model: The forward function of the model or any
                modification of it.
            layer: Layer for which attributions are computed.
                Output size of attribute matches this layer's input or
                output dimensions, depending on whether we attribute to
                the inputs or outputs of the layer, corresponding to
                attribution of each neuron in the input or output of
                this layer.

        Returns:
            Explainer object.
        """
        model = modify_modules(model)

        return GuidedGradCam(model=model, layer=layer)

    def calculate_features(
        self,
        model: ModelType,
        input_data: torch.Tensor,
        pred_label_idx: TargetType = None,
        additional_forward_args: Any = None,
        attribute_to_layer_input: bool = False,
        interpolate_mode: str = "nearest",
        layer: Optional[LayerType] = None,
        **kwargs,
    ) -> AttributionsType:
        """Generate model's attributes with GradCAM algorithm explainer.

        Args:
            model: The forward function of the model or any
                modification of it.
            input_data: Input for which attributions
                are computed. If forward_func takes a single
                tensor as input, a single input tensor should be provided.
            pred_label_idx: Output indices for
                which gradients are computed (for classification cases,
                this is usually the target class).
                If the network returns a scalar value per example,
                no target index is necessary.
                For general 2D outputs, targets can be either:

                - a single integer or a tensor containing a single
                    integer, which is applied to all input examples

                - a list of integers or a 1D tensor, with length matching
                    the number of examples in inputs (dim 0). Each integer
                    is applied as the target for the corresponding example.

                For outputs with > 2 dimensions, targets can be either:

                - A single tuple, which contains #output_dims - 1
                    elements. This target index is applied to all examples.

                - A list of tuples with length equal to the number of
                    examples in inputs (dim 0), and each tuple containing
                    #output_dims - 1 elements. Each tuple is applied as the
                    target for the corresponding example.
                Default: None
            additional_forward_args: If the forward function
                requires additional arguments other than the inputs for
                which attributions should not be computed, this argument
                can be provided. It must be either a single additional
                argument of a Tensor or arbitrary (non-tuple) type or a
                tuple containing multiple additional arguments including
                tensors or any arbitrary python types. These arguments
                are provided to forward_func in order following the
                arguments in inputs.
                Note that attributions are not computed with respect
                to these arguments.
                Default: None
            interpolate_mode: Method for interpolation, which
                must be a valid input interpolation mode for
                torch.nn.functional. These methods are
                "nearest", "area", "linear" (3D-only), "bilinear"
                (4D-only), "bicubic" (4D-only), "trilinear" (5D-only)
                based on the number of dimensions of the chosen layer
                output (which must also match the number of
                dimensions for the input tensor). Note that
                the original GradCAM paper uses "bilinear"
                interpolation, but we default to "nearest" for
                applicability to any of 3D, 4D or 5D tensors.
                Default: "nearest"
            attribute_to_layer_input: Indicates whether to
                compute the attribution with respect to the layer input
                or output in `LayerGradCam`.
                If `attribute_to_layer_input` is set to True
                then the attributions will be computed with respect to
                layer inputs, otherwise it will be computed with respect
                to layer outputs.
                Note that currently it is assumed that either the input
                or the output of internal layer, depending on whether we
                attribute to the input or output, is a single tensor.
                Support for multiple tensors will be added later.
                Default: False

        Returns:
            Element-wise product of (upsampled) GradCAM
            and/or Guided Backprop attributions.
            If a single tensor is provided as inputs, a single tensor is
            returned. If a tuple is provided for inputs, a tuple of
            corresponding sized tensors is returned.
            Attributions will be the same size as the provided inputs,
            with each value providing the attribution of the
            corresponding input index.
            If the GradCAM attributions cannot be upsampled to the shape
            of a given input tensor, None is returned in the corresponding
            index position.

        Raises:
            ValueError: if model does not contain conv layers.
            RuntimeError: if attributions has shape (0)
        """
        attributions: AttributionsType
        if layer is None:
            layer = get_last_conv_model_layer(model=model)

        guided_cam = self.create_explainer(model=model, layer=layer)

        if isinstance(guided_cam, GuidedGradCam):
            attributions = guided_cam.attribute(
                input_data,
                target=pred_label_idx,
                additional_forward_args=additional_forward_args,
                interpolate_mode=interpolate_mode,
                attribute_to_layer_input=attribute_to_layer_input,
            )
        validate_result(attributions=attributions)
        return attributions


class LayerGradCAMCVExplainer(BaseGradCAMCVExplainer):
    """Layer GradCAM algorithm explainer."""

    def create_explainer(
        self,
        model: ModelType,
        layer: LayerType,
        **kwargs,
    ) -> Union[GuidedGradCam, LayerBaseGradCAM]:
        """Create explainer object.

        Args:
            model: The forward function of the model or any
                modification of it.
            layer: Layer for which attributions are computed.
                Output size of attribute matches this layer's input or
                output dimensions, depending on whether we attribute to
                the inputs or outputs of the layer, corresponding to
                attribution of each neuron in the input or output of
                this layer.

        Returns:
            Explainer object.
        """
        model = modify_modules(model)

        return LayerBaseGradCAM(model=model, target_layer=layer)

    def calculate_features(
        self,
        model: ModelType,
        input_data: torch.Tensor,
        pred_label_idx: TargetType = None,  # pylint: disable = (unused-argument)
        additional_forward_args: Any = None,  # pylint: disable = (unused-argument)
        attribute_to_layer_input: bool = False,  # pylint: disable = (unused-argument)
        relu_attributions: bool = False,  # pylint: disable = (unused-argument)
        layer: Optional[LayerType] = None,
        **kwargs,
    ) -> AttributionsType:
        """Generate features image with GradCAM algorithm explainer.

        Args:
            model: The forward function of the model or any
                modification of it.
            input_data: Input for which attributions
                are computed. If forward_func takes a single
                tensor as input, a single input tensor should be provided.
                If forward_func takes multiple tensors as input, a tuple
                of the input tensors should be provided. It is assumed
                that for all given input tensors, dimension 0 corresponds
                to the number of examples, and if multiple input tensors
                are provided, the examples must be aligned appropriately.
            pred_label_idx: Output indices for
                which gradients are computed (for classification cases,
                this is usually the target class).
                If the network returns a scalar value per example,
                no target index is necessary.
                For general 2D outputs, targets can be either:

                - a single integer or a tensor containing a single
                    integer, which is applied to all input examples

                - a list of integers or a 1D tensor, with length matching
                    the number of examples in inputs (dim 0). Each integer
                    is applied as the target for the corresponding example.

                For outputs with > 2 dimensions, targets can be either:

                - A single tuple, which contains #output_dims - 1
                    elements. This target index is applied to all examples.

                - A list of tuples with length equal to the number of
                    examples in inputs (dim 0), and each tuple containing
                    #output_dims - 1 elements. Each tuple is applied as the
                    target for the corresponding example.
                Default: None
            additional_forward_args: If the forward function
                requires additional arguments other than the inputs for
                which attributions should not be computed, this argument
                can be provided. It must be either a single additional
                argument of a Tensor or arbitrary (non-tuple) type or a
                tuple containing multiple additional arguments including
                tensors or any arbitrary python types. These arguments
                are provided to forward_func in order following the
                arguments in inputs.
                Note that attributions are not computed with respect
                to these arguments.
                Default: None
            attribute_to_layer_input: Indicates whether to
                compute the attribution with respect to the layer input
                or output in `LayerGradCam`.
                If `attribute_to_layer_input` is set to True
                then the attributions will be computed with respect to
                layer inputs, otherwise it will be computed with respect
                to layer outputs.
                Note that currently it is assumed that either the input
                or the output of internal layer, depending on whether we
                attribute to the input or output, is a single tensor.
                Support for multiple tensors will be added later.
                Default: False
            relu_attributions: Indicates whether to
                apply a ReLU operation on the final attribution,
                returning only non-negative attributions. Setting this
                flag to True matches the original GradCAM algorithm,
                otherwise, by default, both positive and negative
                attributions are returned.
                Default: False
            layer: Layer for which attributions are computed.
                If None provided, last convolutional layer from the model
                is taken.
                Default: None

        Returns:
            Element-wise product of (upsampled) GradCAM
            and/or Guided Backprop attributions.
            If a single tensor is provided as inputs, a single tensor is
            returned. If a tuple is provided for inputs, a tuple of
            corresponding sized tensors is returned.
            Attributions will be the same size as the provided inputs,
            with each value providing the attribution of the
            corresponding input index.
            If the GradCAM attributions cannot be upsampled to the shape
            of a given input tensor, None is returned in the corresponding
            index position.

        Raises:
            ValueError: if model does not contain conv layers.
            RuntimeError: if attributions has shape (0)
        """
        if layer is None:
            layer = get_last_conv_model_layer(model=model)

        gradcam = self.create_explainer(model=model, layer=layer)
        if isinstance(gradcam, LayerBaseGradCAM):
            attributions = gradcam(input_data)

        if not isinstance(attributions, AttributionsType):
            raise RuntimeError(
                f"Saliency map is `{type(attributions)}`, but expected type is `AttributionsType`."
            )

        validate_result(attributions=attributions)
        return attributions


class LayerGradCAMObjectDetectionExplainer(LayerBaseGradCAM):
    """Layer GradCAM for object detection task.

    Code based on https://github.com/pooya-mohammadi/yolov5-gradcam.
    """

    def __init__(
        self,
        model: BaseObjectDetector,
        target_layer: LayerType,
    ):
        super().__init__(model=model, target_layer=target_layer)

    def forward(
        self,
        input_img: torch.Tensor,
    ) -> ObjectDetectionOutput:
        """Forward pass of GradCAM algorithm.

        Args:
            input_img: Input image with shape of (B, C, H, W).

        Returns:
            ObjectDetectionOutput object.
        """
        saliency_maps: List[torch.Tensor] = []
        _, _, height, width = input_img.size()
        predictions, logits = self.model.forward(input_img)
        for logit, cls in zip(logits[0], [p.class_number for p in predictions]):
            score = logit[cls]
            # clear gradients
            self.model.zero_grad()

            # calculate gradients
            score.backward(retain_graph=True)

            saliency_maps.append(
                self.get_saliency_map(
                    height=height,
                    width=width,
                    gradients=self.gradients,
                    activations=self.activations,
                )
            )
        return ObjectDetectionOutput(
            saliency_maps=saliency_maps,
            logits=logits,
            predictions=predictions,
        )
