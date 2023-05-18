"""File with GradCAM algorithm explainer classes.

Paper: https://arxiv.org/abs/1905.00780
Based on https://github.com/idiap/fullgrad-saliency

Note:
FullGradient does not support inplace operations.
ReLU inplace operations are turned off in the model automatically.
However inplace addition, multiplication etc. is the model design
choice and should be avoided in order to make the FullyGradient explainer work
correctly.

Example modification of model architecture to avoid inplace operations:
out = self.conv(x)
out = self.batch_norm(out)
identity = self.downsample(x)

out_tmp = out.clone()
if out_tmp.requires_grad:
    out_tmp.retain_grad()
out = out_tmp + identity
"""

import logging
from abc import abstractmethod
from functools import wraps
from math import isclose
from sys import exit as terminate
from typing import Any, Callable, Final, List, Optional, Tuple, TypeVar, Union, cast

import torch
import torch.nn.functional as F
from torch.nn.modules.module import _grad_t
from torch.utils.hooks import RemovableHandle

from foxai.array_utils import validate_result
from foxai.explainer.base_explainer import CVExplainer
from foxai.logger import create_logger

Submodule = TypeVar("Submodule", bound=torch.nn.Module)
"""Subclass of the torch.nn.Module"""

_LOGGER: Optional[logging.Logger] = None
"""Global logger instance."""

_INPLACE_ERROR_MSG: Final[
    str
] = """
Error possibly caused by inplace operations.
FullGradient does not support inplace operations.
ReLU inplace operations are turned off in the model automatically.
However inplace addition, multiplication etc. is the model design
choice and should be avoided in order to make the FullyGradient explainer work
correctly.

Example modification of model architecture to avoid inplace operations:

Modify from:
out += identity

Modify to:
out = out + identity
"""


def log() -> logging.Logger:
    """Get or create logger."""
    # pylint: disable = global-statement
    global _LOGGER
    if _LOGGER is None:
        _LOGGER = create_logger(__name__)
    return _LOGGER


class FullGradExtractor:
    """Extract gradient tensors needed for FullGrad computation using pytorch hooks."""

    def __init__(self, model: Submodule, im_size: Tuple[int, int, int] = (3, 224, 224)):
        self.model = model
        self.im_size: Tuple[int, int, int] = im_size

        self._biases: List[torch.Tensor] = []
        self.feature_grads: List[torch.Tensor] = []
        self.grad_handles: List[RemovableHandle] = []

        # Iterate through layers
        for module in self.model.modules():
            if hasattr(module, "inplace"):
                module.inplace = False  # type: ignore

            if isinstance(
                module, (torch.nn.Conv2d, torch.nn.Linear, torch.nn.BatchNorm2d)
            ):
                # Register feature-gradient hooks for each layer
                handle_g = module.register_full_backward_hook(self._extract_layer_grads)
                self.grad_handles.append(handle_g)

                # Collect model biases
                bias: Optional[torch.Tensor] = self._extract_layer_bias(module)
                if bias is not None:
                    self._biases.append(bias)

            if isinstance(
                module,
                (
                    torch.nn.Sigmoid,
                    torch.nn.SiLU,
                    torch.nn.Tanh,
                    torch.nn.LeakyReLU,
                    torch.nn.ELU,
                    torch.nn.SELU,
                ),
            ):
                log().warning(
                    "The FullGradients algorithm only works properly with ReLu and "
                    "linear activation functions by design."
                    f", however {module} layer found in model definition."
                )

    def _extract_layer_bias(self, module: Submodule) -> Optional[torch.Tensor]:
        """Extract bias of each layer

        For batchnorm, the overall "bias" is different from batchnorm bias parameter.
        Let m -> running mean, s -> running std
        Let w -> BN weights, b -> BN bias
        Then, ((x - m)/s)*w + b = x*w/s + (- m*w/s + b)
        Thus (-m*w/s + b) is the effective bias of batchnorm

        Args:
            module: layer module

        Returns:
            bias if present in module else None
        """

        if isinstance(module, torch.nn.BatchNorm2d):
            if module.running_var is None or module.running_mean is None:
                return None
            b = (
                -(
                    module.running_mean
                    * module.weight
                    / torch.sqrt(module.running_var + module.eps)
                )
                + module.bias
            )
            return b.data
        elif module.bias is None:
            return None
        elif isinstance(module.bias, torch.Tensor):
            return module.bias.data

        log().warning(
            "Possible error. Bias present, but of type Module, not torch.Tensor."
        )
        return None

    @property
    def biases(self) -> List[torch.Tensor]:
        """Get biases."""
        return self._biases

    def _extract_layer_grads(  # pylint: disable = unused-argument
        self, module: Submodule, in_grad: _grad_t, out_grad: _grad_t
    ) -> None:
        """Collect gradient outputs from each layer, that contains bias term.

        Args:
            module: the pytorch layer
            in_grad: input gradients
            out_grad: output gradients
        """

        if not module.bias is None:
            self.feature_grads.append(out_grad[0])

    def get_feature_grads(
        self, x: torch.Tensor, output_scalar: torch.Tensor
    ) -> Tuple[torch.Tensor, List[torch.Tensor]]:
        """Get gradients for the input and for each model layer w.r.t. input.

        Args:
            x: the input tensor
            output_scalar: the output tensor

        Returns:
            gradients for input
            gradients for each layer, that has bias term
        """

        # Empty feature grads list
        self.feature_grads = []

        self.model.zero_grad()
        # Gradients w.r.t. input
        input_gradients = torch.autograd.grad(outputs=output_scalar, inputs=x)[0]

        return input_gradients, self.feature_grads


class FullGrad:
    """
    Compute FullGrad saliency map and full gradient decomposition
    """

    def __init__(self, model: Submodule, image_size=(3, 224, 224)):
        self.model: Submodule = model
        self.im_size: Tuple[int, int, int, int] = (1,) + image_size
        self.model_ext: FullGradExtractor = FullGradExtractor(
            model=self.model, im_size=image_size
        )
        self._biases: List[torch.Tensor] = self.model_ext.biases

        cuda = next(self.model.parameters()).is_cuda
        self._device = torch.device("cuda" if cuda else "cpu")
        self.check_completeness()

    @staticmethod
    def _eval_mode(func: Callable[..., Any]):
        """Switch model to the eval mode and back to initial mode after function execution."""

        @wraps(func)
        def wrapper(self, *args, **kwargs):
            # switch all models to eval mode and save the previous state
            was_training: bool = False
            if self.model.training:
                self.model.eval()
                was_training = True

            retval = func(self, *args, **kwargs)

            # restore the original models states
            if was_training:
                self.model.train()

            return retval

        return wrapper

    @_eval_mode
    def check_completeness(self) -> bool:
        """Check if completeness property is satisfied. If not, it usually means that
        some bias gradients are not computed (e.g.: implicit biases of non-linearities).

        Returns:
            whether the completnes condition is fulfielled

        Raises:
            RuntimeError: if the model can't run inference. Usually beocuse of inplace operations.
        """

        # Random input image
        sample_input: torch.Tensor = torch.randn(self.im_size).to(self._device)

        # Get raw outputs
        try:
            raw_output = self.model(sample_input)
        except RuntimeError as e:
            log().error(e)
            log().error(_INPLACE_ERROR_MSG)
            terminate()

        # Compute full-gradients and add them up
        # if class is None, the argmax of output is chosen
        input_grad, bias_grad = self.full_gradient_decompose(
            image=sample_input, target_class=None
        )

        fullgradient_sum = (input_grad * sample_input).sum()
        for i in range(len(bias_grad)):
            fullgradient_sum += bias_grad[i].sum()

        # Compare raw output and full gradient sum
        if not isclose(raw_output.max().item(), fullgradient_sum.item(), rel_tol=1e-2):
            log().warning(
                "Completness not fullfield. Incorrect computation of bias-gradients, "
                "most probably due to using activation functions other than ReLU. The XAI map may be incorect."
                f"Raw output = {str(raw_output.max().item())} Full-gradient sum = {str(fullgradient_sum.item())}"
            )

            return False

        return True

    @_eval_mode
    def full_gradient_decompose(
        self, image: torch.Tensor, target_class: Optional[torch.Tensor] = None
    ) -> Tuple[torch.Tensor, List[torch.Tensor]]:
        """
        Compute full-gradient decomposition for an input image.

        Args:
            image: the input image to the nn
            target_class: the class for which to draw explanations. If None, the argmax of output is chosen.

        Returns:
            input image gradients (contribution of each input pixel to the final output)
            list of contributions of bias terms in each layer in the model (each layer with the bias term)
        """

        image = image.requires_grad_()
        out = self.model(image)

        if target_class is None:
            target_class = out.data.max(1, keepdim=True)[1]

        # Select the output unit corresponding to the target class
        # -1 compensates for negation in nll_loss function
        output_scalar = -1.0 * F.nll_loss(out, target_class.flatten(), reduction="sum")

        input_gradient, feature_gradients = self.model_ext.get_feature_grads(
            image, output_scalar
        )

        # Compute feature-gradients \times bias
        bias_times_gradients: List[torch.Tensor] = []
        length = len(self._biases)

        for i in range(length):

            # feature gradients are indexed backwards
            # because of backprop
            g = feature_gradients[length - 1 - i]

            # reshape bias dimensionality to match gradients
            bias_size = [1] * len(g.size())
            bias_size[1] = self._biases[i].size(0)
            b = self._biases[i].view(tuple(bias_size))

            bias_times_gradients.append(g * b.expand_as(g))

        return input_gradient, bias_times_gradients

    def _post_process(
        self, input_tensor: torch.Tensor, eps: float = 1e-6
    ) -> torch.Tensor:
        """Normalize gradient maps.

        Args:
            input_tensor: the gradient map to be normalized
            eps: epsilon

        Return:
            normalized map
        """

        # Absolute value
        input_tensor = abs(input_tensor)

        # Rescale operations to ensure gradients lie between 0 and 1
        flatin = input_tensor.view((input_tensor.size(0), -1))
        temp, _ = flatin.min(1, keepdim=True)
        input_tensor = input_tensor - temp.unsqueeze(1).unsqueeze(1)

        flatin = input_tensor.view((input_tensor.size(0), -1))
        temp, _ = flatin.max(1, keepdim=True)
        input_tensor = input_tensor / (temp.unsqueeze(1).unsqueeze(1) + eps)
        return input_tensor

    @_eval_mode
    def compute_saliency(
        self,
        input_data: torch.Tensor,
        target: Optional[Union[torch.Tensor, int]] = None,
    ) -> torch.Tensor:
        """Compute FullGrad saliency map.

        The function combines input gradient map with bias layer gradient maps.

        Args:
            input_data: the input image to the model
            target: the target class to be explained. If none the argmax class is chosen.

        Returns:
            silency map
        """
        if isinstance(target, int):
            target = torch.tensor(data=[target] * input_data.shape[0]).to(self._device)

        input_grad, bias_grad = self.full_gradient_decompose(
            input_data, target_class=target
        )

        # Input-gradient * image
        grd = input_grad * input_data
        gradient = self._post_process(grd).sum(1, keepdim=True)
        cam = gradient

        im_size = input_data.size()

        # Aggregate Bias-gradients
        for i in range(len(bias_grad)):

            # Select only Conv layers
            if len(bias_grad[i].size()) == len(im_size):
                temp = self._post_process(bias_grad[i])
                gradient = F.interpolate(
                    temp,
                    size=(im_size[2], im_size[3]),
                    mode="bilinear",
                    align_corners=True,
                )
                cam += gradient.sum(1, keepdim=True)

        return cam

    # pylint: disable = no-staticmethod-decorator
    _eval_mode = staticmethod(_eval_mode)


class BaseFullGradientCVExplainer(CVExplainer):
    """Base FullGradient algorithm explainer."""

    @abstractmethod
    def create_explainer(
        self, model: torch.nn.Module, image_size: Tuple[int, int, int]
    ) -> FullGrad:
        """Create explainer object.

        Args:
            model: The forward function of the model or any
                modification of it.
        Returns:
            Explainer object.
        """

    def calculate_features(
        self,
        model: torch.nn.Module,
        input_data: torch.Tensor,
        pred_label_idx: Optional[int] = None,
        **kwargs,
    ) -> torch.Tensor:
        """Generate features image with Full Gradients algorithm explainer.

        Args:
            model: The forward function of the model or any
                modification of it.
            input_data: Input for which attributions
                are computed. If forward_func takes a single
                tensor as input, a single input tensor should be provided.
            pred_label_idx: Output indices for
                which gradients are computed (for classification cases,
                this is usually the target class).
                If None provided, the argmax of output is chosen for explanation.

                Default: None

        Returns:
            The silency map for the Full Gradient explainer.
            Sf(x) = ψ(∇xf(x) ⊙ x) + Σl∈L Σc∈cl ψ(fb(x)c)
            ,where:
                Sf(x) is a silency map of function f (neural net), given input x
                ∇xf(x) is a derivative of f(x) with respect to input x
                L is number of layer in the neural network
                c run across channels cl of a layer l in a neural network
                fb(x)c = ∇bf(x, b) ⊙ b, so fb(x) is a derivative of f(x), with respect to bias,
                    elementwise multiplied with biases (hadmard product). For each channel c.
                    Notation f(x, b) is just to note explicit dependence on b.
                ψ is a post processing function (for example ReLu, to only display
                    positive attributes of the silency map).
        """

        image_size: Tuple[int, ...] = input_data.shape
        if len(image_size) == 4:
            image_size = image_size[1:]
        elif len(image_size) != 3:
            raise ValueError(
                f"Wrong input size. Expected C,W,H, but found {image_size}"
            )

        fullgrad = self.create_explainer(
            model=model, image_size=cast(Tuple[int, int, int], image_size)
        )
        attributions = fullgrad.compute_saliency(
            input_data=input_data,
            target=pred_label_idx,
        )
        validate_result(attributions=attributions)
        return attributions


class FullGradientCVExplainer(BaseFullGradientCVExplainer):
    """Full-Gradient algorithm explainer."""

    def create_explainer(
        self,
        model: torch.nn.Module,
        image_size: Tuple[int, int, int],
    ) -> FullGrad:
        """Create explainer object.

        Args:
            model: The forward function of the model or any
                modification of it.

        Returns:
            Explainer object.
        """
        fullgradients = FullGrad(model=model, image_size=image_size)

        return fullgradients
