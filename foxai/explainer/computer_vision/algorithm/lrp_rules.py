from abc import ABC, abstractmethod
from typing import Optional

import torch
from torch.nn import Module, Parameter


class PropagationRule(ABC):
    """
    Base class for all propagation rule classes, also called Z-Rule.
    epsilon is used to assure that no zero divison occurs.
    """

    def __init__(self, epsilon=1e-9) -> None:
        self.epsilon = epsilon
        self.relevance_input: Optional[torch.Tensor] = None
        self.relevance_output: Optional[torch.Tensor] = None

    def create_backward_hook_input(self, input_tensor: torch.Tensor):
        def _backward_hook_input(grad):
            relevance = grad * input_tensor
            self.relevance_input = relevance.data
            return relevance

        return _backward_hook_input

    def create_backward_hook_output(self, outputs):
        def _backward_hook_output(grad):
            sign = torch.sign(outputs)
            sign[sign == 0] = 1
            relevance = grad / (outputs + sign * self.epsilon)
            self.relevance_output = grad.data
            return relevance

        return _backward_hook_output

    @abstractmethod
    def modify_weights(self, module: Module, input_tensor: torch.Tensor) -> None:
        raise NotImplementedError


class EpsilonRule(PropagationRule):
    """
    Rule for relevance propagation using a small value of epsilon
    to avoid numerical instabilities and remove noise.

    Use for middle layers.
    """

    def modify_weights(self, module: Module, input_tensor: torch.Tensor) -> None:
        pass


class GammaRule(PropagationRule):
    """
    Gamma rule for relevance propagation, gives more importance to
    positive relevance.

    Use for lower layers.

    Args:
        gamma (float): The gamma parameter determines by how much
        the positive relevance is increased.
    """

    def __init__(self, gamma=0.25, epsilon=1e-9) -> None:
        super().__init__(epsilon=epsilon)
        self.gamma = gamma

    def modify_weights(self, module: Module, input_tensor: torch.Tensor) -> None:
        if hasattr(module, "weight"):
            clamped_weight: torch.Tensor = module.weight.clamp(min=0)  # type: ignore[operator]
            module.weight = Parameter(
                torch.add(module.weight, clamped_weight, alpha=self.gamma)  # type: ignore[arg-type]
            )
