from typing import Callable, Tuple, Union

import torch

CustomAttributionFuncType = Callable[..., Tuple[torch.Tensor, ...]]
StdevsType = Union[float, Tuple[float, ...]]

LayerType = torch.nn.Module
ModelType = torch.nn.Module
BboxType = Tuple[int, int, int, int]

AttributionsType = torch.Tensor
