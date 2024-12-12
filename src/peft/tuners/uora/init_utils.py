from typing import Union
import torch
import torch.nn as nn
import math
from torch.nn.init import _calculate_correct_fan

def _kaiming_init(
    tensor_or_shape: Union[torch.Tensor, tuple[int, ...]],
    generator: torch.Generator= None,
) -> torch.Tensor:
    if isinstance(tensor_or_shape, tuple):
        tensor = torch.empty(tensor_or_shape)
    else:
        tensor = tensor_or_shape
    fan = _calculate_correct_fan(tensor, "fan_in")
    gain = math.sqrt(2)
    std = gain / math.sqrt(fan)
    bound = math.sqrt(3.0) * std

    with torch.no_grad():
        return tensor.uniform_(-bound, bound, generator=generator)

def _random_init(
    tensor_or_shape: Union[torch.Tensor, tuple[int, ...]],
    generator: torch.Generator= None,
) -> torch.Tensor:
    if isinstance(tensor_or_shape, tuple):
        tensor = torch.empty(tensor_or_shape)
    else:
        tensor = tensor_or_shape

    with torch.no_grad():
        return tensor.uniform_(-1, 1, generator=generator)

def _xavier_init(
    tensor_or_shape: Union[torch.Tensor, tuple[int, ...]],
    generator: torch.Generator= None,
) -> torch.Tensor:
    if isinstance(tensor_or_shape, tuple):
        tensor = torch.empty(tensor_or_shape)
    else:
        tensor = tensor_or_shape

    with torch.no_grad():
        return nn.init.xavier_uniform_(tensor, generator=generator)

def _orthogonal_init(
    tensor_or_shape: Union[torch.Tensor, tuple[int, ...]],
    generator: torch.Generator= None,
) -> torch.Tensor:
    if isinstance(tensor_or_shape, tuple):
        tensor = torch.empty(tensor_or_shape)
    else:
        tensor = tensor_or_shape

    with torch.no_grad():
        return nn.init.orthogonal_(tensor, generator=generator)