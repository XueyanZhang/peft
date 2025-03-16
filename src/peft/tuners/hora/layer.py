# Copyright 2024-present the HuggingFace Inc. team.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import warnings
from typing import Any, List, Optional, Union

import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers.pytorch_utils import Conv1D

from peft.tuners.tuners_utils import BaseTunerLayer, check_adapters_to_merge

import torch
import numpy as np
from scipy.fft import dctn, idctn

def ct(input_tensor, norm='ortho'):
    """
    类似于 torch.fft.fft2 的 Chebyshev 变换 (离散余弦变换)。
    支持批量输入。

    Args:
        input_tensor: torch.Tensor, 输入形状为 (batch_size, n, n) 或 (n, n)。
        norm: str, 默认为 'ortho'，定义 DCT 的归一化方式。

    Returns:
        torch.Tensor, 输出形状与输入一致。
    """
    # 将 PyTorch Tensor 转为 NumPy 数组
    input_numpy = input_tensor.detach().cpu().numpy()

    # 计算 2D 离散余弦变换
    output_numpy = dctn(input_numpy, type=2, norm=norm, axes=(-2, -1))

    # 转回 PyTorch Tensor
    output_tensor = torch.from_numpy(output_numpy).to(input_tensor.device)
    return output_tensor


def ict(input_tensor, norm='ortho'):
    """
    类似于 torch.fft.ifft2 的逆 Chebyshev 变换 (离散余弦变换的逆变换)。
    支持批量输入。

    Args:
        input_tensor: torch.Tensor, 输入形状为 (batch_size, n, n) 或 (n, n)。
        norm: str, 默认为 'ortho'，定义 DCT 的归一化方式。

    Returns:
        torch.Tensor, 输出形状与输入一致。
    """
    # 将 PyTorch Tensor 转为 NumPy 数组
    input_numpy = input_tensor.detach().cpu().numpy()

    # 计算 2D 离散余弦逆变换
    output_numpy = idctn(input_numpy, type=2, norm=norm, axes=(-2, -1))

    # 转回 PyTorch Tensor
    output_tensor = torch.from_numpy(output_numpy).to(input_tensor.device)
    return output_tensor

class HoRALayer(BaseTunerLayer):
    # All names of layers that may contain (trainable) adapter weights
    adapter_layer_names = ("hora_spectrum","hora_A")
    # All names of other parameters that may contain adapter-related parameters
    other_param_names = ("r", "dist","hora_n_frequency", "hora_scaling", "hora_random_loc_seed")

    def __init__(self, base_layer: nn.Module, **kwargs) -> None:
        self.base_layer = base_layer
        self.hora_n_frequency = {}
        # self.r = r
        self.hora_scaling = {}
        self.hora_spectrum = nn.ParameterDict({})
        self.indices = {}

        self.hora_random_loc_seed = {}
        # Mark the weight as unmerged
        self._disable_adapters = False
        self.merged_adapters = []
        self.kwargs = kwargs

        base_layer = self.get_base_layer()
        if isinstance(base_layer, nn.Linear):
            self.in_features, self.out_features = base_layer.in_features, base_layer.out_features
        elif isinstance(base_layer, Conv1D):
            self.in_features, self.out_features = (
                base_layer.weight.ds_shape if hasattr(base_layer.weight, "ds_shape") else base_layer.weight.shape
            )
        else:
            raise ValueError(f"Unsupported layer type {type(base_layer)}")

        # self.hora_A = nn.Parameter(torch.randn(self.r, self.in_features), requires_grad=True)
        # nn.init.zeros_(self.hora_A)
        # self.hora_A =  nn.ParameterDict({})
        self.hora_A = {} #todo
        # self.hora_alpha = nn.ParameterDict({})

    def update_layer(self, adapter_name, r, dist, n_frequency, scaling, init_weights, random_loc_seed):
        self.r = r
        self.dist = dist
        if n_frequency <= 0:
            raise ValueError(f"`n_frequency` should be a positive integer value but the value passed is {n_frequency}")
        if n_frequency > self.in_features * self.r:
            raise ValueError(
                f"`n_frequency` should be less than or equal to the product of the input and output dimensions "
                f"but the value passed is {n_frequency} and the product is {self.in_features * self.r}"
            )
        self.hora_n_frequency[adapter_name] = n_frequency
        self.hora_random_loc_seed[adapter_name] = random_loc_seed
        self.indices[adapter_name] = torch.randperm(
            self.r * self.out_features,
            generator=torch.Generator().manual_seed(self.hora_random_loc_seed[adapter_name]),
        )[:n_frequency]
        self.indices[adapter_name] = torch.stack(
            [self.indices[adapter_name] // self.r, self.indices[adapter_name] % self.r], dim=0
        )

        self.hora_A[adapter_name] = torch.randn(self.r, self.in_features, generator=torch.Generator().manual_seed(self.hora_random_loc_seed[adapter_name]))
        # self.hora_alpha[adapter_name] = nn.Parameter(torch.tensor(0.0),requires_grad=True)
        # nn.init.zeros_(self.hora_alpha[adapter_name])
        # print(self.indices)

        self.hora_scaling[adapter_name] = scaling

        # Actual trainable parameters
        self.hora_spectrum[adapter_name] = nn.Parameter(torch.randn(n_frequency), requires_grad=True)

        # self.hora_A[adapter_name] = nn.Parameter(torch.randn(self.r, self.in_features), requires_grad=True)        # print(self.hora_spectrum)
        # nn.init.zeros_(self.hora_A[adapter_name])
        # exit()
        if init_weights:
            self.reset_fourier_parameters(adapter_name)

        self._move_adapter_to_device_of_base_layer(adapter_name)
        self.set_adapter(self.active_adapters)

    @torch.no_grad()
    def reset_fourier_parameters(self, adapter_name):
        if adapter_name in self.hora_spectrum.keys():
            nn.init.zeros_(self.hora_spectrum[adapter_name])

    def get_delta_weight(self, adapter) -> torch.Tensor:
        spectrum = self.hora_spectrum[adapter]
        indices = self.indices[adapter].to(spectrum.device)
        hora_A = self.hora_A[adapter].to(spectrum.device)

        dense_spectrum = torch.zeros(self.out_features,self.r, device=spectrum.device, dtype=spectrum.dtype)

        dense_spectrum[indices[0, :], indices[1, :]] = spectrum

        chunk_size = (self.out_features + self.dist - 1) // self.dist  # 计算每个块的大小，向上取整
        chunks = []

        for i in range(self.dist):
            # 计算当前块的起始和结束索引
            start_idx = i * chunk_size
            end_idx = min(start_idx + chunk_size, self.out_features)

            # 获取当前块并执行 IFFT
            chunk = dense_spectrum[start_idx:end_idx]
            ifft_chunk = torch.fft.ifft2(chunk).real  # 在最后一维执行 IFFT

            # 保存处理后的块
            chunks.append(ifft_chunk)

        # 将所有块沿第一维合并
        B = torch.cat(chunks, dim=0)
        # print(torch.max(B))
        # print(adapter)
        # print(B)
        # print(B.shape)


        # delta_weight = ct(dense_spectrum) * self.hora_scaling[adapter]
        # print(delta_weight)
        # delta_weight = torch.fft.ifft2(dense_spectrum).real * self.hora_scaling[adapter]
        # delta_weight = torch.fft.ifft2(dense_spectrum).real * self.hora_scaling[adapter]*A #不如mm
        delta_weight = torch.matmul(B * self.hora_scaling[adapter], hora_A) # mm(fft,A)效果比较好 cola可以到64.6
        # exit()
        # delta_weight = torch.fft.ifft2(torch.conj(torch.fft.fft2(dense_spectrum)) * torch.fft.fft2(A)).real * self.hora_scaling[adapter]
        return delta_weight


class HoRALinear(nn.Module, HoRALayer):
    # HoRA implemented in a dense layer
    def __init__(
        self,
        base_layer,
        adapter_name: str,
        r:int,
        dist: int,
        n_frequency: int = 1000,
        scaling: float = 150.0,
        fan_in_fan_out: bool = False,  # Set this to True if the layer to replace stores weight like (fan_in, fan_out)
        init_weights: Union[bool, str] = False,
        random_loc_seed: int = 777,
        **kwargs,
    ) -> None:
        super().__init__()
        HoRALayer.__init__(self, base_layer, **kwargs)
        self.fan_in_fan_out = fan_in_fan_out
        self._active_adapter = adapter_name
        self.update_layer(adapter_name, r=r, dist=dist, n_frequency = n_frequency, scaling=scaling, init_weights=init_weights, random_loc_seed=random_loc_seed)

    def merge(self, safe_merge: bool = False, adapter_names: Optional[List[str]] = None) -> None:
        """
        Merge the active adapter weights into the base weights

        Args:
            safe_merge (`bool`, *optional*):
                If True, the merge operation will be performed in a copy of the original weights and check for NaNs
                before merging the weights. This is useful if you want to check if the merge operation will produce
                NaNs. Defaults to `False`.
            adapter_names (`List[str]`, *optional*):
                The list of adapter names that should be merged. If None, all active adapters will be merged. Defaults
                to `None`.
        """
        adapter_names = check_adapters_to_merge(self, adapter_names)
        if not adapter_names:
            # no adapter to merge
            return

        for active_adapter in adapter_names:
            if active_adapter in self.hora_spectrum.keys():
                base_layer = self.get_base_layer()
                if safe_merge:
                    # Note that safe_merge will be slower than the normal merge
                    # because of the copy operation.
                    orig_weights = base_layer.weight.data.clone()
                    orig_weights += self.get_delta_weight(active_adapter)

                    if not torch.isfinite(orig_weights).all():
                        raise ValueError(
                            f"NaNs detected in the merged weights. The adapter {active_adapter} seems to be broken"
                        )

                    base_layer.weight.data = orig_weights
                else:
                    base_layer.weight.data += self.get_delta_weight(active_adapter)
                self.merged_adapters.append(active_adapter)

    def unmerge(self) -> None:
        """
        This method unmerges all merged adapter layers from the base weights.
        """
        if not self.merged:
            warnings.warn("Already unmerged. Nothing to do.")
            return
        while len(self.merged_adapters) > 0:
            active_adapter = self.merged_adapters.pop()
            if active_adapter in self.hora_spectrum.keys():
                self.get_base_layer().weight.data -= self.get_delta_weight(active_adapter)

    def get_delta_weight(self, adapter) -> torch.Tensor:
        return super().get_delta_weight(adapter)

    def forward(self, x: torch.Tensor, *args: Any, **kwargs: Any) -> torch.Tensor:
        previous_dtype = x.dtype

        if self.disable_adapters:
            if self.merged:
                self.unmerge()
            result = self.base_layer(x, *args, **kwargs)
        elif self.merged:
            result = self.base_layer(x, *args, **kwargs)
        else:
            result = self.base_layer(x, *args, **kwargs)
            for active_adapter in self.active_adapters:
                if active_adapter not in self.hora_spectrum.keys():
                    continue

                delta_w = self.get_delta_weight(active_adapter)
                x = x.to(delta_w.dtype)
                result = result + F.linear(x, delta_w)

        result = result.to(previous_dtype)
        return result

    def __repr__(self) -> str:
        rep = super().__repr__()
        return "hora." + rep
