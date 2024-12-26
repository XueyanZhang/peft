# Copyright 2023-present the HuggingFace Inc. team.
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
from __future__ import annotations

import warnings
from dataclasses import dataclass, field
from typing import Optional, Union

from peft.config import PeftConfig
from peft.utils import PeftType


@dataclass
class UoraConfig(PeftConfig):
    """
    This is the configuration class to store the configuration of a [`UoraModel`].

    Paper: https://arxiv.org/abs/2310.11454.

    Args:
        r (`int`, *optional*, defaults to `256`):
            UoRA parameter dimension ("rank"). Choose higher values than LoRA ranks here, since UoRA uses far fewer
            parameters than LoRA (see Table 1).
    """

    r: int = field(
        default=256, metadata={"help": "Uora attention dimension"}
    )
    target_modules: Optional[Union[list[str], str]] = field(
        default=None, metadata={"help": "Modules"}
    )
    projection_prng_key: int = field(
        default=0, metadata={"help": "PRNG"}
    )
    save_projection: bool = field(
        default=True, metadata={"help": "Save"}
    )
    uora_dropout: float = field(
        default=0.0, metadata={"help": "Dropout"}
    )
    d_initial: float = field(
        default=0.1, metadata={"help": "Initial"}
    )
    fan_in_fan_out: bool = field(
        default=False, metadata={"help": "Fan"}
    )
    bias: str = field(
        default="none", metadata={"help": "Bias"}
    )
    modules_to_save: Optional[list[str]] = field(
        default=None, metadata={"help": "Save"}
    )
    init_weights: bool = field(
        default=True, metadata={"help": "Init"}
    )
    layers_to_transform: Optional[Union[list[int], int]] = field(
        default=None, metadata={"help": "Layers"}
    )
    layers_pattern: Optional[Union[list[str], str]] = field(
        default=None, metadata={"help": "Pattern"}
    )
    initialization_method: str = field(
        default="kaiming", metadata={"help": "Initialization method", "choices": ["kaiming", "xavier", "orthogonal", "random"]}
    )
    reinit_threshold: int = field(
        default=0, metadata={"help": "Reinit. 0 means no update"}
    )
    below_threshold_max_count: int = field(
        default=0, metadata={"help": "Below. 0 means no update"}
    )

    def __post_init__(self):
        super().__post_init__()
        self.peft_type = PeftType.UORA
        self.target_modules = (
            set(self.target_modules) if isinstance(self.target_modules, list) else self.target_modules
        )
        # check for layers_to_transform and layers_pattern
        if self.layers_pattern and not self.layers_to_transform:
            raise ValueError("When `layers_pattern` is specified, `layers_to_transform` must also be specified. ")
        if not self.save_projection:
            warnings.warn(
                "Specified to not save uora_A and uora_B within the state dictionary, instead they will be restored "
                "using the PRNG key store in `config.projection_prng_key`. Consider setting `config.save_projection` "
                "to `True` to guarantee restoring the checkpoint correctly on all system configurations."
            )

