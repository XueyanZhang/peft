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

# This test file is for tests specific to UORA, based on VeRA tests since UORA has similar architecture with shared weights.

import pytest
import torch
from torch import nn

from peft import UoraConfig, get_peft_model


class MLP(nn.Module):
    def __init__(self, bias=True):
        super().__init__()
        self.relu = nn.ReLU()
        self.lin0 = nn.Linear(10, 20, bias=bias)
        self.lin1 = nn.Linear(20, 20, bias=bias)  # lin1 and lin2 have same shape
        self.lin2 = nn.Linear(20, 20, bias=bias)
        self.lin3 = nn.Linear(20, 2, bias=bias)
        self.sm = nn.LogSoftmax(dim=-1)

    def forward(self, X):
        X = self.lin0(X)
        X = self.relu(X)
        X = self.lin1(X)
        X = self.relu(X)
        X = self.lin2(X)
        X = self.relu(X)
        X = self.lin3(X)
        X = self.sm(X)
        return X


class TestUora:
    @pytest.fixture
    def mlp(self):
        torch.manual_seed(0)
        model = MLP()
        return model

    @pytest.fixture
    def mlp_same_prng(self, mlp):
        torch.manual_seed(0)

        config = UoraConfig(target_modules=["lin1", "lin2"], init_weights=False)
        # creates a default UORA adapter
        peft_model = get_peft_model(mlp, config)
        config2 = UoraConfig(target_modules=["lin1", "lin2"], init_weights=False)
        peft_model.add_adapter("other", config2)
        return peft_model

    def test_multiple_adapters_same_prng_weights(self, mlp_same_prng):
        # we can have multiple adapters with the same prng key, in which case the weights should be shared
        assert (
            mlp_same_prng.base_model.model.lin1.uora_A["default"]
            is mlp_same_prng.base_model.model.lin1.uora_A["other"]
        )
        assert (
            mlp_same_prng.base_model.model.lin1.uora_B["default"]
            is mlp_same_prng.base_model.model.lin1.uora_B["other"]
        )
        assert (
            mlp_same_prng.base_model.model.lin2.uora_A["default"]
            is mlp_same_prng.base_model.model.lin2.uora_A["other"]
        )
        assert (
            mlp_same_prng.base_model.model.lin2.uora_B["default"]
            is mlp_same_prng.base_model.model.lin2.uora_B["other"]
        )

        input = torch.randn(5, 10)
        mlp_same_prng.set_adapter("default")
        output_default = mlp_same_prng(input)
        mlp_same_prng.set_adapter("other")
        output_other = mlp_same_prng(input)
        assert not torch.allclose(output_default, output_other, atol=1e-3, rtol=1e-3)

    def test_multiple_adapters_different_prng_raises(self):
        # we cannot have multiple adapters with different prng keys
        model = MLP()
        config = UoraConfig(target_modules=["lin1", "lin2"], init_weights=False)
        # creates a default UORA adapter
        peft_model = get_peft_model(model, config)
        config2 = UoraConfig(target_modules=["lin1", "lin2"], init_weights=False, projection_prng_key=123)
        with pytest.raises(ValueError, match="UORA PRNG initialisation key must be the same"):
            peft_model.add_adapter("other", config2)

    def test_multiple_adapters_mixed_save_projection_raises(self, mlp):
        # we cannot have multiple adapters with different save_projection values
        config = UoraConfig(target_modules=["lin1", "lin2"], init_weights=False, save_projection=True)
        # creates a default UORA adapter
        peft_model = get_peft_model(mlp, config)
        config2 = UoraConfig(target_modules=["lin1", "lin2"], init_weights=False, save_projection=False)
        with pytest.raises(ValueError, match="UORA projection weights must be saved for all adapters or none"):
            peft_model.add_adapter("other", config2)

    def test_basic_functionality(self, mlp):
        config = UoraConfig(target_modules=["lin1", "lin2"], r=8, d_initial=0.1)
        peft_model = get_peft_model(mlp, config)
        
        # Test forward pass
        input_tensor = torch.randn(5, 10)
        output = peft_model(input_tensor)
        assert output.shape == (5, 2)
        
        # Test parameter names
        param_names = [name for name, _ in peft_model.named_parameters()]
        uora_params = [name for name in param_names if "uora_lambda_" in name]
        assert len(uora_params) > 0, "No UORA parameters found"
        
        # Test that trainable parameters are correct
        trainable_params = [name for name, param in peft_model.named_parameters() if param.requires_grad]
        uora_trainable = [name for name in trainable_params if "uora_lambda_" in name]
        assert len(uora_trainable) > 0, "No trainable UORA parameters found"

    def test_config_attributes(self):
        config = UoraConfig(
            r=16,
            target_modules=["query", "value"],
            d_initial=0.2,
            uora_dropout=0.1,
            save_projection=False,
        )
        
        assert config.r == 16
        assert config.d_initial == 0.2
        assert config.uora_dropout == 0.1
        assert config.save_projection is False
        assert config.peft_type.name == "UORA"

    def test_state_dict_keys(self, mlp):
        config = UoraConfig(target_modules=["lin1", "lin2"])
        peft_model = get_peft_model(mlp, config)
        
        state_dict = peft_model.state_dict()
        uora_keys = [key for key in state_dict.keys() if "uora_lambda_" in key]
        assert len(uora_keys) > 0, "No UORA keys in state dict"
        
        # Check for lambda_b and lambda_d parameters
        lambda_b_keys = [key for key in uora_keys if "lambda_b" in key]
        lambda_d_keys = [key for key in uora_keys if "lambda_d" in key]
        assert len(lambda_b_keys) > 0, "No lambda_b parameters found"
        assert len(lambda_d_keys) > 0, "No lambda_d parameters found"

    def test_repr(self, mlp):
        config = UoraConfig(target_modules=["lin1"])
        peft_model = get_peft_model(mlp, config)
        
        # Test that the repr contains uora
        lin1_repr = repr(peft_model.base_model.model.lin1)
        assert "uora" in lin1_repr.lower()