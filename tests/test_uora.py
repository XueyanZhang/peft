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
            alpha=0.7,
            tau=1e-6,
            count_k=3,
            gradient_accumulation_steps=8,
        )
        
        assert config.r == 16
        assert config.d_initial == 0.2
        assert config.uora_dropout == 0.1
        assert config.save_projection is False
        assert config.alpha == 0.7
        assert config.tau == 1e-6
        assert config.count_k == 3
        assert config.gradient_accumulation_steps == 8
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

    def test_uora_specific_initialization(self, mlp):
        """Test that UORA uses orthogonal initialization (different from VeRA)."""
        config = UoraConfig(target_modules=["lin1", "lin2"], r=16, projection_prng_key=42)
        peft_model = get_peft_model(mlp, config)
        
        # Check that UORA matrices exist and have correct shape
        lin1_layer = peft_model.base_model.model.lin1
        assert hasattr(lin1_layer, 'uora_A')
        assert hasattr(lin1_layer, 'uora_B')
        
        uora_A = lin1_layer.uora_A["default"]
        uora_B = lin1_layer.uora_B["default"]
        
        # Check shapes
        assert uora_A.shape[0] == config.r
        assert uora_B.shape[1] == config.r
        
        # For orthogonal matrices, check that they're approximately orthogonal
        # (This is a basic sanity check - exact orthogonality might not hold for rectangular matrices)
        if uora_A.shape[0] <= uora_A.shape[1]:
            product = uora_A @ uora_A.T
            identity = torch.eye(uora_A.shape[0])
            max_diff = torch.abs(product - identity).max().item()
            assert max_diff < 1e-3, f"Matrix A not approximately orthogonal, max diff: {max_diff}"

    def test_uora_state_tracking(self, mlp):
        """Test that UORA state tracking variables are initialized."""
        config = UoraConfig(
            target_modules=["lin1"], 
            r=8,
            alpha=0.6,
            tau=1e-5,
            count_k=2,
            gradient_accumulation_steps=4
        )
        peft_model = get_peft_model(mlp, config)
        
        lin1_layer = peft_model.base_model.model.lin1
        
        # Check that UORA state tracking variables are present
        assert hasattr(lin1_layer, 'lambda_d_counter')
        assert hasattr(lin1_layer, 'gradient_step')
        assert hasattr(lin1_layer, 'alpha')
        assert hasattr(lin1_layer, 'tau')
        assert hasattr(lin1_layer, 'count_k')
        assert hasattr(lin1_layer, 'gradient_accumulation_steps')
        
        # Check initial values
        assert lin1_layer.lambda_d_counter.shape == (config.r,)
        assert lin1_layer.gradient_step == 0
        assert lin1_layer.alpha == config.alpha
        assert lin1_layer.tau == config.tau
        assert lin1_layer.count_k == config.count_k
        assert lin1_layer.gradient_accumulation_steps == config.gradient_accumulation_steps

    def test_uora_methods_exist(self, mlp):
        """Test that UORA-specific methods are available."""
        config = UoraConfig(target_modules=["lin1"], alpha=0.5, tau=1e-5, count_k=1)
        peft_model = get_peft_model(mlp, config)
        
        lin1_layer = peft_model.base_model.model.lin1
        
        # Check that UORA methods exist
        assert hasattr(lin1_layer, 'get_dynamic_alpha')
        assert hasattr(lin1_layer, 'update_tau')
        assert hasattr(lin1_layer, 'reinit_uora_matrix_at_index')
        assert hasattr(lin1_layer, 'run_uora')
        
        # Test that methods are callable
        assert callable(lin1_layer.get_dynamic_alpha)
        assert callable(lin1_layer.update_tau)
        assert callable(lin1_layer.reinit_uora_matrix_at_index)
        assert callable(lin1_layer.run_uora)