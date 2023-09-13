import torch

from foxai.explainer.computer_vision.algorithm.lrp_rules import EpsilonRule, GammaRule


class TestLrpRules:
    """Test for RLP rules."""

    def test_epsilon_rule_does_not_modify_module_weights(self):
        layer = torch.nn.Conv2d(16, 33, (3, 5), stride=(2, 1))
        input_tensor = torch.randn(20, 16, 50, 20)
        initial_weight = layer.weight

        EpsilonRule().modify_weights(layer, input_tensor)
        modified_weight = layer.weight

        assert torch.equal(initial_weight, modified_weight)

    def test_gamma_rule_correctly_modify_module_with_default_argument(self):
        layer = torch.nn.Conv2d(16, 33, (2, 5), stride=(2, 1))
        input_tensor = torch.randn(20, 16, 50, 100)
        initial_weight = torch.Tensor([1, -2, -3, 1, 2, 1, 0])
        layer.weight.data = initial_weight

        GammaRule().modify_weights(layer, input_tensor)
        modified_weight = layer.weight

        assert torch.equal(
            torch.Tensor([1.25, -2, -3, 1.25, 2.5, 1.25, 0]), modified_weight.data
        )

    def test_gamma_rule_correctly_modify_module_with_custom_argument(self):
        layer = torch.nn.Conv2d(16, 33, (1, 5), stride=(2, 1))
        input_tensor = torch.randn(20, 16, 50, 80)
        initial_weight = torch.Tensor([1, -2, -3, 1, 2, 1, 0])
        layer.weight.data = initial_weight

        GammaRule(gamma=5).modify_weights(layer, input_tensor)
        modified_weight = layer.weight

        assert torch.equal(torch.Tensor([6, -2, -3, 6, 12, 6, 0]), modified_weight.data)
