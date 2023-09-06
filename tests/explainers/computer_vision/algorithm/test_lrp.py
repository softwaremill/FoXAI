import pytest
import torch

from foxai.explainer import LayerLRPCVExplainer, LRPCVExplainer
from foxai.explainer.computer_vision.algorithm.lrp_rules import EpsilonRule, GammaRule
from tests.sample_model import CNN, AutoEncoder


class TestLRPCVExplainer:
    """Test for RLP rules."""

    def test_add_rules_correctly_adds_rules_to_layers(self):
        model = AutoEncoder()
        explainer = LRPCVExplainer()

        assert not explainer.layer_to_rule
        explainer.add_rules(model=model)
        layers = list(model.modules())

        assert isinstance(explainer.layer_to_rule[layers[0]], GammaRule)
        assert isinstance(explainer.layer_to_rule[layers[1]], GammaRule)
        assert isinstance(explainer.layer_to_rule[layers[2]], GammaRule)
        assert isinstance(explainer.layer_to_rule[layers[3]], GammaRule)
        assert isinstance(explainer.layer_to_rule[layers[4]], GammaRule)
        assert isinstance(explainer.layer_to_rule[layers[5]], EpsilonRule)
        assert isinstance(explainer.layer_to_rule[layers[6]], EpsilonRule)
        assert isinstance(explainer.layer_to_rule[layers[7]], EpsilonRule)
        assert isinstance(explainer.layer_to_rule[layers[8]], EpsilonRule)
        assert explainer.layer_to_rule[layers[8]].epsilon == 0

    # pylint: disable=unsubscriptable-object
    def test_safe_model_modify_restores_initial_model_state(self):
        model = CNN()
        explainer = LRPCVExplainer()
        initial_state_dict = model.state_dict()
        with explainer.safe_model_modify(model) as model_modified:
            model_modified.conv1.weight = torch.nn.Parameter(torch.randn(20, 1, 5, 5))
            assert not torch.equal(
                initial_state_dict["conv1.weight"],
                model_modified.state_dict()["conv1.weight"],
            )

        assert torch.equal(
            initial_state_dict["conv1.weight"], model.state_dict()["conv1.weight"]
        )

    def test_get_relevances_properly_get_relevance_for_non_layer(self):
        model = CNN()
        explainer = LRPCVExplainer()
        gradients = torch.randn(10, 20)
        relevances = explainer.get_relevances(model, gradients)
        assert torch.equal(gradients, relevances)

    def test_get_relevances_raises_assertion_error_when_layer_relevance_not_calculated(
        self,
    ):
        model = CNN()
        explainer = LayerLRPCVExplainer()
        gradients = torch.randn(10, 20)
        explainer.add_rules(model)
        with pytest.raises(AssertionError):
            explainer.get_relevances(model, gradients)

    def test_get_relevances_properly_gets_relevance_for_layer_if_attribute_to_layer_input(
        self,
    ):
        model = CNN()
        explainer = LayerLRPCVExplainer()
        gradients = torch.randn(10, 20)
        explainer.add_rules(model)
        relevance = torch.randn(10, 20)
        explainer.layer_to_rule[model.conv2].relevance_input = relevance

        calculated_relevance = explainer.get_relevances(
            model, gradients, attribute_to_layer_input=True
        )
        assert torch.equal(relevance, calculated_relevance)

    def test_get_relevances_properly_gets_relevance_for_layer_if_not_attribute_to_layer_input(
        self,
    ):
        model = CNN()
        explainer = LayerLRPCVExplainer()
        gradients = torch.randn(10, 20)
        explainer.add_rules(model)
        relevance = torch.randn(10, 20)
        explainer.layer_to_rule[model.conv1].relevance_output = relevance

        calculated_relevance = explainer.get_relevances(
            model, gradients, layer=model.conv1, attribute_to_layer_input=False
        )
        assert torch.equal(relevance, calculated_relevance)
