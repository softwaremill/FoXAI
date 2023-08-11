# pylint: disable = missing-class-docstring
import logging
from typing import Any, Dict, List

import pytest
import torch
from torchvision import transforms

from foxai.context_manager import (
    CVClassificationExplainers,
    ExplainerWithParams,
    FoXaiExplainer,
)
from foxai.explainer import InputXGradientCVExplainer
from tests.pickachu_image import pikachu_image
from tests.sample_model import SampleModel


class TestFoXaiExplainer:
    """Test whether context manager correctly
    switches model to eval mode and thows exception
    if no explainers provided.
    """

    @pytest.fixture
    def classifier(self) -> SampleModel:
        """Sample model to run FoXaiExplainer on."""
        return SampleModel()

    def test_evel_mode(self, classifier: SampleModel, caplog: pytest.LogCaptureFixture):
        """Test whether FoXaiExplainer correctly switches to eval mode
        if the model was given in train mode and if the proper WARNING
        massage if provided."""

        classifier.train()
        transform = transforms.Compose(
            [
                transforms.ToTensor(),
                transforms.Grayscale(),
                transforms.Resize(size=224),
                transforms.CenterCrop(size=224),
            ]
        )
        img_tensor: torch.Tensor = transform(pikachu_image).unsqueeze(0)
        caplog.set_level(level=logging.WARNING, logger="foxai.context_manager")

        with FoXaiExplainer(
            model=classifier,
            explainers=[
                ExplainerWithParams(
                    CVClassificationExplainers.CV_NOISE_TUNNEL_EXPLAINER
                )
            ],
        ) as xai_model:
            _, _ = xai_model(img_tensor)

            assert not xai_model.model.training
            assert "The model should be in the eval model" in caplog.text

    def test_no_explainers_given(self, classifier: torch.nn.Module):
        """Test whether FoXaiExplainer correctly raises error,
        if explainers not provided.

        There should be at least on explainer provided.
        """

        classifier.eval()
        transform = transforms.Compose(
            [
                transforms.ToTensor(),
                transforms.Grayscale(),
                transforms.Resize(size=224),
                transforms.CenterCrop(size=224),
            ]
        )
        img_tensor: torch.Tensor = transform(pikachu_image).unsqueeze(0)

        with pytest.raises(ValueError):
            with FoXaiExplainer(
                model=classifier,
                explainers=[],
            ) as xai_model:
                _, _ = xai_model(img_tensor)

    def test_whether_output_match_requested_inputs(self, classifier: torch.nn.Module):
        """Test whether FoXaiExplainer returns explanations,
        for each requested explainer.
        """

        classifier.eval()
        transform = transforms.Compose(
            [
                transforms.ToTensor(),
                transforms.Grayscale(),
                transforms.Resize(size=224),
                transforms.CenterCrop(size=224),
            ]
        )
        img_tensor: torch.Tensor = transform(pikachu_image).unsqueeze(0)

        explainers: List[ExplainerWithParams] = [
            ExplainerWithParams(CVClassificationExplainers.CV_DEEPLIFT_EXPLAINER),
            ExplainerWithParams(CVClassificationExplainers.CV_NOISE_TUNNEL_EXPLAINER),
            ExplainerWithParams(CVClassificationExplainers.CV_OCCLUSION_EXPLAINER),
        ]
        with FoXaiExplainer(
            model=classifier,
            explainers=explainers,
        ) as xai_model:
            _, xai_explanations = xai_model(img_tensor)

            assert list(
                map(
                    lambda explainer_tuple: f"{explainer_tuple[1].explainer_name.name}_{explainer_tuple[0]}",
                    enumerate(explainers),
                )
            ) == list(xai_explanations.keys())

    def test_model_inference_with_explainer(self, classifier: torch.nn.Module):
        """Test whether regular inference and inference with FoXaiExplainer
        gives same results.
        """

        classifier.eval()
        transform = transforms.Compose(
            [
                transforms.ToTensor(),
                transforms.Grayscale(),
                transforms.Resize(size=224),
                transforms.CenterCrop(size=224),
            ]
        )
        img_tensor: torch.Tensor = transform(pikachu_image).unsqueeze(0)

        inference_output = classifier(img_tensor)

        with FoXaiExplainer(
            model=classifier,
            explainers=[
                ExplainerWithParams(
                    CVClassificationExplainers.CV_NOISE_TUNNEL_EXPLAINER
                )
            ],
        ) as xai_model:
            foxai_inference_output, _ = xai_model(img_tensor)

            assert foxai_inference_output == inference_output

    def test_model_with_disabled_gradients(self, classifier: torch.nn.Module):
        """Test whether model properly turns gradients enabled, when feed
        with model without gradients enabled and whether model correctly
        turns gradient back to the input state.
        """

        classifier.eval()
        transform = transforms.Compose(
            [
                transforms.ToTensor(),
                transforms.Grayscale(),
                transforms.Resize(size=224),
                transforms.CenterCrop(size=224),
            ]
        )
        img_tensor: torch.Tensor = transform(pikachu_image).unsqueeze(0)

        with torch.no_grad():
            with FoXaiExplainer(
                model=classifier,
                explainers=[
                    ExplainerWithParams(
                        CVClassificationExplainers.CV_NOISE_TUNNEL_EXPLAINER
                    )
                ],
            ) as xai_model:
                assert torch.is_grad_enabled()
                _, _ = xai_model(img_tensor)

            assert not torch.is_grad_enabled()

    def test_explainers_kwargs(self, classifier: torch.nn.Module):
        """Test whether kwargs are correctly set up in FoXaiExplainer."""

        classifier.eval()

        with FoXaiExplainer(
            model=classifier,
            explainers=[
                ExplainerWithParams(
                    explainer_name=CVClassificationExplainers.CV_GRADIENT_SHAP_EXPLAINER,
                    n_samples=100,
                    stdevs=0.0005,
                ),
                ExplainerWithParams(
                    explainer_name=CVClassificationExplainers.CV_OCCLUSION_EXPLAINER,
                    stride_value=5,
                    window_value=5,
                ),
            ],
            target=0,
        ) as xai_model:
            explainer_kwargs: Dict[str, Any] = xai_model.explainer_map[
                "CV_GRADIENT_SHAP_EXPLAINER_0"
            ].kwargs
            assert explainer_kwargs["n_samples"] == 100
            assert explainer_kwargs["stdevs"] == 0.0005

            explainer_kwargs = xai_model.explainer_map[
                "CV_OCCLUSION_EXPLAINER_1"
            ].kwargs
            assert explainer_kwargs["stride_value"] == 5
            assert explainer_kwargs["window_value"] == 5

    def test_model_explanation_with_context_manager_and_raw(
        self, classifier: torch.nn.Module
    ):
        """Test whether explanations from context manager and explainer class
        gives same results.
        """

        classifier.eval()
        transform = transforms.Compose(
            [
                transforms.ToTensor(),
                transforms.Grayscale(),
                transforms.Resize(size=224),
                transforms.CenterCrop(size=224),
            ]
        )
        img_tensor: torch.Tensor = transform(pikachu_image).unsqueeze(0)

        explainer_attributes = InputXGradientCVExplainer().calculate_features(
            model=classifier,
            input_data=img_tensor,
            pred_label_idx=0,
        )
        with FoXaiExplainer(
            model=classifier,
            explainers=[
                ExplainerWithParams(
                    CVClassificationExplainers.CV_INPUT_X_GRADIENT_EXPLAINER
                )
            ],
        ) as xai_model:
            _, foxai_attributes_dict = xai_model(img_tensor)

        assert torch.equal(
            foxai_attributes_dict[
                f"{CVClassificationExplainers.CV_INPUT_X_GRADIENT_EXPLAINER.name}_0"
            ],
            explainer_attributes,
        )

    def test_model_with_in_training_mode_after_context_manager(
        self, classifier: torch.nn.Module
    ):
        """Test whether model changes state to `eval` in context manager and restores
        `training` state after exit.
        """

        classifier.train()

        assert classifier.training

        with torch.no_grad():
            with FoXaiExplainer(
                model=classifier,
                explainers=[
                    ExplainerWithParams(
                        CVClassificationExplainers.CV_NOISE_TUNNEL_EXPLAINER
                    )
                ],
            ) as _:
                assert not classifier.training

        assert classifier.training

    def test_context_manager_should_process_multiple_explainers_of_the_same_type(
        self, classifier: torch.nn.Module
    ):
        """Test whether explanations from context manager and explainer class
        gives same results.
        """

        classifier.eval()
        transform = transforms.Compose(
            [
                transforms.ToTensor(),
                transforms.Grayscale(),
                transforms.Resize(size=224),
                transforms.CenterCrop(size=224),
            ]
        )
        img_tensor: torch.Tensor = transform(pikachu_image).unsqueeze(0)
        explainers = [
            ExplainerWithParams(
                CVClassificationExplainers.CV_INPUT_X_GRADIENT_EXPLAINER
            ),
            ExplainerWithParams(
                CVClassificationExplainers.CV_INPUT_X_GRADIENT_EXPLAINER
            ),
        ]
        with FoXaiExplainer(
            model=classifier,
            explainers=explainers,
        ) as xai_model:
            _, foxai_attributes_dict = xai_model(img_tensor)

        assert len(foxai_attributes_dict.keys()) == len(explainers)
