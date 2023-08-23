from unittest.mock import patch

import numpy as np
import pytest
import torch
from torchvision import transforms

from foxai.explainer.computer_vision.algorithm.gradient_utils import (
    compute_gradients,
    compute_layer_gradients,
)
from tests.sample_model import CNN


class TestGradientUtils:
    """Test for gradient utils methods."""

    w = [255, 255, 255]
    c = [0, 0, 0]
    h = [50, 50, 50]
    f = [139, 69, 19]
    b = [255, 210, 100]
    # fmt: off
    sample_image: np.ndarray = np.array([
        [w, w, w, w, c, w, w, w],
        [w, w, w, c, h, c, w, w],
        [w, w, w, c, h, c, w, w],
        [w, w, w, c, h, h, c, w],
        [w, w, c, h, h, h, c, w],
        [w, w, c, h, h, h, f, w],
        [w, w, c, h, c, b, f, w],
        [w, w, c, c, c, c, w, w]],
        dtype=np.uint8,
    )

    # fmt: on
    @patch("torch.autograd.grad", return_value=torch.Tensor([1, 2, 3]))
    @patch(
        "tests.sample_model.AutoEncoder",
        return_value=torch.Tensor([[2, 4, 5], [6, 7, 8]]),
    )
    def test_compute_gradients_calls_autograd_properly_when_target_provided(
        self, auto_encoder, gradient_mock
    ):
        transform = transforms.Compose(
            [
                transforms.ToTensor(),
                transforms.Grayscale(),
                transforms.Resize(size=8),
                transforms.CenterCrop(size=8),
            ]
        )
        batch_img_tensor: torch.Tensor = torch.vstack(
            [transform(self.sample_image).unsqueeze(0) for _ in range(0, 3)]
        )
        batch_img_tensor.requires_grad = True
        gradients_result = compute_gradients(
            forward_fn=auto_encoder,
            input_tensor=batch_img_tensor,
            target_ind=0,
        )
        gradient_mock.assert_called_once()
        assert torch.equal(gradient_mock.call_args[0][0][0], torch.tensor(2.0))
        assert torch.equal(gradient_mock.call_args[0][0][1], torch.tensor(6.0))
        assert torch.equal(gradient_mock.call_args[0][1], batch_img_tensor)
        assert gradients_result == 1.0

    @patch(
        "tests.sample_model.AutoEncoder",
        return_value=torch.Tensor([[[2, 4, 5]], [[6, 7, 8]]]),
    )
    def test_compute_gradients_raises_assertion_error_when_improper_target(
        self, auto_encoder
    ):
        transform = transforms.Compose(
            [
                transforms.ToTensor(),
                transforms.Grayscale(),
                transforms.Resize(size=8),
                transforms.CenterCrop(size=8),
            ]
        )
        batch_img_tensor: torch.Tensor = torch.vstack(
            [transform(self.sample_image).unsqueeze(0) for _ in range(0, 3)]
        )
        batch_img_tensor.requires_grad = True
        with pytest.raises(AssertionError):
            compute_gradients(
                forward_fn=auto_encoder,
                input_tensor=batch_img_tensor,
                target_ind=0,
            )

    @patch("torch.autograd.grad", return_value=torch.Tensor([1, 2, 3]))
    @patch("tests.sample_model.AutoEncoder", return_value=torch.Tensor([[2], [6]]))
    def test_compute_gradients_calls_autograd_properly_when_no_target_provided(
        self, auto_encoder, gradient_mock
    ):
        transform = transforms.Compose(
            [
                transforms.ToTensor(),
                transforms.Grayscale(),
                transforms.Resize(size=8),
                transforms.CenterCrop(size=8),
            ]
        )
        batch_img_tensor: torch.Tensor = torch.vstack(
            [transform(self.sample_image).unsqueeze(0) for _ in range(0, 3)]
        )
        batch_img_tensor.requires_grad = True
        gradients_result = compute_gradients(
            forward_fn=auto_encoder,
            input_tensor=batch_img_tensor,
        )
        gradient_mock.assert_called_once()
        assert torch.equal(gradient_mock.call_args[0][0][0], torch.tensor([2.0]))
        assert torch.equal(gradient_mock.call_args[0][0][1], torch.tensor([6.0]))
        assert torch.equal(gradient_mock.call_args[0][1], batch_img_tensor)
        assert gradients_result == 1.0

    @patch("torch.autograd.grad", return_value=torch.Tensor([1, 2, 3]))
    @patch(
        "tests.sample_model.CNN.forward",
        return_value=torch.Tensor([[2, 4, 5], [6, 7, 8]]),
    )
    def test_compute_layer_gradients_calls_autograd_properly_when_not_attribute_to_layer_input(
        self, _, gradient_mock
    ):
        transform = transforms.Compose(
            [
                transforms.ToTensor(),
                transforms.Grayscale(),
                transforms.Resize(size=8),
                transforms.CenterCrop(size=8),
            ]
        )
        batch_img_tensor: torch.Tensor = torch.vstack(
            [transform(self.sample_image).unsqueeze(0) for _ in range(0, 3)]
        )
        batch_img_tensor.requires_grad = True
        model = CNN()
        gradients_result = compute_layer_gradients(
            model=model,
            layer=model,
            input_tensor=batch_img_tensor,
            target_ind=0,
            attribute_to_layer_input=False,
        )
        gradient_mock.assert_called_once()
        assert torch.equal(gradient_mock.call_args[0][0][0], torch.tensor(2.0))
        assert torch.equal(gradient_mock.call_args[0][0][1], torch.tensor(6.0))
        assert torch.equal(
            gradient_mock.call_args[0][1], torch.Tensor([[2, 4, 5], [6, 7, 8]])
        )
        assert torch.equal(gradients_result[0], torch.tensor(1.0))
        assert torch.equal(gradients_result[1], torch.Tensor([[2, 4, 5], [6, 7, 8]]))

    @patch("torch.autograd.grad", return_value=torch.Tensor([1, 2, 3]))
    @patch(
        "tests.sample_model.CNN.forward",
        return_value=torch.Tensor([[2, 4, 5], [6, 7, 8]]),
    )
    def test_compute_layer_gradients_calls_autograd_properly_when_attribute_to_layer_input(
        self, _, gradient_mock
    ):
        transform = transforms.Compose(
            [
                transforms.ToTensor(),
                transforms.Grayscale(),
                transforms.Resize(size=8),
                transforms.CenterCrop(size=8),
            ]
        )
        batch_img_tensor: torch.Tensor = torch.vstack(
            [transform(self.sample_image).unsqueeze(0) for _ in range(0, 3)]
        )
        batch_img_tensor.requires_grad = True
        model = CNN()
        gradients_result = compute_layer_gradients(
            model=model,
            layer=model,
            input_tensor=batch_img_tensor,
            target_ind=0,
            attribute_to_layer_input=True,
        )
        gradient_mock.assert_called_once()
        assert torch.equal(gradient_mock.call_args[0][0][0], torch.tensor(2.0))
        assert torch.equal(gradient_mock.call_args[0][0][1], torch.tensor(6.0))
        assert torch.equal(gradient_mock.call_args[0][1], batch_img_tensor)
        assert torch.equal(gradients_result[0], torch.tensor(1.0))
        assert torch.equal(gradients_result[1], batch_img_tensor)
