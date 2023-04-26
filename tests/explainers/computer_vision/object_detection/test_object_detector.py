import numpy as np
import torch

from foxai.explainer.computer_vision.object_detection.base_object_detector import (
    BaseObjectDetector,
)


def test_preprocessing_should_resize_and_transpose_image():
    img = np.zeros((1, 200, 100, 3))
    new_shape = (300, 300, 3)
    expected_shape = (1, 3, 300, 150)

    preprocessed_img = BaseObjectDetector.preprocessing(
        img=img,
        new_shape=new_shape,
        change_original_ratio=False,
        scaleup=True,
    )

    assert isinstance(preprocessed_img, torch.Tensor)
    assert preprocessed_img.dtype == torch.float64
    assert preprocessed_img.shape == expected_shape


def test_preprocessing_should_resize_and_transpose_image_and_change_original_ratio_of_image():
    img = np.zeros((1, 200, 100, 3))
    new_shape = (300, 300, 3)
    expected_shape = (1, 3, 300, 300)

    preprocessed_img = BaseObjectDetector.preprocessing(
        img=img,
        new_shape=new_shape,
        change_original_ratio=True,
        scaleup=True,
    )

    assert isinstance(preprocessed_img, torch.Tensor)
    assert preprocessed_img.dtype == torch.float64
    assert preprocessed_img.shape == expected_shape


def test_preprocessing_should_add_batch_dimension():
    img = np.zeros((200, 100, 3))
    new_shape = (300, 200)
    expected_shape = (1, 3, 200, 100)

    preprocessed_img = BaseObjectDetector.preprocessing(
        img=img,
        new_shape=new_shape,
        change_original_ratio=False,
        scaleup=False,
    )

    assert isinstance(preprocessed_img, torch.Tensor)
    assert preprocessed_img.dtype == torch.float64
    assert preprocessed_img.shape == expected_shape
