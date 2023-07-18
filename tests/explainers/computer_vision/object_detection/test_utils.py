# pylint: disable = missing-class-docstring
import numpy as np
import pytest

from foxai.explainer.computer_vision.object_detection.utils import resize_image


def test_resize_image_should_resize_to_minimal_new_shape_image_without_change_original_ratio():
    image = np.zeros((100, 100))
    new_shape = (300, 200)
    resized_image = resize_image(
        image=np.asarray(image),
        new_shape=new_shape,
        change_original_ratio=False,
        scaleup=True,
    )

    assert isinstance(resized_image, np.ndarray)
    assert resized_image.shape == (200, 200)


def test_resize_image_should_not_upscale_image_to_new_shape_image_without_change_original_ratio():
    image = np.zeros((100, 100))
    new_shape = (300, 200)
    resized_image = resize_image(
        image=np.asarray(image),
        new_shape=new_shape,
        change_original_ratio=False,
        scaleup=False,
    )

    assert isinstance(resized_image, np.ndarray)
    assert resized_image.shape == image.shape


@pytest.mark.parametrize("scaleup", [False, True])
def test_resize_image_should_resize_to_exact_new_shape_image_with_changing_original_ratio(
    scaleup: bool,
):
    image = np.zeros((100, 100))
    new_shape = (300, 200)
    resized_image = resize_image(
        image=np.asarray(image),
        new_shape=new_shape,
        change_original_ratio=True,
        scaleup=scaleup,
    )

    assert isinstance(resized_image, np.ndarray)
    assert resized_image.shape == new_shape
