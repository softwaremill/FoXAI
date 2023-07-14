import random

import torch

from foxai.metrics.focus import (
    MosaicData,
    create_mosaic_picture,
    create_mosaics_from_images,
    focus,
)


def test_create_mosaic_picture_should_create_correct_mosaic_data():
    images_with_labels = [
        [torch.Tensor([[[1, 2], [3, 4]]]), torch.Tensor([1])],
        [torch.Tensor([[[5, 6], [7, 8]]]), torch.Tensor([2])],
        [torch.Tensor([[[9, 10], [11, 12]]]), torch.Tensor([3])],
        [torch.Tensor([[[13, 14], [15, 16]]]), torch.Tensor([4])],
    ]
    random.seed(4)
    expected_mosaic_data = MosaicData(
        mosaic_image=torch.tensor(
            [
                [
                    [9.0, 10.0, 13.0, 14.0],
                    [11.0, 12.0, 15.0, 16.0],
                    [1.0, 2.0, 5.0, 6.0],
                    [3.0, 4.0, 7.0, 8.0],
                ]
            ]
        ),
        mosaic_labels=torch.tensor([[3.0, 4.0], [1.0, 2.0]]),
    )
    actual_mosaic_data = create_mosaic_picture(images_with_labels=images_with_labels)
    assert torch.equal(
        expected_mosaic_data.mosaic_image, actual_mosaic_data.mosaic_image
    )
    assert torch.equal(
        expected_mosaic_data.mosaic_labels, actual_mosaic_data.mosaic_labels
    )


def test_create_mosaics_from_images_should_create_correct_list():
    images_with_labels = [
        [torch.Tensor([[[1, 2], [3, 4]]]), torch.Tensor([1])],
        [torch.Tensor([[[5, 6], [7, 8]]]), torch.Tensor([2])],
        [torch.Tensor([[[9, 10], [11, 12]]]), torch.Tensor([3])],
        [torch.Tensor([[[13, 14], [15, 16]]]), torch.Tensor([4])],
        [torch.Tensor([[[13, 14], [15, 16]]]), torch.Tensor([3])],
    ]
    random.seed(4)
    expected_mosaic_data_list = [
        MosaicData(
            mosaic_image=torch.Tensor(
                [
                    [
                        [13.0, 14.0, 5.0, 6.0],
                        [15.0, 16.0, 7.0, 8.0],
                        [13.0, 14.0, 9.0, 10.0],
                        [15.0, 16.0, 11.0, 12.0],
                    ]
                ]
            ),
            mosaic_labels=torch.Tensor([[4.0, 2.0], [3.0, 3.0]]),
        ),
        MosaicData(
            mosaic_image=torch.Tensor(
                [
                    [
                        [13.0, 14.0, 1.0, 2.0],
                        [15.0, 16.0, 3.0, 4.0],
                        [13.0, 14.0, 9.0, 10.0],
                        [15.0, 16.0, 11.0, 12.0],
                    ]
                ]
            ),
            mosaic_labels=torch.Tensor([[3.0, 1.0], [4.0, 3.0]]),
        ),
    ]
    actual_mosaic_data_list = create_mosaics_from_images(
        images_with_labels=images_with_labels,
        mosaics_num=2,
        target_class=3,
        pandas_random_state=10,
    )
    assert torch.equal(
        expected_mosaic_data_list[0].mosaic_image,
        actual_mosaic_data_list[0].mosaic_image,
    )
    assert torch.equal(
        expected_mosaic_data_list[0].mosaic_labels,
        actual_mosaic_data_list[0].mosaic_labels,
    )
    assert torch.equal(
        expected_mosaic_data_list[1].mosaic_image,
        actual_mosaic_data_list[1].mosaic_image,
    )
    assert torch.equal(
        expected_mosaic_data_list[1].mosaic_labels,
        actual_mosaic_data_list[1].mosaic_labels,
    )


def test_focus_should_calculate_correctly_for_non_negative_values():
    attributions = torch.tensor(
        [[[0.5, 0.5, 0, 0], [0.2, 0.2, 0.1, 0], [0.4, 0.2, 0, 0], [0.9, 0, 0.8, 0.2]]],
        dtype=torch.float64,
    )
    mosaic_labels = torch.Tensor([[3.0, 1.0], [4.0, 3.0]])
    expected_focus = 0.6
    actual_focus = focus(
        attributions=attributions, mosaic_labels=mosaic_labels, target_class=3
    )
    assert actual_focus == expected_focus


def test_focus_should_calculate_correctly_for_negative_values():
    attributions = torch.tensor(
        [
            [
                [0.5, 0.5, -1.0, -0.6],
                [0.2, 0.2, 0.1, -0.4],
                [0.4, 0.2, 0, 0],
                [0.9, -0.9, 0.8, 0.2],
            ]
        ],
        dtype=torch.float64,
    )
    mosaic_labels = torch.Tensor([[3.0, 1.0], [4.0, 3.0]])
    expected_focus = 0.6
    actual_focus = focus(
        attributions=attributions, mosaic_labels=mosaic_labels, target_class=3
    )
    assert actual_focus == expected_focus
