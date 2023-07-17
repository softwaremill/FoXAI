import random
from dataclasses import dataclass
from typing import List, Optional

import numpy as np
import pandas as pd
import torch

from foxai.array_utils import retain_only_positive


@dataclass
class MosaicData:
    """Object representing mosaic image data.

    Args:
        mosaic_image: Torch Tensor representing image where each quadrant is coming from different source image.
        mosaic_labels: Torch Tensor of 2x2 dimension representing labels of each quadrant image.
    """

    mosaic_image: torch.Tensor
    mosaic_labels: torch.Tensor


def create_mosaic_picture(images_with_labels: List[List[torch.Tensor]]) -> MosaicData:
    """Create mosaic from a list of four pictures. Images are placed at random positions.

    Args:
        images_with_labels: List of images in a format of List[torch.Tensor],
                            where first element is an image and second is a label.

    Returns:
        MosaicData object which contains 2x2 images mosaic and a 2x2 matrix of their labels.

    Raises:
        AttributeError: if number of passed images is different from four.
    """
    if len(images_with_labels) != 4:
        raise AttributeError(
            f"Mosaic has to be created from 4 images. Passed number is {len(images_with_labels)}"
        )
    random.shuffle(images_with_labels)
    images, labels = zip(*images_with_labels)
    size = len(images[0].size())
    mosaic_image = torch.cat(
        [torch.cat(images[:2], size - 2), torch.cat(images[2:], size - 2)], size - 1
    )
    mosaic_labels = torch.stack([torch.cat(labels[:2], 0), torch.cat(labels[2:], 0)], 1)
    return MosaicData(mosaic_image=mosaic_image, mosaic_labels=mosaic_labels)


def create_mosaics_from_images(
    images_with_labels: List[List[torch.Tensor]],
    mosaics_num: int,
    target_class: int,
    pandas_random_state: Optional[int] = None,
) -> List[MosaicData]:
    """Create a list of mosaics from a list of images according to rules defined in https://arxiv.org/abs/2109.15035:
        - each mosaic contains two images from target class (c(img1) == c(img2) == target_class)
        - each mosaic contains two images from non-target different classes (c(img3) != c(img4) != target_class)

    Args:
        images_with_labels: List of images in a format of List[torch.Tensor],
                            where first element is an image and second is a label.
        mosaics_num: A number of mosaics to generate.
        target_class: Target class to be used for mosaics creation.
        pandas_random_state: a random state to be passed to DataFrame.sample() method, use for repetitive results

    Returns:
        A list of MosaicData objects which contain 2x2 images mosaic and a 2x2 matrices of their labels.
    """
    mosaic_data_list = []
    for _ in range(mosaics_num):
        df = pd.DataFrame(images_with_labels, columns=["image", "label"])
        df["label"] = df["label"].apply(lambda label: label.item())
        img3_label, img4_label = random.sample(df["label"].unique().tolist(), 2)
        img1, img2 = (
            df[df["label"] == target_class]
            .sample(2, random_state=pandas_random_state)["image"]
            .tolist()
        )
        img3 = (
            df[df["label"] == img3_label]
            .sample(1, random_state=pandas_random_state)["image"]
            .values[0]
        )
        img4 = (
            df[df["label"] == img4_label]
            .sample(1, random_state=pandas_random_state)["image"]
            .values[0]
        )
        mosaic_data_list.append(
            create_mosaic_picture(
                [
                    [img1, torch.tensor([target_class])],
                    [img2, torch.tensor([target_class])],
                    [img3, torch.tensor([img3_label])],
                    [img4, torch.tensor([img4_label])],
                ]
            )
        )
    return mosaic_data_list


def focus(
    attributions: torch.Tensor, mosaic_labels: torch.Tensor, target_class: int
) -> float:
    """Calculate focus metric based on formula defined in https://arxiv.org/abs/2109.15035:
        focus = (relevance(img1) + relevance(img2)) / (relevance(mosaic)),
        where: img1, img2 belong to target class.

    Args:
        attributions: Torch Tensor corresponding to importance map. This is assumed to be only non-negative values,
                      however if any negative values are provided they will be treated as 0.
        mosaic_labels: Torch Tensor of 2x2 dimension representing labels of each quadrant image.
        target_class: Target class to be used for focus calculation.

    Returns:
        A float value representing calculated focus metric.
    """
    relevance_matrix = attributions.detach().cpu().numpy()
    relevance_matrix = retain_only_positive(relevance_matrix)
    total_relevance = np.sum(relevance_matrix)
    one_picture_width, one_picture_height = int(relevance_matrix.shape[1] / 2), int(
        relevance_matrix.shape[2] / 2
    )
    class_relevance = 0.0
    if mosaic_labels[0][0] == target_class:
        class_relevance += np.sum(
            relevance_matrix[:, :one_picture_width, :one_picture_height]
        )
    if mosaic_labels[0][1] == target_class:
        class_relevance += np.sum(
            relevance_matrix[:, :one_picture_width, one_picture_height:]
        )
    if mosaic_labels[1][0] == target_class:
        class_relevance += np.sum(
            relevance_matrix[:, one_picture_width:, :one_picture_height]
        )
    if mosaic_labels[1][1] == target_class:
        class_relevance += np.sum(
            relevance_matrix[:, one_picture_width:, one_picture_height:]
        )
    return class_relevance / total_relevance
