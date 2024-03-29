from enum import Enum
from typing import List, Tuple

import numpy as np
import torch
from torchvision.transforms.functional import gaussian_blur

from foxai.types import AttributionsType, ModelType
from foxai.visualizer import _preprocess_img_and_attributes


class Metrics(Enum):
    """
    Helper Enum representing insertion and deletion metrics.
    """

    INSERTION = 1
    DELETION = 2


def _get_stepped_attrs(sorted_attrs: np.ndarray, steps_num: int) -> np.ndarray:
    """Get elements from array according to the number of wanted steps.

    Args:
        sorted_attrs: Numpy array of floats correponding to importance map values
            sorted in ascending or descending order.
        stepns_num: wanted number of steps

    Returns:
        Numpy array of sample values according to decided number of steps.
    """
    total_samples: int = len(sorted_attrs)
    required_step: int = total_samples // steps_num
    return sorted_attrs[::required_step]


def _metric_calculation(
    attributions: AttributionsType,
    transformed_img: torch.Tensor,
    model: ModelType,
    chosen_class: int,
    steps_num=30,
    metric_type=Metrics.INSERTION,
    kernel=(101, 101),
) -> Tuple[np.ndarray, List]:
    """Calculate metric (insertion or deletion) given importance map, image, model and chosen class.
    Implementation of both metrics (insertion and deletion) are inspired by the paper
    "RISE: Randomized Input Sampling for Explanation of Black-box Models": https://arxiv.org/abs/1806.07421

    Args:
        attributions: Torch Tensor corresponding to importance map.
        transformed_img: Torch Tensor corresponding to image.
        model: model which we are explaining.
        chosen_class: index of the class we are creating metric for.
        metric_type: type of metric presented using enum, supported ones are: Insertion and Deletion.
        kernel: define a tuple regarding the used blurring kernel. Default value is 101 to produce very blurred value.

    Returns:
        metric: numerical value of chosen metric for given picture and explanation.
        importance_list: list of numpy elements corresponding to confidence value at each step.

    Raises:
        AttributeError: if metric type is not enum of Metrics.INSERTION or Metrics.DELETION
    """

    if metric_type not in [Metrics.INSERTION, Metrics.DELETION]:
        raise AttributeError(f"Metric type not in {['INSERTION', 'DELETION']}")

    attributes_matrix: np.ndarray = attributions.detach().cpu().numpy()
    transformed_img_np: np.ndarray = transformed_img.detach().cpu().numpy()

    preprocessed_attrs, _ = _preprocess_img_and_attributes(
        attributes_matrix=attributes_matrix,
        transformed_img_np=transformed_img_np,
        only_positive_attr=True,
    )

    sorted_attrs: np.ndarray = np.flip(np.sort(np.unique(preprocessed_attrs)))
    stepped_attrs: np.ndarray = _get_stepped_attrs(sorted_attrs, steps_num)

    importance_list: List[Tuple[float, float]] = []

    cuda = next(model.parameters()).is_cuda
    device = torch.device("cuda" if cuda else "cpu")

    removed_img_part: torch.Tensor = torch.zeros(transformed_img.shape).to(device)
    removed_img_part[:] = transformed_img.mean()

    if metric_type == Metrics.INSERTION:
        removed_img_part = gaussian_blur(transformed_img, kernel)

    for val in stepped_attrs:
        attributes_map_np: np.ndarray = np.expand_dims(
            np.where(preprocessed_attrs <= val, 1, 0), axis=-1
        )
        attributes_map_np = attributes_map_np.repeat(3, axis=-1)
        attributes_map: torch.Tensor = (
            torch.from_numpy(attributes_map_np).permute(2, 1, 0).to(device)
        )

        attributes_map_inv_np: np.ndarray = np.expand_dims(
            np.where(preprocessed_attrs <= val, 0, 1), axis=-1
        )
        attributes_map_inv_np = attributes_map_inv_np.repeat(3, axis=-1)
        attributes_map_inv: torch.Tensor = (
            torch.from_numpy(attributes_map_inv_np).permute(2, 1, 0).to(device)
        )

        if metric_type == Metrics.DELETION:
            perturbed_img: torch.Tensor = (
                transformed_img * attributes_map + removed_img_part * attributes_map_inv
            )
        else:
            perturbed_img = (
                removed_img_part * attributes_map + transformed_img * attributes_map_inv
            )

        perturbed_img = perturbed_img.to(device)

        output = model(perturbed_img.unsqueeze(dim=0))
        softmax_output: torch.Tensor = torch.nn.functional.softmax(output)[0]
        importance_val: float = float(
            softmax_output[chosen_class].detach().cpu().numpy()
        )
        importance_list.append((val, importance_val))

    importance_values: List = [elem[0] for elem in importance_list]
    metric: np.ndarray = np.round(
        np.trapz(importance_values) / len(importance_values), 4
    )

    return metric, importance_list


def deletion(
    attributions: AttributionsType,
    transformed_img: torch.Tensor,
    model: ModelType,
    chosen_class: int,
) -> Tuple[np.ndarray, List]:
    """Calculate deletion metric given importance map, image, model and chosen class.

    Args:
        attributions: Torch Tensor corresponding to importance map.
        transformed_img: Torch Tensor corresponding to image.
        model: model which we are explaining.
        chosen_class: index of the class we are creating metric for.

    Returns:
        metric: numerical value of chosen metric for given picture and explanation.
        importance_list: list of numpy elements corresponding to confidence value at each step.

    Raises:
        AttributeError: if metric type is not enum of Metrics.INSERTION or Metrics.DELETION
    """
    return _metric_calculation(
        attributions, transformed_img, model, chosen_class, metric_type=Metrics.DELETION
    )


def insertion(
    attributions: AttributionsType,
    transformed_img: torch.Tensor,
    model: ModelType,
    chosen_class: int,
    kernel=(101, 101),
) -> Tuple[np.ndarray, List]:
    """Calculate insertion metric given importance map, image, model and chosen class.

    Args:
        attributions: Torch Tensor corresponding to importance map.
        transformed_img: Torch Tensor corresponding to image.
        model: model which we are explaining.
        chosen_class: index of the class we are creating metric for.

    Returns:
        metric: numerical value of chosen metric for given picture and explanation.
        importance_list: list of numpy elements corresponding to confidence value at each step.

    Raises:
        AttributeError: if metric type is not enum of Metrics.INSERTION or Metrics.DELETION
    """
    return _metric_calculation(
        attributions,
        transformed_img,
        model,
        chosen_class,
        metric_type=Metrics.INSERTION,
        kernel=kernel,
    )
