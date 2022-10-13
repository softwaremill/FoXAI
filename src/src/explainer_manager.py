import torch
from PIL import Image

import os
import numpy as np

from typing import Dict, Any, List

from explainer import (
    CVExplainer,
)
from data_transformer import (
    ComputerVisionTransformer,
)
from cache_manager import CacheManager
from path_manager import ExperimentDataClass
from model_utils import get_prediction


class OfflineExplainer:
    """Explainer for offline mode."""

    def make_explanation(
        self,
        experiment,
        model: Any,
        input_data: torch.Tensor,
        pred_label_idx: torch.Tensor,
        explainer: CVExplainer,
        params: Dict[str, Any],
        result_filename: str,
        cache_manager: CacheManager,
    ) -> None:
        """Make single explanation given algorithm and arguments.

        Args:
            experiment (_type_): Dataclass to help generate paths.
            model (Any): Explained model.
            input_data (torch.Tensor): Resized and normalized image.
            pred_label_idx (torch.Tensor): Predicted label.
            explainer (CVExplainer): Algorithm used to explain prediction.
            params (Dict[str, Any]): Dictionary of arguments.
            result_filename (str): Filename of resulting image.
            cache_manager (CacheManager): Cache manager.
        """
        explanations_path = os.path.join(experiment.path_to_explanations, explainer.algorithm_name)
        explanations_path_figures = experiment.generate_path_to_experiment_figures(explanations_path)

        attributes = explainer.calculate_features(model, input_data, pred_label_idx, **params)
        cache_manager.save_artifact(os.path.join(explanations_path_figures, "params.json"), params)
        np_attr: np.ndarray = attributes.cpu().detach().numpy()
        np.save(os.path.join(explanations_path_figures, result_filename), np_attr)

    def explain_cv_prediction(
        self,
        transformer: ComputerVisionTransformer,
        experiment: ExperimentDataClass,
        model: Any,
        img: Image,
        algorithm_list: List[CVExplainer],
        parameter_list: List[Dict[str, Any]],
        result_name_list: List[str],
        cache_manager: CacheManager,
    ) -> None:
        """Make explanations for all provided algorithms.

        Args:
            transformer (ComputerVisionTransformer): Data transformer.
            experiment (ExperimentDataClass): Dataclass to help generate paths.
            model (Any): Explained model.
            img (Image): Image to explain.
            algorithm_list (List[CVExplainer]): List of algorithms to explain.
            parameter_list (List[Dict[str, Any]]): List of arguments to algorithms.
            result_name_list (List[str]): List of result image names.
            cache_manager (CacheManager): Cache manager.
        """
        transformed_img = transformer.resize_and_center(img=img)
        input_data = transformer.transform(img=transformed_img)
        pred_label_idx = get_prediction(model, input_data)

        cache_manager.save_artifact(os.path.join(experiment.path_to_model, "model.pkl"), model)
        cache_manager.save_artifact(
            os.path.join(experiment.path_to_data, "original", experiment.image_name),
            img,
        )
        cache_manager.save_artifact(
            os.path.join(experiment.path_to_data, "input_data", experiment.image_name),
            input_data,
        )
        cache_manager.save_artifact(
            os.path.join(
                experiment.path_to_data,
                "normalized",
                experiment.image_name.split(".")[0] + ".pt",
            ),
            transformed_img,
        )
        cache_manager.save_artifact(
            os.path.join(experiment.path_to_data, "predictions", "data.json"),
            {"predictions": [pred_label_idx.item()]},
        )

        for explainer, params, result_filename in zip(algorithm_list, parameter_list, result_name_list):
            self.make_explanation(
                experiment,
                model,
                input_data,
                pred_label_idx,
                explainer,
                params,
                result_filename,
                cache_manager,
            )
