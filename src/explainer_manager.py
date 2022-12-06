"""Explainer manager class."""
import os
from typing import Any

import torch

from src.cache_manager import CacheManager
from src.data_transformer import CVTransformer
from src.model_utils import get_prediction
from src.path_manager import ExperimentDataClass


class ExplainerManager:  # pylint: disable = (too-few-public-methods)
    """Explainer manager class."""

    def explain_cv_prediction(  # pylint: disable = (too-many-arguments)
        self,
        transformer: CVTransformer,
        experiment: ExperimentDataClass,
        model: Any,
        img: torch.Tensor,
        image_name: str,
        cache_manager: CacheManager,
    ) -> None:
        """Make explanations for all provided algorithms.

        Args:
            transformer: Data transformer.
            experiment: Dataclass to help generate paths.
            model: Explained model.
            img: Image to explain.
            image_name: Filename of image.
            cache_manager: Cache manager.
        """
        transformed_img: torch.Tensor = transformer.resize_and_center(img=img)
        input_data: torch.Tensor = transformer.transform(img=transformed_img)
        pred_label_idx: torch.Tensor = get_prediction(model, input_data)

        cache_manager.save_artifact(
            os.path.join(experiment.path_to_model, "model.pkl"), model
        )
        cache_manager.save_artifact(
            os.path.join(experiment.path_to_data, "original", image_name),
            img,
        )
        cache_manager.save_artifact(
            os.path.join(experiment.path_to_data, "input_data", image_name),
            input_data,
        )
        cache_manager.save_artifact(
            os.path.join(
                experiment.path_to_data,
                "normalized",
                image_name.split(".")[0] + ".pt",
            ),
            transformed_img,
        )
        cache_manager.save_artifact(
            os.path.join(experiment.path_to_data, "predictions", "data.json"),
            {"predictions": [pred_label_idx.item()]},
        )
