"""Entry point for CLI."""
import datetime
import os
import uuid
from argparse import ArgumentParser, Namespace
from pathlib import Path

from src.cache_manager import LocalDirCacheManager
from src.data_transformer import ExplainerCVTransformer
from src.image_utils import load_image
from src.model_utils import get_prediction, load_model, load_model_idx_to_label
from src.path_manager import ExperimentDataClass


def parse_args() -> Namespace:
    """Parse CLI arguments.

    Returns:
        Namespace with parsed arguments.
    """
    parser = ArgumentParser(description="AutoXAI - Explainable AI")
    parser.add_argument(
        "--cache",
        type=str,
        default="autoxai_cache/",
        help="Path to AutoXAI cache directory",
    )
    parser.add_argument("--img_path", type=str, required=True, help="Path to image")
    return parser.parse_args()


def main():
    """Entry point for application."""
    args = parse_args()

    experiment = ExperimentDataClass(
        base_path=args.cache,
        date=str(datetime.date.today()),
        uuid=str(uuid.uuid1()),
    )
    model = load_model()
    idx_to_label = load_model_idx_to_label()

    transformer = ExplainerCVTransformer()
    cache_manager = LocalDirCacheManager()
    cache_manager.save_artifact(
        os.path.join(experiment.path, "labels", "idx_to_label.json"),
        idx_to_label,
    )

    for filepath in [args.img_path]:
        image_name = Path(filepath).name
        img = load_image(filepath)

        cache_manager.save_artifact(
            os.path.join(experiment.path_to_data, "original", image_name),
            transformer.center(transformer.resize(img)),
        )
        input_data = transformer.preprocess(data=img)
        cache_manager.save_artifact(
            os.path.join(experiment.path_to_data, "input_data", image_name),
            input_data,
        )

        for epoch in [0, 1, 2]:
            cache_manager.save_artifact(
                os.path.join(
                    experiment.path, "training", str(epoch), "model", "model.pkl"
                ),
                model,
            )

            pred_label_idx = get_prediction(model=model, input_data=input_data)
            cache_manager.save_artifact(
                os.path.join(
                    experiment.path,
                    "training",
                    str(epoch),
                    "predictions",
                    Path(Path(filepath).name).with_suffix(".json"),
                ),
                {"predictions": [pred_label_idx.item()]},
            )


if __name__ == "__main__":
    main()
