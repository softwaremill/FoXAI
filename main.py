"""Entry point for CLI."""
import datetime
import os
import uuid
from argparse import ArgumentParser, Namespace
from pathlib import Path

from src.cache_manager import LocalDirCacheManager
from src.data_transformer import ExplainerCVTransformer
from src.explainer_manager import ExplainerManager
from src.image_utils import load_image
from src.model_utils import load_model, load_model_idx_to_label
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
    image_name = Path(args.img_path).name

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
        os.path.join(experiment.path_to_model, "idx_to_label.json"),
        idx_to_label,
    )
    offline_explainer = ExplainerManager()
    offline_explainer.explain_cv_prediction(
        transformer=transformer,
        experiment=experiment,
        model=model,
        img=load_image(args.img_path),
        image_name=image_name,
        cache_manager=cache_manager,
    )


if __name__ == "__main__":
    main()
