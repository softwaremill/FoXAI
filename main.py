"""Entry point for CLI."""
import datetime
import uuid
from argparse import ArgumentParser, Namespace

from PIL import Image

from src.cache_manager import LocalDirCacheManager
from src.data_transformer import ExplainerCVTransformer
from src.explainer import (
    GradientSHAPCVExplainer,
    IntegratedGradientsCVExplainer,
    NoiseTunnelCVExplainer,
    OcculusionCVExplainer,
)
from src.explainer_manager import ExplainerManager
from src.model_utils import load_model
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
        default="../autoxai_cache/",
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
        image_path=args.img_path,
    )
    model = load_model()

    img = Image.open(args.img_path)

    algorithm_list = [
        IntegratedGradientsCVExplainer(),
        NoiseTunnelCVExplainer(),
        GradientSHAPCVExplainer(),
        OcculusionCVExplainer(),
        OcculusionCVExplainer(),
        OcculusionCVExplainer(),
    ]

    parameter_list = [
        {},
        {},
        {},
        {
            "stride": (3, 8, 8),
        },
        {
            "stride": (3, 25, 25),
            "sliding_window_shapes": (3, 30, 30),
        },
        {
            "stride": (3, 50, 50),
            "sliding_window_shapes": (3, 60, 60),
        },
    ]

    result_name_list = [
        "attributes",
        "attributes",
        "attributes",
        "attributes_stride_8",
        "attributes_stride_25",
        "attributes_stride_50",
    ]

    transformer = ExplainerCVTransformer()
    cache_manager = LocalDirCacheManager()
    offline_explainer = ExplainerManager()
    offline_explainer.explain_cv_prediction(
        transformer,
        experiment,
        model,
        img,
        algorithm_list,
        parameter_list,
        result_name_list,
        cache_manager,
    )


if __name__ == "__main__":
    main()
