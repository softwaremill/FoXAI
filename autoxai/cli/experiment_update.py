"""File contains CLI application for updating W&B experiment artifacts."""

import os
from typing import Any, Dict, List, cast

import hydra
import pandas as pd
import pytorch_lightning as pl
import torch
import wandb
from omegaconf import DictConfig
from torchvision.io import ImageReadMode, read_image

from autoxai.cli.config_model import ConfigDataModel, MethodDataModel
from autoxai.context_manager import AutoXaiExplainer, Explainers, ExplainerWithParams


def download_upload_metadata(api_run: wandb.apis.public.Run) -> Dict[str, Any]:
    """Fetch uploaded data samples metadata from logs history.

    Args:
        run_api: WandB Api object.

    Returns:
        Metadata of upload data sample entry.
    """
    # get uploaded sample data from experiment run
    history: pd.DataFrame = api_run.history()

    # get log entry with saving all sample data images from log history
    upload_data_metadata = history["validation_data"][
        history["validation_data"].notnull()
    ].iloc[0]

    return upload_data_metadata


def get_labels(upload_data_metadata: Dict[str, Any]) -> List[int]:
    """Extract and convert ground truth labels from image captions.

    Args:
        upload_data_metadata: Metadata of upload data sample entry.

    Returns:
        List of labels.
    """
    # get ground truth labels from captions
    captions: List[str] = upload_data_metadata["captions"]
    labels: List[int] = [int(val.replace("label: ", "")) for val in captions]
    return labels


def fetch_model_checkpoints(
    api: wandb.Api, model_artifact_root: str, project: str
) -> List[str]:
    """Fetch all model checkpoints and return local paths to them.

    Args:
        api: WandB Api object.
        model_artifact_root: Path to local directory where model checkpoints
            will be saved.
        project: Name of a project.

    Returns:
        List of paths to saved downloaded model checkpoints.
    """
    # https://stackoverflow.com/questions/68952727/wandb-get-a-list-of-all-artifact-collections-and-all-aliases-of-those-artifacts
    # list all artifacts (saved model checkpoints)
    collections = list(
        api.artifact_type(type_name="model", project=project).collections()
    )
    model_paths: List[str] = []
    for coll in collections:
        for artifact in coll.versions():
            model_version_root = os.path.join(
                model_artifact_root,
                artifact._artifact_name,  # pylint: disable = (protected-access)
            )
            datadir = artifact.download(root=model_version_root)
            model_paths.append(os.path.join(datadir, "model.ckpt"))

    return model_paths


def load_images_to_tensors(
    filenames: List[str], artifacts_path: str
) -> List[torch.Tensor]:
    """Load images from local directory.

    Args:
        filenames: List of filepaths to images from metadata.
        artifacts_path: Root directory of downloaded images.

    Returns:
        List of tensors representing images.
    """
    # create dataset from downloaded data samples
    image_list: List[torch.Tensor] = []
    for filename in filenames:
        path: str = os.path.join(artifacts_path, filename)
        image: torch.Tensor = read_image(path=path, mode=ImageReadMode.GRAY)
        image_list.append(image)

    return image_list


def load_config(config: DictConfig) -> ConfigDataModel:
    """Load and validate config with explainers to apply.

    Args:
        config: DictConfig object loaded from yaml config file.

    Returns:
        ConfigDataModel object.
    """

    method_config_list: List[MethodDataModel] = []
    for entry in config["explainers"]:
        explainer_config: Dict[str, Any] = entry["explainer_with_params"]
        explainer: ExplainerWithParams = ExplainerWithParams(
            explainer_name=Explainers[explainer_config["explainer_name"]],
            kwargs=explainer_config.get("kwargs", {}),
        )
        method_config_list.append(
            MethodDataModel(
                explainer_with_params=explainer,
                artifact_name=entry.get("artifact_name", None),
            )
        )

    config_data = ConfigDataModel(method_config_list=method_config_list)
    return config_data


@hydra.main()
def main(cfg: DictConfig) -> None:  # pylint: disable = (too-many-locals)
    """Entry point for CLI application."""

    model_class: pl.LightningModule = hydra.utils.instantiate(cfg.classifier)
    wandb.login()

    config: ConfigDataModel = load_config(cfg)

    # resume experiment run that has to be updated
    run = wandb.init(
        project=cfg.experiment,
        id=cfg.run_id,
        resume="allow",
    )
    if run is None:
        return

    artifacts_path: str = run.settings.files_dir
    image_artifacts_path: str = os.path.join(artifacts_path, "media", "images")
    if not os.path.exists(image_artifacts_path):
        os.makedirs(image_artifacts_path)

    api = wandb.Api()
    run_api: wandb.apis.public.Run = api.run(
        f"{cfg.username}/{cfg.experiment}/{cfg.run_id}"
    )

    upload_data_metadata = download_upload_metadata(api_run=run_api)

    filenames: List[str] = upload_data_metadata["filenames"]
    # download all data sample images
    for filename in filenames:
        run_api.file(filename).download(root=artifacts_path)

    labels: List[int] = get_labels(upload_data_metadata=upload_data_metadata)

    model_artifact_root: str = os.path.join(run.settings.tmp_dir, "models")
    model_paths: List[str] = fetch_model_checkpoints(
        api=api,
        model_artifact_root=model_artifact_root,
        project=cfg.experiment,
    )

    image_list: List[torch.Tensor] = load_images_to_tensors(
        filenames=filenames,
        artifacts_path=artifacts_path,
    )

    # sort paths from the earliest model to the latest
    # wand.log has internal counter and to have order in explanations we have to sort
    # them ascending by version
    sorted_paths = sorted(
        model_paths, key=lambda x: int(x.split("/")[-2].split(":v")[-1])
    )
    # filter out artifacts from different runs
    sorted_paths = [
        val for val in sorted_paths if cfg.run_id in val.split("/")[-2].split(":")[0]
    ]

    for explainer_config in config.method_config_list:
        artifact_name: str
        if explainer_config.artifact_name is None:
            artifact_name = explainer_config.explainer_with_params.explainer_name.name
        else:
            artifact_name = explainer_config.artifact_name

        for path in sorted_paths:
            model = model_class.load_from_checkpoint(path, batch_size=1, data_dir=".")

            explanations: List[wandb.Image] = []
            for input_data, label in zip(image_list, labels):
                with AutoXaiExplainer(
                    model=model,
                    explainers=[explainer_config.explainer_with_params],
                    target=label,
                ) as xai_model:
                    input_data = input_data.float()
                    _, attributes_dict = xai_model(
                        input_data.to(cast(torch.device, xai_model.model.device))
                    )
                    explainer_name: str = (
                        explainer_config.explainer_with_params.explainer_name.name
                    )
                    attributes: torch.Tensor = attributes_dict[explainer_name]
                    explanations.append(
                        wandb.Image(attributes, caption=f"label: {label}")
                    )

            if explanations:
                wandb.log({artifact_name: explanations})
