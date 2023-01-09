"""Callback for Weights and Biases."""
from collections import defaultdict
from typing import Dict, Generator, List, Optional, Tuple

import matplotlib
import pytorch_lightning as pl
import torch
from pytorch_lightning.loggers import WandbLogger
from torch.utils.data import DataLoader

import wandb
from autoxai.context_manager import AutoXaiExplainer, ExplainerWithParams
from autoxai.explainer.base_explainer import CVExplainer


class WandBCallback(pl.callbacks.Callback):
    """Library callback for Weights and Biases."""

    def __init__(  # pylint: disable = (too-many-arguments)
        self,
        wandb_logger: WandbLogger,
        explainers: List[ExplainerWithParams],
        idx_to_label: Dict[int, str],
        max_artifacts: int = 3,
    ):
        """Initialize Callback class.

        Args:
            wandb_logger: Pytorch-lightning wandb logger.
            idx_to_label: Index to label mapping.
            explainers: List of explainer algorithms of type ExplainerWithParams.
            idx_to_label: Dictionary with mapping from model index to label.
            max_artifacts: Number of maximum number of artifacts to be logged.
                Defaults to 3.
        """
        super().__init__()
        self.explainers = explainers
        self.wandb_logger = wandb_logger
        self.idx_to_label = idx_to_label
        self.max_artifacts = max_artifacts

    def _save_idx_mapping(self) -> None:
        """Saving index to label mapping to experiment logs directory."""
        self.wandb_logger.log_table(
            key="idx2label",
            columns=["index", "label"],
            data=[[key, val] for key, val in self.idx_to_label.items()],
        )

    def iterate_dataloader(
        self, dataloader_list: List[DataLoader], max_items: int
    ) -> Generator[Tuple[torch.Tensor, torch.Tensor], None, None]:
        """Iterate over dataloader list with constraint on max items returned.

        Args:
            dataloader: Trainer dataloader.
            max_items: Max items to return.

        Yields:
            Tuple containing training sample and corresponding label.
        """
        index: int = 0
        dataloader: DataLoader
        items: torch.Tensor
        predictions: torch.Tensor
        for dataloader in dataloader_list:
            for batch in dataloader:
                items, predictions = batch
                for item, prediction in zip(items, predictions):
                    if index >= max_items:
                        break

                    index += 1
                    yield item, prediction

    def explain(  # pylint: disable = (too-many-arguments)
        self,
        model: pl.LightningModule,
        item: torch.Tensor,
        prediction: torch.Tensor,
        attributes_dict: Dict[str, List[torch.Tensor]],
        caption_dict: Dict[str, List[str]],
        figures_dict: Dict[str, List[matplotlib.pyplot.Figure]],
    ) -> Tuple[
        Dict[str, List[torch.Tensor]],
        Dict[str, List[str]],
        Dict[str, List[matplotlib.pyplot.Figure]],
    ]:
        """Calculate explainer attributes, creates captions and figures.

        Args:
            model: Model to explain.
            item: Input data sample tensor.
            prediction: Sample label.
            attributes_dict: List of attributes for every explainer and sample.
            caption_dict: List of captions for every explainer and sample.
            figures_dict: List of figures for every explainer and sample.

        Returns:
            Tuple of maps containing attributes, captions and figures for
            every explainer and sample.
        """
        with AutoXaiExplainer(
            model=model,
            explainers=self.explainers,
            target=int(prediction.item()),
        ) as xai_model:
            _, attributes = xai_model(item.to(model.device))

        for explainer in self.explainers:
            explainer_name: str = explainer.explainer_name.name
            explainer_attributes: torch.Tensor = attributes[explainer_name]
            attributes_dict[explainer_name].append(explainer_attributes)
            caption_dict[explainer_name].append(f"label: {prediction}")
            figure = CVExplainer.visualize(
                attributions=explainer_attributes,
                transformed_img=item,
            )
            figures_dict[explainer_name].append(figure)

        return attributes_dict, caption_dict, figures_dict

    def on_train_start(
        self,
        trainer: pl.Trainer,
        pl_module: pl.LightningModule,  # pylint: disable = (unused-argument)
    ) -> None:
        """Save index to labels mapping and validation samples to experiment
        at `fit`.

        Args:
            trainer: Trainer object.
            pl_module: Model to explain.
        """
        if trainer.val_dataloaders is None:
            return

        self._save_idx_mapping()

        image_matrix: Optional[torch.Tensor] = None
        image_labels: List[str] = []

        for item, prediction in self.iterate_dataloader(
            dataloader_list=trainer.val_dataloaders, max_items=self.max_artifacts
        ):
            if image_matrix is None:
                image_matrix = item
            else:
                image_matrix = torch.cat(  # pylint: disable = (no-member)
                    [image_matrix, item]
                )

            image_labels.append(f"label: {prediction.item()}")

        if image_matrix is None:
            return

        list_of_images: List[torch.Tensor] = list(torch.split(image_matrix, 1))
        self.wandb_logger.log_image(
            key="validation_data",
            images=list_of_images[: min(len(list_of_images), self.max_artifacts)],
            caption=image_labels[: min(len(image_labels), self.max_artifacts)],
        )

    def on_validation_epoch_end(  # pylint: disable = (too-many-arguments, too-many-locals)
        self,
        trainer: pl.Trainer,
        pl_module: pl.LightningModule,
    ) -> None:
        """Export model's state dict in log directory on validation epoch end.

        Args:
            trainer: Trainer object.
            pl_module: Model to explain.
        """
        if trainer.val_dataloaders is None:
            return

        attributes_dict: Dict[str, List[torch.Tensor]] = defaultdict(list)
        caption_dict: Dict[str, List[str]] = defaultdict(list)
        figures_dict: Dict[str, List[matplotlib.pyplot.Figure]] = defaultdict(list)

        for item, prediction in self.iterate_dataloader(
            dataloader_list=trainer.val_dataloaders,
            max_items=self.max_artifacts,
        ):
            attributes_dict, caption_dict, figures_dict = self.explain(
                model=pl_module,
                item=item,
                prediction=prediction,
                attributes_dict=attributes_dict,
                caption_dict=caption_dict,
                figures_dict=figures_dict,
            )

        self.log_explanations(
            attributes_dict=attributes_dict,
            caption_dict=caption_dict,
            figures_dict=figures_dict,
        )

    def log_explanations(
        self,
        attributes_dict: Dict[str, List[torch.Tensor]],
        caption_dict: Dict[str, List[str]],
        figures_dict: Dict[str, List[matplotlib.pyplot.Figure]],
    ) -> None:
        """Log explanation artifacts to W&B experiment.

        Args:
            attributes_dict: Tensor attributes for every sample and every explainer.
            caption_dict: Caption for every sample and every explainer.
            figures_dict: Figure with attributes for every sample and every explainer.
        """
        # upload artifacts to the wandb experiment
        for explainer in self.explainers:
            explainer_name: str = explainer.explainer_name.name
            self.wandb_logger.log_image(
                key=f"{explainer_name}",
                images=[val.numpy() for val in attributes_dict[explainer_name]],
                caption=caption_dict[explainer_name],
            )

            # matplotlib Figures can not be directly logged via WandbLogger
            # we have to use native Run object from wandb which is more powerfull
            wandb_image_list: List[wandb.Image] = []
            for figure in figures_dict[explainer_name]:
                wandb_image_list.append(wandb.Image(figure))

            self.wandb_logger.experiment.log(
                {f"{explainer_name}_explanations": wandb_image_list}
            )
