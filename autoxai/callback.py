"""Library callback for pytorch-lightning."""
import logging
import os
from pathlib import Path
from typing import Dict, Optional

import pytorch_lightning as pl
import torch

from autoxai.cache_manager import LocalDirCacheManager
from autoxai.path_manager import ExperimentDataClass

logger = logging.getLogger(__name__)


class CustomPytorchLightningCallback(pl.callbacks.Callback):
    """Library callback for pytorch-lightning."""

    def __init__(
        self,
        idx_to_label: Dict[int, str],
        experiment: Optional[ExperimentDataClass] = None,
        cache_manager: Optional[LocalDirCacheManager] = None,
    ):
        """Initialize Callback class.

        Args:
            idx_to_label: Index to label mapping.
            experiment: Helper object for creating paths to store artifacts.
            cache_manager: Helper class to store artifacts in log direcotry.
        """
        super().__init__()
        self.experiment = experiment
        self.cache_manager = cache_manager
        self.idx_to_label = idx_to_label

    def _save_idx_mapping(self) -> None:
        """Saving index to label mapping to experiment logs directory."""
        if self.cache_manager is not None and self.experiment is not None:
            self.cache_manager.save_artifact(
                path=os.path.join(self.experiment.path, "labels", "idx_to_label.json"),
                obj=self.idx_to_label,
            )

    def on_sanity_check_end(
        self,
        trainer: pl.Trainer,
        pl_module: pl.LightningModule,  # pylint: disable = (unused-argument)
    ) -> None:
        """Save index to labels mapping and validation samples to log directory
        before `fit`.

        Args:
            trainer: Trainer object.
            pl_module: Model to explain.
        """
        if (
            self.experiment is None
            or self.cache_manager is None
            or trainer.val_dataloaders is None
        ):
            return

        logger.info("Saving all validation samples to data directory.")
        index: int = 0
        for dataloader in trainer.val_dataloaders:
            for batch in dataloader:
                items, target_labels = batch
                for item, _ in zip(items, target_labels):
                    self.cache_manager.save_artifact(
                        path=os.path.join(self.experiment.path_to_data, str(index)),
                        obj=item,
                    )
                    index += 1

        self._save_idx_mapping()

    def on_validation_epoch_end(
        self,
        trainer: pl.Trainer,
        pl_module: pl.LightningModule,  # pylint: disable = (unused-argument)
    ) -> None:
        """Export model's state dict in log directory on validation epoch end.

        Args:
            trainer: Trainer object.
            pl_module: Model to explain.
        """
        if self.experiment is None:
            return

        epoch: int = trainer.current_epoch
        logger.info("Saving all validation samples to data directory.")
        logger.info(  # pylint: disable = (logging-fstring-interpolation)
            f"Saving model checkpoint at epoch: {str(epoch)}."
        )
        model_path: str = os.path.join(
            self.experiment.path,
            "training",
            str(trainer.current_epoch),
            "model.pt",
        )
        path = Path(model_path)
        if not os.path.exists(path.parent):
            os.makedirs(path.parent)

        torch.save(pl_module.state_dict(), model_path)
