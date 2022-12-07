"""Library callback for pytorch-lightning."""
import logging
import os
from typing import Dict, Optional

import pytorch_lightning as pl

from src.cache_manager import LocalDirCacheManager
from src.path_manager import ExperimentDataClass

logger = logging.getLogger(__name__)


class AutoXAIPytorchLightningCallback(pl.callbacks.Callback):
    """Library callback for pytorch-lightning."""

    def __init__(
        self,
        experiment: Optional[ExperimentDataClass] = None,
        cache_manager: Optional[LocalDirCacheManager] = None,
        idx_to_label: Optional[Dict[int, str]] = None,
    ):
        """Initialize Callback class."""
        super().__init__()
        self.experiment = experiment
        self.cache_manager = cache_manager
        self.idx_to_label = idx_to_label

    def _save_idx_mapping(self) -> None:
        """Saving index to label mapping to experiment logs directory."""
        if (
            self.idx_to_label is not None
            and self.cache_manager is not None
            and self.experiment is not None
        ):
            self.cache_manager.save_artifact(
                path=os.path.join(self.experiment.path, "labels", "idx_to_label.json"),
                obj=self.idx_to_label,
            )

    def on_sanity_check_end(
        self,
        trainer: pl.Trainer,
        pl_module: pl.LightningModule,  # pylint: disable = (unused-argument)
    ) -> None:
        """Called when fit begins."""
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
                items, predictions = batch
                for item, _ in zip(items, predictions):
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
        """Called when the validation epoch ends."""
        if self.experiment is None:
            return

        epoch: int = trainer.current_epoch
        logger.info("Saving all validation samples to data directory.")
        logger.info(  # pylint: disable = (logging-fstring-interpolation)
            f"Saving model checkpoint at epoch: {str(epoch)}."
        )

        trainer.save_checkpoint(
            os.path.join(
                self.experiment.path,
                "training",
                str(trainer.current_epoch),
                "model.cpkt",
            ),
        )
