"""Library callback for pytorch-lightning."""
import logging
import os
from pathlib import Path
from typing import Dict, Optional

import numpy as np
import pytorch_lightning as pl
import tensorflow as tf
import torch

from src.cache_manager import LocalDirCacheManager
from src.explainer import CVExplainer
from src.path_manager import ExperimentDataClass

logger = logging.getLogger(__name__)


class CustomPytorchLightningCallback(pl.callbacks.Callback):
    """Library callback for pytorch-lightning."""

    def __init__(
        self,
        idx_to_label: Dict[int, str],
        experiment: Optional[ExperimentDataClass] = None,
        cache_manager: Optional[LocalDirCacheManager] = None,
    ):
        """Initialize Callback class."""
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

        model_path: str = os.path.join(
            self.experiment.path,
            "training",
            str(trainer.current_epoch),
            "model.onnx",
        )
        path = Path(model_path)
        if not os.path.exists(path.parent):
            os.makedirs(path.parent)

        # TODO: check if no validation dataloader is provided it will work  # pylint: disable = (fixme)
        input_sample: Optional[torch.Tensor] = None
        if trainer.val_dataloaders:
            for batch in trainer.val_dataloaders[0]:
                items, _ = batch
                input_sample = items[0]
                break

        if input_sample is None:
            logger.warning(
                "Unable to obtain `input_sample` for exportig model to ONNX."
            )

        pl_module.to_onnx(
            model_path,
            input_sample,
            export_params=True,
        )


class TensorboardCallback(pl.callbacks.Callback):
    """Library callback for pytorch-lightning."""

    def __init__(
        self,
        explainer: CVExplainer,
        idx_to_label: Dict[int, str],
        experiment: Optional[ExperimentDataClass] = None,
        cache_manager: Optional[LocalDirCacheManager] = None,
    ):
        """Initialize Callback class."""
        super().__init__()
        self.experiment = experiment
        self.cache_manager = cache_manager
        self.idx_to_label = idx_to_label
        self.explainer = explainer
        self.file_writer: Optional[tf.summary.SummaryWriter] = None

    def convert_tensor(self, data: torch.Tensor) -> np.ndarray:
        """Convert and transpoze tensor to numpy array.

        The function transforms a tensor in 4D or 3D form with dimensions
        `(K x H x W x C)` to the form `(K x C x H x W)`, where `K` is number of
        samples, `C` is colour dimension, `H` is height and `W` is width.
        Additionally object is moved from accelerators (e.g. GPU) to CPU.

        Args:
            data: Tensor to convert.

        Returns:
            Converted and transpozed numpy array on CPU.
        """
        if len(data.shape) == 4 and data.shape[1] == 1:
            data = data.expand(1, 3, data.shape[2], data.shape[3])
        elif data.shape[0] == 1:
            data = data.expand(1, 3, data.shape[1], data.shape[2])

        # change dimension from (K x C x H x W) to (K x H x W x C)
        # where C is colour dimension, H and W are height and width dimensions
        attributions_np: np.ndarray = data.cpu().detach().numpy()
        attributions_np = np.transpose(attributions_np, (0, 2, 3, 1))
        return attributions_np

    def on_validation_epoch_end(
        self,
        trainer: pl.Trainer,
        pl_module: pl.LightningModule,  # pylint: disable = (unused-argument)
    ) -> None:
        """Called when the validation epoch ends."""
        if self.experiment is None:
            return

        epoch: int = trainer.current_epoch
        if self.file_writer is None:
            self.file_writer = self._configure_file_writer(trainer)

        if self.file_writer is None:
            return

        input_sample = self._get_sample_data(trainer)
        org_img = self.convert_tensor(input_sample)
        with self.file_writer.as_default():
            tf.summary.image("Original image", org_img, step=epoch)

        logger.info("Generate explanations")
        for index in range(0, len(self.idx_to_label.keys())):
            attribution = self._explain_sample(pl_module, input_sample, index)

            if attribution is not None:
                with self.file_writer.as_default():
                    tf.summary.image(
                        f"{self.explainer.algorithm_name} class: {index}",
                        self.convert_tensor(attribution),
                        step=epoch,
                    )

    def _explain_sample(
        self,
        pl_module: pl.LightningModule,
        input_sample: torch.Tensor,
        class_index: int,
    ) -> Optional[torch.Tensor]:
        """Get the model attribution on the given sample.

        Args:
            pl_module: Model object.
            input_sample: Data sample to explain.
            class_index: Index of class against which attribution was calculated.

        Returns:
            Tensor representing attribution of a model or None.
        """
        attribution: Optional[torch.Tensor] = None
        if input_sample is not None:
            input_data: torch.Tensor = input_sample.to(
                "cuda" if pl_module.on_gpu else "cpu"
            )
            if isinstance(input_data, torch.Tensor):
                attribution = self.explainer.calculate_features(
                    model=pl_module,
                    input_data=input_data,
                    pred_label_idx=class_index,
                )

        return attribution

    def _get_sample_data(self, trainer: pl.Trainer) -> torch.Tensor:
        """Get sample data to explain during current epoch.

        Args:
            trainer: Trainer object.

        Returns:
            Tensor representing one sample data.
        """
        input_sample: torch.Tensor
        if trainer.val_dataloaders:
            for batch in trainer.val_dataloaders[0]:
                items, _ = batch
                input_sample = items[np.random.randint(0, len(items))]
                break

        return input_sample

    def _configure_file_writer(
        self, trainer: pl.Trainer
    ) -> Optional[tf.summary.SummaryWriter]:
        """Setup `file_writer` field to match experiment log directory.

        Args:
            trainer: Trainer object.
        """
        return_value: Optional[tf.summary.SummaryWriter] = None
        for log in trainer.loggers:
            if isinstance(log, pl.loggers.tensorboard.TensorBoardLogger):
                version: str = str(log.version)
                return_value = tf.summary.create_file_writer(
                    f"lightning_logs/version_{str(version)}/"
                )

        return return_value
