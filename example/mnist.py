"""Source: https://pytorch-lightning.readthedocs.io/en/\
    stable/notebooks/lightning_examples/mnist-hello-world.html"""

import datetime
import os
import uuid

import torch
from pytorch_lightning import Trainer
from pytorch_lightning.callbacks.progress import TQDMProgressBar
from streamlit_app.mnist_model import LitMNIST

from autoxai.cache_manager import LocalDirCacheManager
from autoxai.callback import CustomPytorchLightningCallback
from autoxai.path_manager import ExperimentDataClass


def main() -> None:  # pylint: disable = (duplicate-code)
    """Entry point of application."""
    data_dir: str = os.environ.get("PATH_DATASETS", ".")
    batch_size: int = 256 if torch.cuda.is_available() else 64
    max_epochs: int = 3

    experiment = ExperimentDataClass(
        base_path="logs/",
        date=str(datetime.date.today()),
        uuid=str(uuid.uuid1()),
    )
    cache_manager = LocalDirCacheManager()
    callback = CustomPytorchLightningCallback(
        experiment=experiment,
        cache_manager=cache_manager,
        idx_to_label={index: index for index in range(0, 10)},
    )

    model = LitMNIST(data_dir=data_dir, batch_size=batch_size)
    trainer = Trainer(
        accelerator="auto",
        devices=1 if torch.cuda.is_available() else None,
        max_epochs=max_epochs,
        callbacks=[TQDMProgressBar(refresh_rate=20), callback],
    )
    trainer.tune(model)
    trainer.fit(model)
    trainer.test()


if __name__ == "__main__":
    main()
