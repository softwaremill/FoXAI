"""Source: https://pytorch-lightning.readthedocs.io/en/\
    stable/notebooks/lightning_examples/mnist-hello-world.html"""

import os

import torch
from pytorch_lightning import Trainer
from pytorch_lightning.loggers import WandbLogger
from streamlit_app.mnist_model import LitMNIST

import wandb
from autoxai.callbacks.wandb_callback import WandBCallback
from autoxai.context_manager import Explainers, ExplainerWithParams


def main() -> None:  # pylint: disable = (duplicate-code)
    """Entry point of application."""

    data_dir: str = os.environ.get("PATH_DATASETS", ".")
    project_name: str = os.environ.get("PROJECT_NAME", "mnist-classifier")
    accelerator: str = os.environ.get("ACCELERATOR", "cpu")
    batch_size: int = int(os.environ.get("BATCH_SIZE", "64"))
    max_epochs: int = int(os.environ.get("EPOCHS", "3"))
    wandb.login()
    wandb_logger = WandbLogger(project=project_name, log_model="all")
    callback = WandBCallback(
        wandb_logger=wandb_logger,
        explainers=[
            ExplainerWithParams(
                explainer_name=Explainers.CV_INTEGRATED_GRADIENTS_EXPLAINER
            ),
            ExplainerWithParams(explainer_name=Explainers.CV_GRADIENT_SHAP_EXPLAINER),
        ],
        idx_to_label={index: index for index in range(0, 10)},
    )
    model = LitMNIST(data_dir=data_dir, batch_size=batch_size)
    trainer = Trainer(
        accelerator=accelerator,
        devices=1 if torch.cuda.is_available() else None,
        max_epochs=max_epochs,
        logger=wandb_logger,
        callbacks=[callback],
    )
    trainer.tune(model)
    trainer.fit(model)
    trainer.test()


if __name__ == "__main__":
    main()
