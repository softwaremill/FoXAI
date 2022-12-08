"""Source: https://pytorch-lightning.readthedocs.io/en/\
    stable/notebooks/lightning_examples/mnist-hello-world.html"""

import datetime
import os
import uuid

import torch
from pytorch_lightning import LightningModule, Trainer
from pytorch_lightning.callbacks.progress import TQDMProgressBar
from torch.nn import functional as F
from torch.utils.data import DataLoader, random_split
from torchmetrics import Accuracy
from torchvision import transforms
from torchvision.datasets import MNIST

from src.cache_manager import LocalDirCacheManager
from src.callback import AutoXAIPytorchLightningCallback
from src.path_manager import ExperimentDataClass


class LitMNIST(
    LightningModule
):  # pylint: disable = (too-many-ancestors, too-many-instance-attributes)
    """Model to classify MNIST images."""

    def __init__(
        self,
        batch_size: int,
        data_dir: str,
        hidden_size: int = 64,
        learning_rate: float = 2e-4,
    ):
        """Initialize class.

        Args:
            data_dir: Path to dataset. Defaults to PATH_DATASETS.
            hidden_size: Network hidden layer size. Defaults to 64.
            learning_rate: Learning rate parameter. Defaults to 2e-4.
        """

        super().__init__()

        # Set our init args as class attributes
        self.data_dir = data_dir
        self.hidden_size = hidden_size
        self.learning_rate = learning_rate
        self.batch_size = batch_size

        # Hardcode some dataset specific attributes
        self.num_classes = 10
        self.dims = (1, 28, 28)
        channels, width, height = self.dims
        self.transform = transforms.Compose(
            [
                transforms.ToTensor(),
                transforms.Normalize((0.1307,), (0.3081,)),
            ]
        )

        # Define PyTorch model
        self.model = torch.nn.Sequential(
            torch.nn.Flatten(),
            torch.nn.Linear(channels * width * height, hidden_size),
            torch.nn.ReLU(),
            torch.nn.Dropout(0.1),
            torch.nn.Linear(hidden_size, hidden_size),
            torch.nn.ReLU(),
            torch.nn.Dropout(0.1),
            torch.nn.Linear(hidden_size, self.num_classes),
        )

        mnist_full = MNIST(self.data_dir, train=True, transform=self.transform)
        self.mnist_train, self.mnist_val = random_split(mnist_full, [55000, 5000])

        self.mnist_test = MNIST(self.data_dir, train=False, transform=self.transform)

        num_classes = len(self.mnist_test.classes)

        self.val_accuracy = Accuracy(task="multiclass", num_classes=num_classes)
        self.test_accuracy = Accuracy(task="multiclass", num_classes=num_classes)

    def forward(  # pylint: disable = (arguments-differ)
        self, x: torch.Tensor
    ) -> torch.Tensor:
        x = self.model(x)
        return F.log_softmax(x, dim=1)

    def training_step(  # pylint: disable = (arguments-differ)
        self,
        batch: torch.Tensor,
        batch_idx: int,  # pylint: disable = (unused-argument)
    ) -> torch.Tensor:
        samples, targets = batch
        logits = self(samples)
        loss = F.nll_loss(logits, targets)
        return loss

    def validation_step(  # pylint: disable = (arguments-differ)
        self,
        batch: torch.Tensor,
        batch_idx: int,  # pylint: disable = (unused-argument)
    ) -> None:
        samples, targets = batch
        logits = self(samples)
        loss = F.nll_loss(logits, targets)
        preds = torch.argmax(logits, dim=1)  # pylint: disable = (no-member)
        self.val_accuracy.update(preds, targets)  # pylint: disable = (no-member)

        # Calling self.log will surface up scalars for you in TensorBoard
        self.log("val_loss", loss, prog_bar=True)
        self.log("val_acc", self.val_accuracy, prog_bar=True)

    def test_step(  # pylint: disable = (arguments-differ)
        self,
        batch: torch.Tensor,
        batch_idx: int,  # pylint: disable = (unused-argument)
    ) -> None:
        samples, targets = batch
        logits = self(samples)
        loss = F.nll_loss(logits, targets)
        preds = torch.argmax(logits, dim=1)  # pylint: disable = (no-member)
        self.test_accuracy.update(preds, targets)  # pylint: disable = (no-member)

        # Calling self.log will surface up scalars for you in TensorBoard
        self.log("test_loss", loss, prog_bar=True)
        self.log("test_acc", self.test_accuracy, prog_bar=True)

    def configure_optimizers(self) -> torch.optim.Optimizer:
        optimizer = torch.optim.Adam(self.parameters(), lr=self.learning_rate)
        return optimizer

    def train_dataloader(self) -> DataLoader:
        return DataLoader(self.mnist_train, batch_size=self.batch_size)

    def val_dataloader(self) -> DataLoader:
        return DataLoader(self.mnist_val, batch_size=self.batch_size)

    def test_dataloader(self) -> DataLoader:
        return DataLoader(self.mnist_test, batch_size=self.batch_size)


def main() -> None:
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
    callback = AutoXAIPytorchLightningCallback(
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
