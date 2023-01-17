import torch
from torch.nn import functional as F


class SampleModel(torch.nn.Module):
    """Sample pytorch model for experiment."""

    def __init__(
        self,
        in_channels: int = 1,
        resolution: int = 224,
    ):
        super().__init__()
        self.stride: int = 16
        self.out_channels: int = 16
        self.conv1 = torch.nn.Conv2d(
            in_channels=in_channels,
            out_channels=self.out_channels,
            kernel_size=5,
            stride=16,
        )

        output_channels: int = ((resolution // self.stride) ** 2) * self.out_channels
        self.cls = torch.nn.Sequential(
            torch.nn.Dropout(),
            torch.nn.Linear(in_features=output_channels, out_features=1, bias=True),
        )
        self.sigmoid = torch.nn.Sigmoid()
        self.name = "SampleModel"

    def forward(self, x_tensor: torch.Tensor) -> torch.Tensor:
        """Forward methid for the module."""
        x_tensor = F.relu(self.conv1(x_tensor))
        x_tensor = x_tensor.view(x_tensor.size(0), -1)
        x_tensor = self.cls(x_tensor)
        x_tensor = self.sigmoid(x_tensor)
        return x_tensor


class AutoEncoder(torch.nn.Module):
    """Sample pytorch auto-encoder model for experiment."""

    def __init__(self):
        super().__init__()
        self.encoder = torch.nn.Sequential(
            torch.nn.Linear(28 * 28, 64), torch.nn.ReLU(), torch.nn.Linear(64, 3)
        )
        self.decoder = torch.nn.Sequential(
            torch.nn.Linear(3, 64), torch.nn.ReLU(), torch.nn.Linear(64, 28 * 28)
        )

    def forward(self, x_tensor: torch.Tensor) -> torch.Tensor:
        # training_step defines the train loop.
        # it is independent of forward
        x_tensor = x_tensor.view(x_tensor.size(0), -1)
        z = self.encoder(x_tensor)
        x_hat = self.decoder(z)
        return x_hat


class CNN(torch.nn.Module):
    """Sample pytorch CNN model for experiment."""

    def __init__(self):
        super().__init__()
        self.conv1 = torch.nn.Conv2d(1, 20, 5)
        self.conv2 = torch.nn.Conv2d(20, 20, 5)

    def forward(self, x_tensor: torch.Tensor) -> torch.Tensor:
        x_tensor = F.relu(self.conv1(x_tensor))
        return F.relu(self.conv2(x_tensor))
