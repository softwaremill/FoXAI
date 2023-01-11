"""Helper class for handling paths creation."""
import os
from dataclasses import dataclass


@dataclass
class ExperimentDataClass:  # pylint: disable = (too-many-instance-attributes)
    """Class to gather information about directory structure of an experiment."""

    base_path: str
    """Path to root directory of cache."""

    date: str
    """Date of experiment in format YYYY-MM-DD."""

    uuid: str
    """ID of experiment."""

    data_dir: str = "data"
    """Name of directory with cached data samples."""

    model_dir: str = "model"
    """Name of directory with cached model checkpoints."""

    @property
    def path(self) -> str:
        """Path to experiment directory."""
        return os.path.join(self.base_path, self.date, self.uuid)

    @property
    def path_to_data(self) -> str:
        """Path to experiment data directory."""
        return os.path.join(self.path, self.data_dir)

    @property
    def path_to_model(self) -> str:
        """Path to experiment model directory."""
        return os.path.join(self.path, self.model_dir)
