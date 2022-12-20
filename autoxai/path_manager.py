"""Helper class for handling paths creation."""
import os
from dataclasses import dataclass


@dataclass
class ExperimentDataClass:  # pylint: disable = (too-many-instance-attributes)
    """Class to gather information about directory structure of an experiment."""

    base_path: str
    date: str
    uuid: str
    data_dir: str = "data"
    model_dir: str = "model"

    @property
    def path(self):
        """Path to experiment directory."""
        return os.path.join(self.base_path, self.date, self.uuid)

    @property
    def path_to_data(self):
        """Path to experiment data directory."""
        return os.path.join(self.path, self.data_dir)

    @property
    def path_to_model(self):
        """Path to experiment model directory."""
        return os.path.join(self.path, self.model_dir)
