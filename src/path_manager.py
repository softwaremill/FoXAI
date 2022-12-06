"""Helper class for handling paths creation."""
import os
from dataclasses import dataclass


@dataclass
class ExperimentDataClass:  # pylint: disable = (too-many-instance-attributes)
    """Class to gather information about directory structure of an experiment."""

    base_path: str
    date: str
    uuid: str
    image_path: str
    data_dir: str = "data"
    model_dir: str = "model"
    explanations_dir: str = "explanations"
    figures_dir: str = "figures"

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

    @property
    def path_to_explanations(self):
        """Path to experiment explanations algorithms results directory."""
        return os.path.join(self.path, self.explanations_dir)

    @property
    def image_name(self):
        """Image name."""
        return self.image_path.rsplit("/", maxsplit=1)[-1]

    def generate_path_to_experiment_figures(self, path: str) -> str:
        """Generate path to experiment explanations figures.

        Args:
            path: Path to experiment explanations algorithm directory.

        Returns:
            Path.
        """
        return os.path.join(path, self.figures_dir)
