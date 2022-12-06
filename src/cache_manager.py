"""Class to perform caching on local storage."""
import os
import pickle
from abc import ABC
from typing import Any


class CacheManager(ABC):
    """Abstract class for cache management."""

    def save_artifact(self, path: str, obj: Any):  # pylint: disable=unused-argument
        """Save artifact to given path.

        Args:
            path: Desired path to saved object.
            obj: Object to be saved.
        """

    def load_artifact(self, path: str):  # pylint: disable=unused-argument
        """Load artifact from given path.

        Args:
            path: Path to saved object.
        """


class LocalDirCacheManager(CacheManager):
    """Cache manager with saving to local directory."""

    def save_artifact(self, path: str, obj: Any) -> None:
        """Save artifact to given path in local directory.

        Args:
            path: Desired path to saved object.
            obj: Object to be saved.
        """
        path = f"{path}.pkl"
        base_filepath = os.path.dirname(path)
        if not os.path.exists(base_filepath):
            os.makedirs(base_filepath)

        with open(path, "wb") as file:
            pickle.dump(obj, file)

    def load_artifact(self, path: str) -> Any:
        """Load artifact from given path in local directory.

        Args:
            path: Path to saved object.
        """
        with open(path, "rb") as file:
            data = pickle.load(file)

        return data
