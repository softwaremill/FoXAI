import os
import pickle
from typing import Any
from abc import ABC


class CacheManager(ABC):
    """Abstract class for cache management."""

    def save_artifact(self, path: str, obj: Any):  # pylint: disable=unused-argument
        """Save artifact to given path.

        Args:
            path (str): Desired path to saved object.
            obj (Any): Object to be saved.
        """
        ...

    def load_artifact(self, path: str):  # pylint: disable=unused-argument
        """Load artifact from given path.

        Args:
            path (str): Path to saved object.
        """
        ...


class LocalDirCacheManager(CacheManager):
    """Cache manager with saving to local directory."""

    def save_artifact(self, path: str, obj: Any) -> None:
        """Save artifact to given path in local directory.

        Args:
            path (str): Desired path to saved object.
            obj (Any): Object to be saved.
        """
        path = f"{path}.pkl"
        base_filepath = os.path.dirname(path)
        if not os.path.exists(base_filepath):
            os.makedirs(base_filepath)

        with open(path, "wb") as f:
            pickle.dump(obj, f)

    def load_artifact(self, path: str) -> Any:
        """Load artifact from given path in local directory.

        Args:
            path (str): Path to saved object.
        """
        with open(path, "rb") as f:
            data = pickle.load(f)

        return data
