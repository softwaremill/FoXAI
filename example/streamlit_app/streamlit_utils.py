"""File contains functions """

import os
from typing import Any, Dict, List, Union, cast

import streamlit as st
import torch
from settings import Settings

from src.cache_manager import LocalDirCacheManager


def load_subdir(path: str) -> Union[Dict[str, Any], List[str]]:
    """Load sub-directory.

    Args:
        path: Path to directory.

    Returns:
        List of directory content or dictionary with directory names and content.
    """
    subdir: Dict[str, List[str]] = {}
    paths: List[str] = []
    for method in os.listdir(path):
        if not os.path.isdir(os.path.join(path, method)):
            paths.append(method)
        else:
            subdir[method] = load_subdir(os.path.join(path, method))

    retval = subdir
    if paths:
        retval = paths

    return retval


def load_input_data(
    cache_path: str, selected_date: str, experiment_hash: str
) -> torch.Tensor:
    """Load preprocessed input data sample from given path.

    Args:
        cache_path: Path to cache directory.
        selected_date: Selected date.
        experiment_hash: Selected experiment hash.

    Returns:
        Preprocessed data sample.
    """
    cache_manager = LocalDirCacheManager()

    path = os.path.join(
        cache_path,
        selected_date,
        experiment_hash,
        "data",
        st.session_state[Settings.sample_name_key],
    )
    input_data = cache_manager.load_artifact(path)

    return input_data


def load_original_data(
    cache_path: str, selected_date: str, experiment_hash: str
) -> torch.Tensor:
    """Load original data sample from given path.

    Args:
        cache_path: Path to cache directory.
        selected_date: Selected date.
        experiment_hash: Selected experiment hash.

    Returns:
        Original data sample.
    """
    cache_manager = LocalDirCacheManager()

    path = os.path.join(
        cache_path,
        selected_date,
        experiment_hash,
        "data",
        st.session_state[Settings.sample_name_key],
    )
    original_data = cache_manager.load_artifact(path)

    return original_data


def load_idx_to_labels(
    cache_path: str,
    selected_date: str,
    experiment_hash: str,
) -> Dict[int, str]:
    """Load index to class label mapping.

    Args:
        cache_path: Path to cache directory.
        selected_date: Selected date.
        experiment_hash: Selected experiment hash.

    Returns:
        Index to class label mapping dictionary.
    """
    cache_manager = LocalDirCacheManager()
    path = os.path.join(
        cache_path,
        selected_date,
        experiment_hash,
        "labels",
        "idx_to_label.json.pkl",
    )
    return cast(Dict[int, str], cache_manager.load_artifact(path))


def initialize_session_state(key: str, value: Any) -> None:
    """Initialize `st.session_state` with given key and value.

    Args:
        key: Key in session state.
        value: Value of key in session state.
    """
    if key not in st.session_state:
        st.session_state[key] = value


def change_state(key: str, value: Any) -> None:
    """Update key state with given value in `st.session_state`.

    Args:
        key: Key in session state.
        value: Value of key in session state.
    """
    if (
        Settings.experiment_date_label not in st.session_state
        or key == Settings.experiment_date_label
    ):
        st.session_state[Settings.experiment_date_label] = value
    if Settings.hash_label not in st.session_state or key == Settings.hash_label:
        st.session_state[Settings.hash_label] = value
    if Settings.method_label not in st.session_state or key == Settings.method_label:
        st.session_state[Settings.method_label] = value


def enable_explain() -> None:
    """Set `explain` flag in `st.session_state` to `True`."""
    st.session_state[Settings.explain_key] = True


def disable_explain():
    """Set `explain` flag in `st.session_state` to `False`."""
    st.session_state[Settings.explain_key] = False
