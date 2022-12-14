import os

import matplotlib
import numpy as np
import streamlit as st
import torch
from settings import Settings
from matplotlib.backends.backend_agg import FigureCanvasAgg as FigureCanvas

from src.cache_manager import LocalDirCacheManager
from src.explainer import IntegratedGradientsCVExplainer


def load_subdir(path):
    subdir = {}
    paths = []
    for method in os.listdir(path):
        if not os.path.isdir(os.path.join(path, method)):
            paths.append(method)
        else:
            subdir[method] = load_subdir(os.path.join(path, method))

    retval = subdir
    if paths:
        retval = paths

    return retval


def convert_figure_to_numpy(figure) -> np.ndarray:
    canvas = FigureCanvas(figure)
    canvas.draw()
    buf = canvas.buffer_rgba()
    image = np.asarray(buf)
    return image


def load_input_data(cache_path: str, date_selectbox: str, hash_selectbox: str) -> torch.Tensor:
    cache_manager = LocalDirCacheManager()

    path = os.path.join(
        cache_path,
        date_selectbox,
        hash_selectbox,
        "data",
        st.session_state[Settings.sample_name_key],
    )
    input_data = cache_manager.load_artifact(path)

    return input_data


def load_original_data(cache_path: str, date_selectbox: str, hash_selectbox: str) -> torch.Tensor:
    cache_manager = LocalDirCacheManager()

    path = os.path.join(
        cache_path,
        date_selectbox,
        hash_selectbox,
        "data",
        st.session_state[Settings.sample_name_key],
    )
    original_data = cache_manager.load_artifact(path)

    return original_data


def load_idx_to_labels(cache_path, date_selectbox, hash_selectbox):
    cache_manager = LocalDirCacheManager()
    path = os.path.join(
        cache_path,
        date_selectbox,
        hash_selectbox,
        "labels",
        "idx_to_label.json.pkl",
    )
    return cache_manager.load_artifact(path)


def initialize_session_state(key, value):
    if key not in st.session_state:
        st.session_state[key] = value


def change_state(key, value):
    if Settings.experiment_date_label not in st.session_state or key == Settings.experiment_date_label:
        st.session_state[Settings.experiment_date_label] = value
    if Settings.hash_label not in st.session_state or key == Settings.hash_label:
        st.session_state[Settings.hash_label] = value
    if Settings.method_label not in st.session_state or key == Settings.method_label:
        st.session_state[Settings.method_label] = value


def enable_explain():
    st.session_state.explain = True


def disable_explain():
    st.session_state.explain = False


def selected_date():
    st.session_state.selected_date = False
    disable_explain()
    st.session_state.selected_date = True


def create_figure(path: str, transformed_img: torch.Tensor) -> matplotlib.pyplot.Figure:
    attributions = torch.Tensor(np.load(path))
    return IntegratedGradientsCVExplainer().visualize(attributions, transformed_img)