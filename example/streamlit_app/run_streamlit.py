"""File with app entry point and functions to display widgets and views."""

import os
from typing import Any, Dict, cast

import numpy as np
import streamlit as st
from model_utils import get_model_layers, load_model
from settings import Settings
from streamlit_utils import (
    disable_explain,
    initialize_session_state,
    load_idx_to_labels,
    load_input_data,
    load_original_data,
    load_subdir,
)
from torch import fx
from visualization_utils import convert_figure_to_numpy

from src.explainer.base_explainer import CVExplainer
from src.explainer.gradcam import GuidedGradCAMCVExplainer, LayerGradCAMCVExplainer
from src.explainer.gradient_shap import (
    GradientSHAPCVExplainer,
    LayerGradientSHAPCVExplainer,
)
from src.explainer.integrated_gradients import (
    IntegratedGradientsCVExplainer,
    LayerIntegratedGradientsCVExplainer,
)
from src.explainer.lrp import LayerLRPCVExplainer, LRPCVExplainer
from src.explainer.noise_tunnel import (
    LayerNoiseTunnelCVExplainer,
    NoiseTunnelCVExplainer,
)
from src.explainer.occulusion import OcculusionCVExplainer

cache_path = os.environ.get("LOGDIR", "logs")

explainer_map = {
    "occulusion": OcculusionCVExplainer(),
    "integrated_gradients": IntegratedGradientsCVExplainer(),
    "noise_tunnel": NoiseTunnelCVExplainer(),
    "gradient_shap": GradientSHAPCVExplainer(),
    "lrp": LRPCVExplainer(),
    "guided_gradcam": GuidedGradCAMCVExplainer(),
    "layer_integrated_gradients": LayerIntegratedGradientsCVExplainer(),
    "layer_noise_tunnel": LayerNoiseTunnelCVExplainer(),
    "layer_gradient_shap": LayerGradientSHAPCVExplainer(),
    "layer_lrp": LayerLRPCVExplainer(),
    "layer_gradcam": LayerGradCAMCVExplainer(),
}
method_list = list(explainer_map)
layer_explainers = [
    LayerNoiseTunnelCVExplainer,
    LayerGradCAMCVExplainer,
    LayerGradientSHAPCVExplainer,
    LayerIntegratedGradientsCVExplainer,
    LayerLRPCVExplainer,
]


def check_if_layer_explainer(explainer: CVExplainer) -> bool:
    """Check if given explainer is explaining on layer level.

    Args:
        explainer: Checked explainer object.

    Returns:
        If `True` it is explaining layers.
    """
    return any(isinstance(explainer, class_type) for class_type in layer_explainers)


@st.cache
def load_data() -> Dict[str, Any]:
    """Load log directory structure.

    Returns:
        Nested dictionary containing experiments hierarchy.
    """
    return cast(Dict[str, Any], load_subdir(cache_path))


def initialize() -> None:
    """Initialize default state of `st.session_state`."""
    st.session_state[Settings.experiment_label] = load_data()
    initial_date = list(st.session_state[Settings.experiment_label])[-1]
    initialize_session_state(Settings.experiment_date_label, initial_date)
    initial_hash: str = list(st.session_state[Settings.experiment_label][initial_date])[
        0
    ]
    initialize_session_state(
        Settings.hash_label,
        initial_hash,
    )
    initialize_session_state(Settings.explain_key, False)
    initialize_session_state(Settings.window_value, 3)
    initialize_session_state(Settings.stride_value, 3)
    initialize_session_state(Settings.selected_layer_key, -1)
    initialize_session_state(Settings.model_layers_key, [])


def generate_feature_map(
    explainer: CVExplainer,
    selected_date: str,
    experiment_hash: str,
    model: Any,
    **kwargs,
) -> np.ndarray:
    """Generate explanation for given model and given input data sample.

    Args:
        explainer: Object representing XAI algorithm to use.
        selected_date: Selected date.
        experiment_hash: Selected experiment hash.
        model: Model to explain.

    Returns:
        Model's attributes.
    """
    input_data = load_input_data(
        cache_path,
        selected_date,
        experiment_hash,
    )
    original_data = load_original_data(
        cache_path,
        selected_date,
        experiment_hash,
    )

    # captum accepts images in 4D, where the first dimension is number of examples
    # if data has 3 dimensions we need to add one
    if len(input_data.shape) == 3:
        input_data = input_data[None, :]
    if len(original_data.shape) == 3:
        original_data = original_data[None, :]

    attributions = explainer.calculate_features(
        model=model,
        input_data=input_data,
        pred_label_idx=st.session_state[Settings.target_class_index],
        **kwargs,
    )

    figure = CVExplainer.visualize(
        attributions=attributions, transformed_img=original_data
    )
    return convert_figure_to_numpy(figure)


def get_kwargs(method: str) -> Dict[str, Any]:
    """Get `kwargs` from `st.session_state`.

    For `occulusion` method we have some additional arguments to pass.

    Args:
        method: Algorithm name.

    Returns:
        Dictionary of additional arguments for explainer.
    """
    kwargs = {}
    try:
        kwargs = {
            Settings.selected_layer_key: st.session_state[Settings.model_layers_key][
                st.session_state[Settings.selected_layer_key]
            ],
        }
    except IndexError:
        pass

    if method == "occulusion":
        kwargs[Settings.window_value] = st.session_state[Settings.window_value]
        kwargs[Settings.stride_value] = st.session_state[Settings.stride_value]

    return kwargs


def configure_params(method: str) -> None:
    """Display widgets to configure parameters of explanation.

    Args:
        method: Algorithm name.
    """

    idx_to_label = load_idx_to_labels(
        cache_path,
        st.session_state[Settings.date_selectbox_key],
        st.session_state[Settings.hash_selectbox_key],
    )
    label_to_idx = {val: idx for idx, val in idx_to_label.items()}

    st.selectbox(
        label="Select target class",
        options=list(idx_to_label.values()),
        key="target_class",
    )

    target_class = st.session_state.get("target_class", None)
    st.session_state[Settings.target_class_index] = label_to_idx[target_class]

    if method == "occulusion":
        with st.expander("Parameters"):
            st.slider(
                "Stride value",
                min_value=1,
                max_value=50,
                key=Settings.stride_value,
            )
            st.slider(
                "Sliding window value",
                min_value=1,
                max_value=50,
                key=Settings.window_value,
            )


def sidebar_view() -> None:
    """Display app configuration widgets in sidebar."""
    date = st.sidebar.selectbox(
        "Select experiment date:",
        list(st.session_state[Settings.experiment_label]),
        key=Settings.date_selectbox_key,
        on_change=disable_explain,
    )

    experiment_hash: str = st.sidebar.selectbox(
        "Select experiment hash:",
        list(st.session_state[Settings.experiment_label][date]),
        key=Settings.hash_selectbox_key,
        on_change=disable_explain,
    )
    st.sidebar.selectbox(
        "Select sample:",
        options=st.session_state[Settings.experiment_label][date][experiment_hash][
            "data"
        ],
        key=Settings.sample_name_key,
        on_change=disable_explain,
    )

    st.sidebar.selectbox(
        "Select epoch:",
        options=st.session_state[Settings.experiment_label][date][experiment_hash][
            "training"
        ],
        key=Settings.epoch_number_key,
        on_change=disable_explain,
    )

    st.sidebar.selectbox(
        "Select explainable method:",
        options=method_list,
        key=Settings.method_label,
    )


def main_view() -> None:
    """Display main app view."""
    hash_selectbox = st.session_state[Settings.hash_selectbox_key]
    date_selectbox = st.session_state[Settings.date_selectbox_key]
    epoch_number = st.session_state[Settings.epoch_number_key]

    model_path = os.path.join(
        cache_path,
        date_selectbox,
        hash_selectbox,
        "training",
        epoch_number,
        "model.onnx",
    )

    model: fx.GraphModule = load_model(model_path=model_path)
    model_layers = get_model_layers(model=model)
    st.session_state[Settings.model_layers_key] = model_layers
    method = st.session_state[Settings.method_label]

    st.markdown("## Configure parameters")
    configure_params(method=method)
    button = st.button(
        "Generate explanation",
    )
    if button:
        kwargs = get_kwargs(method=method)
        explainer = explainer_map[method]
        if check_if_layer_explainer(explainer=explainer):
            if not model_layers:
                st.error(
                    "No layers available to explain. `Conv2d` layers from "
                    + "CNN are mostly used for this method. Make sure You are "
                    + "using valid neural network.",
                )
            else:
                for layer in model_layers:
                    kwargs[Settings.selected_layer_key] = layer
                    st.write(f"Selected layer: {layer}")
                    image = generate_feature_map(
                        explainer=explainer_map[method],
                        selected_date=date_selectbox,
                        experiment_hash=hash_selectbox,
                        model=model,
                        **kwargs if kwargs else kwargs,
                    )
                    st.image(image)
        else:
            image = generate_feature_map(
                explainer=explainer_map[method],
                selected_date=date_selectbox,
                experiment_hash=hash_selectbox,
                model=model,
                **kwargs if kwargs else kwargs,
            )
            st.image(image)


def main() -> None:
    """Entry point for application."""
    st.title("AutoXAI PoC")
    initialize()

    sidebar_view()
    main_view()


if __name__ == "__main__":
    main()
