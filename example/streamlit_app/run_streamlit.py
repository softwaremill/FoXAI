import os
from typing import Any, Dict

import numpy as np
import streamlit as st
from settings import Settings

from src.explainer import (
    CVExplainer,
    GradientSHAPCVExplainer,
    GuidedGradCamCVExplainer,
    IntegratedGradientsCVExplainer,
    LRPCVExplainer,
    NoiseTunnelCVExplainer,
    OcculusionCVExplainer,
)
from streamlit_utils import (
    convert_figure_to_numpy,
    disable_explain,
    initialize_session_state,
    load_input_data,
    load_idx_to_labels,
    load_subdir,
    selected_date,
    load_original_data,
)

import onnx
import onnx2torch

cache_path = os.environ.get("LOGDIR", "logs")

explainer_map = {
    "occulusion": OcculusionCVExplainer(),
    "integrated_gradients": IntegratedGradientsCVExplainer(),
    "noisy_tunnel": NoiseTunnelCVExplainer(),
    "gradient_shap": GradientSHAPCVExplainer(),
    "lrp": LRPCVExplainer(),
    "guided_gradcam": GuidedGradCamCVExplainer(),
}
method_list = [val for val in explainer_map.keys()]


@st.cache
def load_data():
    return load_subdir(cache_path)


def initialize() -> None:
    st.session_state[Settings.experiment_label] = load_data()
    initial_date = [val for val in st.session_state[Settings.experiment_label].keys()][
        -1
    ]
    initialize_session_state(Settings.experiment_date_label, initial_date)
    initial_hash: str = [
        val for val in st.session_state[Settings.experiment_label][initial_date].keys()
    ][0]
    initialize_session_state(
        Settings.hash_label,
        initial_hash,
    )
    initialize_session_state("explain", False)
    initialize_session_state("selected_date", False)
    initialize_session_state(Settings.window_value, 3)
    initialize_session_state(Settings.stride_value, 3)


def generate_feature_map(
    explainer: CVExplainer,
    date_selectbox: str,
    hash_selectbox: str,
    model: Any,
    **kwargs,
) -> np.ndarray:
    input_data = load_input_data(
        cache_path,
        date_selectbox,
        hash_selectbox,
    )
    original_data = load_original_data(
        cache_path,
        date_selectbox,
        hash_selectbox,
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

    figure = explainer.visualize(
        attributions=attributions, transformed_img=original_data
    )
    return convert_figure_to_numpy(figure)


def get_kwargs(method: str) -> Dict[str, Any]:
    if method == "occulusion":
        return {
            Settings.window_value: st.session_state[Settings.window_value],
            Settings.stride_value: st.session_state[Settings.stride_value],
        }
    else:
        return {}


def configure_params(method: str) -> None:

    idx_to_label = load_idx_to_labels(
        cache_path,
        st.session_state[Settings.date_selectbox_key],
        st.session_state[Settings.hash_selectbox_key],
    )
    label_to_idx = {val: idx for idx, val in idx_to_label.items()}

    st.selectbox(
        label="Select target class",
        options=[val for val in idx_to_label.values()],
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
    date = st.sidebar.selectbox(
        "Select experiment date:",
        [val for val in st.session_state[Settings.experiment_label].keys()],
        key=Settings.date_selectbox_key,
        on_change=selected_date,
    )

    hash: str = st.sidebar.selectbox(
        "Select experiment hash:",
        [val for val in st.session_state[Settings.experiment_label][date].keys()],
        key=Settings.hash_selectbox_key,
        on_change=disable_explain,
    )
    st.sidebar.selectbox(
        "Select sample:",
        options=st.session_state[Settings.experiment_label][date][hash]["data"],
        key=Settings.sample_name_key,
        on_change=disable_explain,
    )

    st.sidebar.selectbox(
        "Select epoch:",
        options=st.session_state[Settings.experiment_label][date][hash]["training"],
        key=Settings.epoch_number_key,
        on_change=disable_explain,
    )

    st.sidebar.selectbox(
        "Select explainable method:",
        options=method_list,
        key=Settings.method_label,
    )


def main_view() -> None:
    hash_selectbox = st.session_state[Settings.hash_selectbox_key]
    date_selectbox = st.session_state[Settings.date_selectbox_key]
    epoch_number = st.session_state[Settings.epoch_number_key]

    model_path = os.path.join(
        cache_path, date_selectbox, hash_selectbox, "training", epoch_number, "model.onnx"
    )

    model = onnx2torch.convert(onnx.load(model_path))
    method = st.session_state[Settings.method_label]
    st.markdown("## Configure parameters")
    configure_params(method=method)
    button = st.button(
        "Generate explanation",
    )
    if button:
        kwargs = get_kwargs(method=method)
        image = generate_feature_map(
            explainer=explainer_map[method],
            date_selectbox=date_selectbox,
            hash_selectbox=hash_selectbox,
            model=model,
            **kwargs if kwargs else kwargs,
        )
        st.image(image)

def main() -> None:
    st.title("AutoXAI PoC")
    initialize()

    sidebar_view()
    main_view()


if __name__ == "__main__":
    main()