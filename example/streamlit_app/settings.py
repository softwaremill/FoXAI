"""File contains class holding all keys used in st.session_state."""


class Settings:  # pylint: disable = (too-few-public-methods)
    """Class holding all keys in st.session_state."""

    experiment_label: str = "experiment_data"
    experiment_date_label: str = "experiment_date"
    hash_label: str = "experiment_hash"
    method_label: str = "experiment_method"
    stride_value: str = "stride_value"
    window_value: str = "window_value"
    target_class: str = "target_class"
    target_class_index: str = "target_class_index"
    hash_selectbox_key: str = "hash_selectbox"
    sample_name_key: str = "sample_name"
    epoch_number_key: str = "epoch_number"
    date_selectbox_key: str = "date_selectbox"
    selected_layer_key: str = "selected_layer"
    model_layers_key: str = "model_layers"
    explain_key: str = "explain"
