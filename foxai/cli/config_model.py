"""File contains definition of CLI program config data model."""

from dataclasses import dataclass
from typing import List, Optional

from foxai.context_manager import ExplainerWithParams


@dataclass
class MethodDataModel:
    """Method data model for CLI program."""

    explainer_with_params: ExplainerWithParams
    artifact_name: Optional[str] = None


@dataclass
class ConfigDataModel:
    """Config data model for CLI program."""

    method_config_list: List[MethodDataModel]
