"""File contains definition of CLI program config data model."""

from dataclasses import dataclass
from typing import Dict, List, Optional


@dataclass
class MethodDataModel:
    """Method data model for CLI program."""

    name: str
    params: Dict[str, str]
    artifact_name: Optional[str] = None


@dataclass
class ConfigDataModel:
    """Config data model for CLI program."""

    method_config_list: List[MethodDataModel]
