from pathlib import Path

from single_source import get_version

__version__: str = get_version(
    __name__, Path(__file__).parent, fail=True  # type: ignore
)
