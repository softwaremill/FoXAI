# pylint: disable = missing-module-docstring
import logging
from logging import StreamHandler

__LOG_FORMAT: str = "%(asctime)s %(levelname)s %(name)s - %(message)s"


def create_logger(logger_name: str, level: int = logging.DEBUG) -> logging.Logger:
    """Create logger with the goven name.

    Args:
        logger_name: the name of the logger

    Return:
        created logger object
    """

    result: logging.Logger = logging.getLogger(logger_name)
    result.setLevel(level)
    result.propagate = False

    formatter: logging.Formatter = logging.Formatter(__LOG_FORMAT)
    console_handler = StreamHandler()
    console_handler.setFormatter(formatter)
    result.addHandler(console_handler)

    return result
