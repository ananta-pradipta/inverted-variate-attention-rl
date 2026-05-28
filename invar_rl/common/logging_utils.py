"""Plain-text logger setup.

Log messages are plain text only: no decorative icons, no em-dashes or
en-dashes. Use commas, parentheses, or colons for punctuation.
"""

from __future__ import annotations

import logging
import sys

_FORMAT = "%(asctime)s %(levelname)s %(name)s: %(message)s"
_DATEFMT = "%Y-%m-%d %H:%M:%S"


def get_logger(name: str, level: str = "INFO") -> logging.Logger:
    """Return a configured plain-text logger.

    Args:
        name: Logger name, typically the module name.
        level: Logging level as a string, for example "INFO" or "DEBUG".

    Returns:
        A ``logging.Logger`` writing plain-text records to stdout.
    """
    logger = logging.getLogger(name)
    numeric_level = getattr(logging, level.upper(), logging.INFO)
    logger.setLevel(numeric_level)

    if not logger.handlers:
        handler = logging.StreamHandler(stream=sys.stdout)
        handler.setFormatter(logging.Formatter(fmt=_FORMAT, datefmt=_DATEFMT))
        logger.addHandler(handler)
        logger.propagate = False

    return logger
