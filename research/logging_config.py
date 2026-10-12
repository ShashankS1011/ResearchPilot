import logging

from .config import SETTINGS


logger = logging.getLogger(
    "researchpilot"
)


def configure_logging():
    logging.basicConfig(
        level=getattr(
            logging,
            SETTINGS.log_level,
            logging.INFO,
        ),
        format=(
            "%(asctime)s | "
            "%(levelname)s | "
            "%(name)s | "
            "%(message)s"
        ),
    )