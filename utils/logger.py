import logging

from omegaconf import DictConfig

logger = logging.getLogger(__name__)


def setup_logger(cfg: DictConfig, root_logger: bool = False) -> logging.Logger:
    """Sets up the logger.

    Args:
        cfg: the configuration for the logger
        root_logger: flag indicating whether to use the root logger. Defaults to False

    Returns:
        logging.Logger: the configured logger

    """
    if root_logger:
        custom_logger = logging.getLogger(__name__)
    else:
        custom_logger = logging.getLogger(cfg.name)

    formatter = logging.Formatter(fmt=cfg.format, datefmt=cfg.date_format)

    if "out_file" in cfg:
        handler = logging.FileHandler(cfg.out_file)
    else:
        handler = logging.StreamHandler()

    handler.setLevel(cfg.level)
    handler.setFormatter(formatter)
    custom_logger.addHandler(handler)

    if root_logger:
        str_handler = logging.StreamHandler()
        str_handler.setLevel(cfg.level)
        str_handler.setFormatter(formatter)
        custom_logger.addHandler(str_handler)

    custom_logger.propagate = False
    custom_logger.setLevel(cfg.level)

    return custom_logger


def initialize_logger(cfg: DictConfig) -> None:
    """Initializes the logger.

    Args:
        cfg: the configuration dictionary

    Returns:
        None
    """
    global logger

    # Setup logger
    logger = setup_logger(cfg.logger, True)
