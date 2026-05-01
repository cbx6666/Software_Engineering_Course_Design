"""平整度检测流程的日志工具。"""

import logging
from typing import Optional


LOGGER_NAME = "flatness_detection"


def configure_logging(level: str = "INFO") -> logging.Logger:
    """初始化主日志记录器。"""
    logger = logging.getLogger(LOGGER_NAME)
    numeric_level = getattr(logging, str(level).upper(), logging.INFO)
    logger.setLevel(numeric_level)

    if not logger.handlers:
        handler = logging.StreamHandler()
        handler.setFormatter(logging.Formatter("[%(levelname)s] %(message)s"))
        logger.addHandler(handler)

    logger.propagate = False
    return logger


def get_logger(name: Optional[str] = None) -> logging.Logger:
    """获取指定子模块的日志记录器。"""
    base_name = LOGGER_NAME if not name else f"{LOGGER_NAME}.{name}"
    logger = logging.getLogger(base_name)
    if not logging.getLogger(LOGGER_NAME).handlers:
        configure_logging()
    return logger
