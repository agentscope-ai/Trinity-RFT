"""A Ray compatible logging module with actor-scope logger support."""
import contextvars
import logging
import os
import sys
from logging.handlers import RotatingFileHandler
from typing import Optional

import ray

from trinity.common.constants import (
    LOG_DIR_ENV_VAR,
    LOG_LEVEL_ENV_VAR,
    LOG_NODE_IP_ENV_VAR,
)

_LOG_FORMAT = "%(levelname)s %(asctime)s [%(filename)s:%(lineno)d] %(message)s"
_LOG_DATE_FORMAT = "%m-%d %H:%M:%S"


class NewLineFormatter(logging.Formatter):
    """
    Formatter that adds logging prefix to newlines to align multi-line messages.
    """

    def __init__(self, fmt: str, datefmt: Optional[str] = None):
        super().__init__(fmt, datefmt)

    def format(self, record: logging.LogRecord) -> str:
        msg = super().format(record)
        if record.message:
            prefix = msg.split(record.message)[0]
            msg = msg.replace("\n", f"\r\n{prefix}")
        return msg


# Process global logger for Ray actors, since Ray may create multiple processes and we want to reuse the logger instance within the same process.
_ray_logger: Optional[logging.Logger] = None
# Contextvars to support async tasks within the same Ray actor to reuse the same logger instance.
_ray_logger_ctx: contextvars.ContextVar[Optional[logging.Logger]] = contextvars.ContextVar(
    "ray_logger", default=None
)


def get_logger(
    name: Optional[str] = None, level: Optional[int] = None, in_ray_actor: bool = False
) -> logging.Logger:
    """
    Get a logger instance, compatible with Ray Actor and standard usage.

    In most cases, DO NOT USE this function in file-level scope, because the logger
    instance will be created at import time, which will cause issues when used in Ray actors.

    Args:
        name (Optional[str]): The name of the logger. If None, uses 'trinity'.
        level (Optional[int]): The logging level. If None, uses LOG_LEVEL_ENV_VAR or INFO.
        in_ray_actor (bool): Whether the logger is used within a Ray actor.

    Returns:
        logging.Logger: Configured logger instance.
    """
    # Reuse logger created by the actor if exists (Ray context)
    logger = _ray_logger_ctx.get()
    global _ray_logger

    if logger is not None:
        return logger
    if _ray_logger is not None:
        return _ray_logger

    resolved_level = (
        level
        if level is not None
        else getattr(logging, os.environ.get(LOG_LEVEL_ENV_VAR, "INFO").upper())
    )
    logger_name = f"trinity.{name}" if name else "trinity"
    logger = logging.getLogger(logger_name)
    logger.propagate = False
    logger.setLevel(resolved_level)
    logger.handlers.clear()

    # Stream handler (stdout)
    stream_handler = logging.StreamHandler(sys.stdout)
    stream_handler.setLevel(resolved_level)
    formatter = NewLineFormatter(_LOG_FORMAT, datefmt=_LOG_DATE_FORMAT)
    stream_handler.setFormatter(formatter)
    logger.addHandler(stream_handler)

    if in_ray_actor:
        # File handler (rotating file log)
        log_dir = os.environ.get(LOG_DIR_ENV_VAR)
        assert name is not None, "Logger name must be set when logging from a Ray actor"
        if log_dir:
            if os.environ.get(LOG_NODE_IP_ENV_VAR, "0") != "0":
                # organize logs by node IP
                node_ip = ray.util.get_node_ip_address()
                log_dir = os.path.join(log_dir, node_ip)
            os.makedirs(log_dir, exist_ok=True)
            # save log into log_dir/{actor_name}.log
            file_path = os.path.join(log_dir, f"{name}.log")
            file_handler = RotatingFileHandler(
                file_path, encoding="utf-8", maxBytes=64 * 1024 * 1024
            )
            file_handler.setLevel(resolved_level)
            file_handler.setFormatter(formatter)
            logger.addHandler(file_handler)
            _ray_logger_ctx.set(logger)
            _ray_logger = logger
        # If LOG_DIR_ENV_VAR is not set, file logging is disabled

    return logger
