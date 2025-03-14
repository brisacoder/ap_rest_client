import logging
import logging.config
import os
from pathlib import Path

import logging
from pythonjsonlogger.json import JsonFormatter


def get_log_dir() -> Path:
    """
    Returns the log directory path relative to the main execution directory and ensures it exists.

    Returns:
        Path: The path to the logs directory (logs/)
    """
    log_dir = Path.cwd() / "logs"
    log_dir.mkdir(parents=True, exist_ok=True)
    return log_dir


def get_log_file() -> Path:
    """Returns the log file path and ensures it is removed on every startup."""
    log_file = get_log_dir() / "ap_rest_client.log"

    # Remove old log file on startup to ensure fresh logs
    try:
        if log_file.exists():
            log_file.unlink()
    except PermissionError as e:
        logging.error(f"{e}: {log_file}")
        raise

    return log_file


def get_log_level() -> str:
    """Retrieves the log level from environment variables (defaults to INFO)."""
    return os.getenv("LOG_LEVEL", "INFO").upper()


def configure_logging() -> logging.Logger:
    """Configures structured JSON logging for the client."""
    log_file = get_log_file()  # Get log file path
    log_level = get_log_level()  # Get log level (e.g., DEBUG, INFO, ERROR)

    logger = logging.getLogger()
    logger.setLevel(log_level)  # Ensure log level is set globally

    # ✅ Fix: Prevent double escaping in JSON output
    formatter = JsonFormatter(
        "{asctime} {levelname} {pathname} {module} {funcName} {message} {exc_info}",
        style="{",

    )

    # ✅ Log to Console (StreamHandler)
    stream_handler = logging.StreamHandler()
    stream_handler.setFormatter(formatter)
    logger.addHandler(stream_handler)

    # ✅ Log to File (FileHandler)
    file_handler = logging.FileHandler(log_file, mode="a", encoding="utf-8")
    file_handler.setFormatter(formatter)
    logger.addHandler(file_handler)

    logger.info("Client logging is initialized.", extra={"log_destination": log_file})

    return logger
