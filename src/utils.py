"""Utility functions for the PDF Parser RAG system."""

import hashlib
import logging
from pathlib import Path
from typing import Any

import yaml
from dotenv import load_dotenv
load_dotenv()

def load_config(config_path: str = "config/config.yaml") -> dict[str, Any]:
    """Load YAML configuration file.

    Args:
        config_path: Path to the YAML config file

    Returns:
        Dictionary containing configuration

    Raises:
        FileNotFoundError: If config file doesn't exist
        yaml.YAMLError: If config file is invalid
    """
    config_file = Path(config_path)
    if not config_file.exists():
        raise FileNotFoundError(f"Config file not found: {config_path}")

    with open(config_file, "r", encoding="utf-8") as f:
        config = yaml.safe_load(f)

    return config


def setup_logging(log_dir: str, name: str, level: int = logging.INFO) -> logging.Logger:
    """Configure logging with both console and file handlers.

    Args:
        log_dir: Directory to store log files
        name: Logger name (will be used for log filename)
        level: Logging level (default: INFO)

    Returns:
        Configured logger instance
    """
    # Create log directory if it doesn't exist
    log_path = Path(log_dir)
    log_path.mkdir(parents=True, exist_ok=True)

    # Create logger
    logger = logging.getLogger(name)
    logger.setLevel(level)

    # Avoid duplicate handlers if logger already exists
    if logger.handlers:
        return logger

    # Format
    formatter = logging.Formatter(
        "%(asctime)s - %(name)s - %(levelname)s - %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S"
    )

    # Console handler
    console_handler = logging.StreamHandler()
    console_handler.setLevel(level)
    console_handler.setFormatter(formatter)
    logger.addHandler(console_handler)

    # File handler
    log_file = log_path / f"{name}.log"
    file_handler = logging.FileHandler(log_file, encoding="utf-8")
    file_handler.setLevel(level)
    file_handler.setFormatter(formatter)
    logger.addHandler(file_handler)

    return logger


def compute_file_hash(file_path: str, hash_length: int = 16) -> str:
    """Compute SHA256 hash of a file.

    Args:
        file_path: Path to the file
        hash_length: Length of the returned hash string (default: 16)

    Returns:
        Truncated hex digest of the file hash

    Raises:
        FileNotFoundError: If file doesn't exist
    """
    file = Path(file_path)
    if not file.exists():
        raise FileNotFoundError(f"File not found: {file_path}")

    hasher = hashlib.sha256()

    # Read file in chunks to handle large files
    with open(file, "rb") as f:
        for chunk in iter(lambda: f.read(8192), b""):
            hasher.update(chunk)

    # Return truncated hash
    return hasher.hexdigest()[:hash_length]
