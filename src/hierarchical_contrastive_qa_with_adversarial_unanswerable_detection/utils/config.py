"""Configuration management utilities."""

import logging
import random
from pathlib import Path
from typing import Any, Dict, Optional

import numpy as np
import torch
import yaml

logger = logging.getLogger(__name__)


def load_config(config_path: str) -> Dict[str, Any]:
    """
    Load configuration from YAML file.

    Args:
        config_path: Path to YAML configuration file

    Returns:
        Dictionary containing configuration parameters

    Raises:
        FileNotFoundError: If config file does not exist
        yaml.YAMLError: If config file is invalid YAML
    """
    config_file = Path(config_path)
    if not config_file.exists():
        raise FileNotFoundError(f"Configuration file not found: {config_path}")

    try:
        with open(config_file, "r") as f:
            config = yaml.safe_load(f)
        logger.info(f"Loaded configuration from {config_path}")
        return config
    except yaml.YAMLError as e:
        logger.error(f"Error parsing YAML configuration: {e}")
        raise


def save_config(config: Dict[str, Any], save_path: str) -> None:
    """
    Save configuration to YAML file.

    Args:
        config: Configuration dictionary
        save_path: Path where to save the configuration
    """
    save_file = Path(save_path)
    save_file.parent.mkdir(parents=True, exist_ok=True)

    with open(save_file, "w") as f:
        yaml.dump(config, f, default_flow_style=False, sort_keys=False)
    logger.info(f"Saved configuration to {save_path}")


def set_seed(seed: int) -> None:
    """
    Set random seeds for reproducibility.

    Args:
        seed: Random seed value
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

    # For deterministic behavior (may impact performance)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

    logger.info(f"Set random seed to {seed}")


def get_device() -> torch.device:
    """
    Get the appropriate device (CUDA if available, else CPU).

    Returns:
        torch.device object
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info(f"Using device: {device}")
    return device


def setup_logging(
    log_dir: Optional[str] = None, log_level: int = logging.INFO
) -> logging.Logger:
    """
    Setup logging configuration.

    Args:
        log_dir: Directory to save log files (optional)
        log_level: Logging level

    Returns:
        Configured logger
    """
    # Create formatter
    formatter = logging.Formatter(
        "%(asctime)s - %(name)s - %(levelname)s - %(message)s", datefmt="%Y-%m-%d %H:%M:%S"
    )

    # Setup root logger
    root_logger = logging.getLogger()
    root_logger.setLevel(log_level)

    # Remove existing handlers
    root_logger.handlers = []

    # Console handler
    console_handler = logging.StreamHandler()
    console_handler.setLevel(log_level)
    console_handler.setFormatter(formatter)
    root_logger.addHandler(console_handler)

    # File handler (if log_dir specified)
    if log_dir is not None:
        log_path = Path(log_dir)
        log_path.mkdir(parents=True, exist_ok=True)
        file_handler = logging.FileHandler(log_path / "training.log")
        file_handler.setLevel(log_level)
        file_handler.setFormatter(formatter)
        root_logger.addHandler(file_handler)

    return root_logger
