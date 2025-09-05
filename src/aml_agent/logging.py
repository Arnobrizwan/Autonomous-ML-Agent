"""
Logging configuration for the Autonomous ML Agent.
"""

import logging
import sys
from pathlib import Path
from typing import Optional

from rich.console import Console
from rich.logging import RichHandler
from rich.traceback import install

# Install rich traceback handler
install(show_locals=True)

# Global console instance
console = Console()


def setup_logging(
    level: str = "INFO", log_file: Optional[str] = None, rich_console: bool = True
) -> logging.Logger:
    """
    Setup logging configuration.

    Args:
        level: Logging level (DEBUG, INFO, WARNING, ERROR, CRITICAL)
        log_file: Optional log file path
        rich_console: Whether to use rich console handler

    Returns:
        Configured logger
    """
    # Create logger
    logger = logging.getLogger("aml_agent")
    logger.setLevel(getattr(logging, level.upper()))

    # Clear existing handlers
    logger.handlers.clear()

    # Create formatters
    if rich_console:
        # Use rich handler for console
        console_handler = RichHandler(
            console=console,
            show_time=True,
            show_path=True,
            markup=True,
            rich_tracebacks=True,
        )
        console_handler.setLevel(getattr(logging, level.upper()))
        logger.addHandler(console_handler)
    else:
        # Use standard console handler
        console_handler = logging.StreamHandler(sys.stdout)
        console_handler.setLevel(getattr(logging, level.upper()))
        formatter = logging.Formatter(
            "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
        )
        console_handler.setFormatter(formatter)
        logger.addHandler(console_handler)

    # Add file handler if specified
    if log_file:
        log_path = Path(log_file)
        log_path.parent.mkdir(parents=True, exist_ok=True)

        file_handler = logging.FileHandler(log_path)
        file_handler.setLevel(logging.DEBUG)
        file_formatter = logging.Formatter(
            "%(asctime)s - %(name)s - %(levelname)s - %(funcName)s:%(lineno)d - %(message)s"
        )
        file_handler.setFormatter(file_formatter)
        logger.addHandler(file_handler)

    return logger


def get_logger(name: str = "aml_agent") -> logging.Logger:
    """Get logger instance."""
    return logging.getLogger(name)


def log_function_call(func):
    """Decorator to log function calls."""

    def wrapper(*args, **kwargs):
        logger = get_logger()
        logger.debug(f"Calling {func.__name__} with args={args}, kwargs={kwargs}")
        try:
            result = func(*args, **kwargs)
            logger.debug(f"{func.__name__} completed successfully")
            return result
        except Exception as e:
            logger.error(f"{func.__name__} failed with error: {str(e)}")
            raise

    return wrapper


def log_trial_result(
    trial_id: int, model_type: str, score: float, status: str = "completed"
):
    """Log trial result."""
    logger = get_logger()
    if status == "completed":
        logger.info(f"Trial {trial_id} ({model_type}): score={score:.4f}")
    else:
        logger.warning(f"Trial {trial_id} ({model_type}): {status}")


def log_budget_status(elapsed: float, remaining: float, total: float):
    """Log budget status."""
    logger = get_logger()
    progress = (elapsed / total) * 100
    logger.info(
        f"Budget status: {elapsed:.1f}s elapsed, {remaining:.1f}s remaining ({progress:.1f}%)"
    )


def log_model_performance(model_type: str, score: float, rank: int, total_models: int):
    """Log model performance."""
    logger = get_logger()
    logger.info(f"Model {model_type}: score={score:.4f}, rank={rank}/{total_models}")


def log_ensemble_creation(ensemble_method: str, top_k: int, models: list):
    """Log ensemble creation."""
    logger = get_logger()
    model_names = [m.model_type for m in models]
    logger.info(
        f"Creating {ensemble_method} ensemble with top {top_k} models: {model_names}"
    )


def log_export_artifacts(artifacts_dir: str, files: list):
    """Log exported artifacts."""
    logger = get_logger()
    logger.info(f"Exported artifacts to {artifacts_dir}:")
    for file in files:
        logger.info(f"  - {file}")


def log_api_startup(host: str, port: int):
    """Log API startup."""
    logger = get_logger()
    logger.info(f"Starting API server on {host}:{port}")


def log_prediction_request(run_id: str, n_features: int, single: bool = True):
    """Log prediction request."""
    logger = get_logger()
    request_type = "single" if single else "batch"
    logger.info(
        f"Prediction request ({request_type}): run_id={run_id}, features={n_features}"
    )


def log_error(error: Exception, context: str = ""):
    """Log error with context."""
    logger = get_logger()
    if context:
        logger.error(f"Error in {context}: {str(error)}")
    else:
        logger.error(f"Error: {str(error)}")
    logger.debug("Full traceback:", exc_info=True)
