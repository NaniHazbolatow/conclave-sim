"""
Standardized logging utilities for the Conclave simulation.

This module provides a centralized logging setup to ensure consistent
logging patterns across all components of the simulation.
"""

import logging
import sys
from typing import Optional
from pathlib import Path
import json


# Logger cache to avoid recreating loggers
_logger_cache = {}


class JsonFormatter(logging.Formatter):
    """
    Formats log records as a JSON string.
    """
    def format(self, record):
        log_object = {
            "timestamp": self.formatTime(record, self.datefmt),
            "level": record.levelname,
            "name": record.name,
        }
        # The actual message is expected to be a dict
        if isinstance(record.msg, dict):
            log_object.update(record.msg)
        else:
            log_object["message"] = record.getMessage()
            
        return json.dumps(log_object)


def get_logger(name: str) -> logging.Logger:
    """
    Get a standardized logger for the given component.
    
    This function ensures consistent logger naming and configuration
    across all Conclave modules.
    
    Args:
        name: The logger name, typically the module name or component name
        
    Returns:
        A configured logger instance
        
    Examples:
        >>> logger = get_logger("conclave.agents")
        >>> logger = get_logger(__name__)
        >>> logger = get_logger("conclave.agents.voting")
    """
    if name in _logger_cache:
        return _logger_cache[name]
    
    # Ensure all conclave loggers follow the naming convention
    if not name.startswith("conclave.") and name != "conclave":
        if name == "__main__":
            name = "conclave.main"
        elif not name.startswith("conclave"):
            name = f"conclave.{name}"
    
    logger = logging.getLogger(name)
    _logger_cache[name] = logger
    
    return logger


def setup_logging(
    level: str = "INFO",
    log_file: Optional[Path] = None,
    format_string: Optional[str] = None,
    include_timestamp: bool = True
) -> None:
    """
    Set up standardized logging configuration for the Conclave simulation.
    
    This function configures the root logger with consistent formatting
    and output destinations.
    
    Args:
        level: Logging level (DEBUG, INFO, WARNING, ERROR, CRITICAL)
        log_file: Optional file path to write logs to
        format_string: Custom format string (uses default if None)
        include_timestamp: Whether to include timestamps in log messages
        
    Examples:
        >>> setup_logging("DEBUG")
        >>> setup_logging("INFO", log_file=Path("simulation.log"))
        >>> setup_logging("WARNING", include_timestamp=False)
    """
    # Convert string level to logging constant
    numeric_level = getattr(logging, level.upper(), logging.INFO)
    
    # Default format string
    if format_string is None:
        if include_timestamp:
            format_string = "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
        else:
            format_string = "%(name)s - %(levelname)s - %(message)s"
    
    # Create formatter
    formatter = logging.Formatter(format_string)
    
    # Get root logger
    root_logger = logging.getLogger()
    root_logger.setLevel(numeric_level)
    
    # Clear existing handlers
    for handler in root_logger.handlers[:]:
        root_logger.removeHandler(handler)
    
    # Console handler
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setLevel(numeric_level)
    console_handler.setFormatter(formatter)
    root_logger.addHandler(console_handler)

    if log_file:
        file_handler = logging.FileHandler(log_file)
        file_handler.setLevel(numeric_level)
        file_handler.setFormatter(formatter)
        root_logger.addHandler(file_handler)


def setup_stance_logger(log_dir: Path) -> logging.Logger:
    """
    Sets up a dedicated logger for recording agent stances in JSONL format.

    Args:
        log_dir: The directory where the log file will be created.

    Returns:
        A configured logger instance for stances.
    """
    stance_logger_name = "conclave.stances"
    if stance_logger_name in _logger_cache:
        return _logger_cache[stance_logger_name]

    logger = logging.getLogger(stance_logger_name)
    logger.setLevel(logging.INFO)
    logger.propagate = False  # Prevent logs from propagating to the root logger

    # Stance log file
    stance_log_file = log_dir / "stances.log"

    # Create file handler
    file_handler = logging.FileHandler(stance_log_file)
    file_handler.setLevel(logging.INFO)

    # Create JSON formatter
    formatter = JsonFormatter()
    file_handler.setFormatter(formatter)

    # Clear existing handlers and add the new one
    if logger.hasHandlers():
        logger.handlers.clear()
    logger.addHandler(file_handler)

    _logger_cache[stance_logger_name] = logger
    return logger


def configure_component_loggers() -> None:
    """
    Configure specific logging levels for different Conclave components.
    
    This allows fine-grained control over logging verbosity for different
    parts of the simulation.
    """
    # Set component-specific levels
    component_levels = {
        'conclave.agents': logging.INFO,
        'conclave.environments': logging.INFO,
        'conclave.llm': logging.WARNING,
        'conclave.discussions': logging.INFO,
        'conclave.prompting': logging.WARNING,
        'conclave.config': logging.WARNING,
        'conclave.visualization': logging.INFO,
        'conclave.system': logging.INFO,
    }
    
    for component, level in component_levels.items():
        logger = logging.getLogger(component)
        logger.setLevel(level)


def log_error_with_context(logger: logging.Logger, error: Exception, context: str = "") -> None:
    """
    Log an error with additional context information.
    
    Args:
        logger: The logger to use
        error: The exception that occurred
        context: Additional context about where/when the error occurred
    """
    error_msg = f"Error: {str(error)}"
    if context:
        error_msg = f"{context} - {error_msg}"
    
    logger.error(error_msg, exc_info=True)


def log_performance_timing(logger: logging.Logger, operation: str, duration: float) -> None:
    """
    Log performance timing information.
    
    Args:
        logger: The logger to use
        operation: Description of the operation
        duration: Duration in seconds
    """
    logger.info(f"Performance: {operation} took {duration:.3f}s")
