"""
Standardized logging utilities for the Conclave simulation.

This module provides a centralized logging setup to ensure consistent
logging patterns across all components of the simulation.
"""

import logging
import sys
from typing import Optional
from pathlib import Path


# Logger cache to avoid recreating loggers
_logger_cache = {}


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
    
    # File handler (if specified)
    if log_file:
        log_file.parent.mkdir(parents=True, exist_ok=True)
        file_handler = logging.FileHandler(log_file)
        file_handler.setLevel(numeric_level)
        file_handler.setFormatter(formatter)
        root_logger.addHandler(file_handler)
    
    # Set specific logger levels for common external libraries
    logging.getLogger('httpx').setLevel(logging.WARNING)
    logging.getLogger('urllib3').setLevel(logging.WARNING)
    logging.getLogger('requests').setLevel(logging.WARNING)


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
