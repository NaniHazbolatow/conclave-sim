"""
Configuration management for the Conclave simulation.

This module provides centralized configuration access through a singleton
ConfigManager, eliminating redundant configuration loading throughout the codebase.
"""

from .manager import (
    ConfigManager,
    get_config_manager,
    get_config,
    get_agent_config,
    get_simulation_config,
    get_output_config,
)

__all__ = [
    'ConfigManager',
    'get_config_manager',
    'get_config',
    'get_agent_config', 
    'get_simulation_config',
    'get_output_config',
]
