"""
Refactored configuration system for the Conclave simulation.

This package provides a modular and Pydantic-validated configuration system,
replacing the older monolithic config.yaml.

Key components:
- `models.py`: Pydantic models defining the structure of configuration files.
- `loader.py`: Loads configuration from individual YAML files (agent.yaml, simulation.yaml, etc.)
             and merges them into a single `RefactoredConfig` object.
- `adapter.py`: Provides a `ConfigAdapter` class that offers a backward-compatible
              interface similar to the old `ConfigManager`, facilitating a smoother
              transition for existing code.
- `migrate.py`: (Optional) Script to help migrate from the old config.yaml to the new structure.

Configuration files are expected to be in the same directory:
- `agent.yaml`: LLM and embedding model settings.
- `simulation.yaml`: Core simulation parameters.
- `output.yaml`: Logging, results, and visualization settings.
- `predefined_groups.yaml`: Configuration for predefined simulation groups.
- `simulation_settings.yaml`: Configuration for simulation parameters.
"""

from .loader import RefactoredConfigLoader, load_config, get_config_loader
from .adapter import ConfigAdapter, get_config
from .models import RefactoredConfig

__all__ = [
    "RefactoredConfigLoader", 
    "load_config", 
    "get_config_loader",
    "ConfigAdapter",
    "get_config",
    "RefactoredConfig"
]
