"""Configuration management for the Conclave simulation.

This package handles loading, validation, and access to configuration settings
defined in YAML files.

Modules:
- prompt_loader.py: Contains PromptLoader for prompts.yaml.
- prompt_variable_generator.py: Generates prompt variables.

Configuration is now primarily handled by the scripts in the top-level 'config/scripts' directory.
This package (`conclave/config`) is being phased out for general configuration
but still handles prompt loading.

Usage:
    from conclave.prompting import get_prompt_loader # For prompts
    from config.scripts import get_config # For main simulation config

    # Access main configuration (config/*.yaml)
    config = get_config()
    llm_backend = config.config.agent.llm.backend # Example access

    # Access prompt configurations (prompts.yaml)
    prompts = get_prompt_loader()
"""

# Export prompt-related tools
from .prompt_loader import PromptLoader, get_prompt_loader
from .unified_generator import UnifiedPromptVariableGenerator

__all__ = ["PromptLoader", "get_prompt_loader", "UnifiedPromptVariableGenerator"]
