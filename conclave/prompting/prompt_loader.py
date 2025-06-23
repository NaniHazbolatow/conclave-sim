\
# filepath: /Users/kbverlaan/Library/Mobile Documents/com~apple~CloudDocs/Universiteit/Computational Science/ABM/conclave-sim/conclave/config/prompt_loader.py
"""
Prompt loader for the Conclave Simulation.
Handles loading and managing prompt configurations from YAML files using Pydantic.
"""

import yaml
import logging
from typing import Dict, Any, Optional, List
from pathlib import Path
from pydantic import ValidationError # Added import for ValidationError

from .prompt_models import PromptsConfig, ToolDefinition # Changed from .config_models

logger = logging.getLogger(__name__)

class PromptLoader: # Renamed from PromptManager
    """Manages loading, validation, and access for prompts defined in prompts.yaml."""

    prompts_config: PromptsConfig

    def __init__(self, prompts_path: Optional[str] = None):
        """
        Initialize the PromptLoader.

        Args:
            prompts_path: Path to the prompts YAML file. If None, looks for 'prompts.yaml'
                          in a 'data/' subdirectory relative to common execution paths
                          or the project root.
        """
        if prompts_path is None:
            potential_paths = [
                Path("conclave/prompting/prompts.yaml"), # Direct path from project root
                Path("prompts.yaml"), # If CWD is conclave/prompting
                Path(__file__).parent / "prompts.yaml", # Relative to this file
                Path(__file__).parent.parent / "prompting/prompts.yaml", # Relative from conclave/
                Path(__file__).parent.parent.parent / "conclave/prompting/prompts.yaml" # Absolute from project root (if scripts/ is elsewhere)
            ]
            resolved_prompts_path = None
            for p_path in potential_paths:
                try:
                    # Check if the path exists and is a file
                    if p_path.is_file():
                        resolved_prompts_path = p_path.resolve()
                        logger.debug(f"Found prompts.yaml at: {resolved_prompts_path}")
                        break
                    else:
                        logger.debug(f"Checked path, prompts.yaml not found or not a file: {p_path.resolve()}")
                except Exception as e:
                    logger.debug(f"Error checking path {p_path}: {e}")
            
            if resolved_prompts_path is None:
                # Fallback or error if not found - this indicates a setup issue
                default_path = Path(__file__).parent / "prompts.yaml"
                logger.error(f"Could not find prompts.yaml in any of the expected locations. Defaulting to {default_path.resolve()}, but this may fail.")
                self.prompts_path = default_path # Keep a default to avoid crashing init
            else:
                self.prompts_path = resolved_prompts_path
        else:
            self.prompts_path = Path(prompts_path)

        self._load_prompts()

    def _load_prompts(self) -> None:
        """Load prompts from YAML file and validate with Pydantic models."""
        try:
            if not self.prompts_path.exists():
                raise FileNotFoundError(f"Prompts file not found: {self.prompts_path.resolve()}")

            with open(self.prompts_path, 'r', encoding='utf-8') as f:
                raw_prompts = yaml.safe_load(f)

            self.prompts_config = PromptsConfig(**raw_prompts)
            logger.info(f"Prompts loaded and validated from {self.prompts_path.resolve()}")
            # Log the keys that were successfully loaded
            loaded_keys = list(self.prompts_config.model_dump().keys())
            logger.debug(f"Loaded prompt keys: {loaded_keys}")

        except FileNotFoundError:
            logger.error(f"Prompts file not found: {self.prompts_path.resolve()}")
            raise
        except yaml.YAMLError as e:
            logger.error(f"Error parsing YAML for prompts from {self.prompts_path.resolve()}: {e}")
            raise
        except ValidationError as e:
            logger.error(f"Prompts validation error for {self.prompts_path.resolve()}:\\n{e}")
            raise
        except Exception as e: 
            logger.error(f"Unexpected error loading prompts from {self.prompts_path.resolve()}: {e}")
            raise

    def get_prompt(self, prompt_name: str) -> Optional[str]:
        """
        Get a specific prompt template by its name.

        Args:
            prompt_name: The name of the prompt (e.g., 'internal_persona_extractor').

        Returns:
            The prompt template string, or None if not found.
        """
        prompt_template = getattr(self.prompts_config, prompt_name, None)

        # Pydantic v2 compatibility: extra fields are in `model_extra`
        if prompt_template is None and self.prompts_config.model_extra:
            prompt_template = self.prompts_config.model_extra.get(prompt_name)

        if prompt_template is None:
            logger.warning(f"Prompt '{prompt_name}' not found in prompts configuration.")
        
        return prompt_template

    def get_prompt_template(self, prompt_name: str) -> Optional[str]:
        """
        Get a specific prompt template by its name.

        Args:
            prompt_name: The name of the prompt (e.g., 'internal_persona_extractor').

        Returns:
            The prompt template string, or None if not found.
        """
        if hasattr(self.prompts_config, prompt_name):
            return getattr(self.prompts_config, prompt_name)
        logger.warning(f"Prompt '{prompt_name}' not found in configuration.")
        return None

    def get_tool_definition(self, tool_name: str) -> Optional[ToolDefinition]:
        """
        Retrieves a specific tool definition from the loaded prompts configuration.

        Args:
            tool_name: The name of the tool to retrieve (e.g., "generate_stance").

        Returns:
            A dictionary containing the tool definition, or None if not found.
        """
        try:
            tool_def_model = self.prompts_config.tool_definitions.get(tool_name)
            if tool_def_model:
                # Manually construct the dictionary to avoid potential recursion in model_dump()
                # This assumes ToolDefinition and ParametersDefinition have the expected attributes.
                
                parameters_dict = {
                    "type": tool_def_model.parameters.type,
                    "properties": {},
                    "required": tool_def_model.parameters.required # Added required field
                }
                if tool_def_model.parameters.properties:
                    for prop_name, prop_details in tool_def_model.parameters.properties.items():
                        # Assuming prop_details is a Pydantic model (PropertyDetail)
                        # and calling model_dump() on it if it's simple enough,
                        # or manually constructing if it also causes issues.
                        # For now, let's try model_dump() on the simpler PropertyDetail.
                        parameters_dict["properties"][prop_name] = prop_details.model_dump()

                tool_function_dict = {
                    "name": tool_def_model.name,
                    "description": tool_def_model.description,
                    "parameters": parameters_dict,
                }
                
                return {
                    "type": "function",
                    "function": tool_function_dict
                }
            else:
                logger.warning(f"Tool definition '{tool_name}' not found in prompts configuration.")
                return None
        except Exception as e:
            # Log the full traceback for recursion errors or other issues
            logger.error(f"Error retrieving tool definition '{tool_name}': {e}", exc_info=True)
            return None

    def get_all_prompt_names(self) -> List[str]:
        """
        Get a list of all prompt names available in the configuration.

        Returns:
            A list of strings, each representing a prompt name.
        """
        return [name for name in self.prompts_config.dict().keys()]

    def get_all_prompts(self) -> Dict[str, Any]:
        """Returns all loaded prompts as a dictionary."""
        return self.prompts_config

    def get_all_tool_definitions(self) -> Dict[str, ToolDefinition]:
        """Get all tool definitions."""
        return self.prompts_config.tool_definitions
    
    def get_prompts_path(self) -> Path:
        """Get the path to the loaded prompts file."""
        return self.prompts_path

    def reload_prompts(self, prompts_path: Optional[str] = None) -> None:
        """
        Reload prompts from the YAML file.
        Optionally updates the path to the prompts file.
        """
        if prompts_path:
            self.prompts_path = Path(prompts_path)
        self._load_prompts()
        logger.info(f"Prompts reloaded and validated from {self.prompts_path.resolve()}")

# Global prompt loader instance
_prompt_loader: Optional[PromptLoader] = None

def get_prompt_loader(prompts_path: Optional[str] = None) -> PromptLoader:
    """
    Get the global PromptLoader instance.

    Args:
        prompts_path: Path to the prompts YAML file. Only used on the first call
                      or if the loader hasn't been initialized yet.

    Returns:
        The PromptLoader instance.
    """
    global _prompt_loader
    if _prompt_loader is None:
        _prompt_loader = PromptLoader(prompts_path)
    return _prompt_loader

def reload_prompts(prompts_path: Optional[str] = None) -> None:
    """
    Reload the global prompts configuration.
    If prompts_path is provided, it updates the path for subsequent reloads.
    """
    global _prompt_loader
    if _prompt_loader is not None:
        _prompt_loader.reload_prompts(prompts_path)
    elif prompts_path: 
        get_prompt_loader(prompts_path)
    else: 
        get_prompt_loader()