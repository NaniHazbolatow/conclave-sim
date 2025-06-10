"""
Prompt management for the Conclave Simulation.
Loads and formats prompt templates from external files.
"""

import yaml
import logging
from typing import Dict, Any, Optional
from pathlib import Path

logger = logging.getLogger(__name__)

class PromptManager:
    """Manages loading and formatting of prompt templates."""
    
    def __init__(self, prompts_path: Optional[str] = None):
        """
        Initialize the prompt manager.
        
        Args:
            prompts_path: Path to the prompts YAML file. If None, looks for prompts.yaml
                         in the data directory.
        """
        if prompts_path is None:
            # Look for prompts.yaml in the data directory
            prompts_path = Path("data") / "prompts.yaml"
            
        self.prompts_path = Path(prompts_path)
        self.prompts = self._load_prompts()
        
    def _load_prompts(self) -> Dict[str, str]:
        """Load prompt templates from YAML file."""
        try:
            if not self.prompts_path.exists():
                raise FileNotFoundError(f"Prompts file not found: {self.prompts_path}")
                
            with open(self.prompts_path, 'r', encoding='utf-8') as f:
                prompts = yaml.safe_load(f)
                
            logger.info(f"Prompts loaded from {self.prompts_path}")
            return prompts
            
        except FileNotFoundError:
            logger.error(f"Prompts file not found: {self.prompts_path}")
            raise
        except yaml.YAMLError as e:
            logger.error(f"Error parsing YAML prompts file: {e}")
            raise
        except Exception as e:
            logger.error(f"Error loading prompts: {e}")
            raise
    
    def get_voting_prompt(self, **kwargs) -> str:
        """
        Get the voting prompt with variable substitution.
        
        Args:
            **kwargs: Variables to substitute in the prompt template
            
        Returns:
            Formatted voting prompt
        """
        template = self.prompts.get("voting_prompt", "")
        return template.format(**kwargs)
    

    def get_discussion_prompt(self, **kwargs) -> str:
        """
        Get the discussion prompt with variable substitution.
        
        Args:
            **kwargs: Variables to substitute in the prompt template
            
        Returns:
            Formatted discussion prompt
        """
        template = self.prompts.get("discussion_prompt", "")
        return template.format(**kwargs)
    

    def get_internal_stance_prompt(self, **kwargs) -> str:
        """
        Get the internal stance prompt with variable substitution.
        
        Args:
            **kwargs: Variables to substitute in the prompt template
            
        Returns:
            Formatted internal stance prompt
        """
        template = self.prompts.get("internal_stance_prompt", "")
        return template.format(**kwargs)
    

    def reload_prompts(self):
        """Reload prompt templates from file."""
        self.prompts = self._load_prompts()
        logger.info("Prompts reloaded")
    
    def __str__(self) -> str:
        """String representation of prompt manager."""
        return f"PromptManager(prompts_path={self.prompts_path}, templates_loaded={len(self.prompts)})"
    
    def __repr__(self) -> str:
        return self.__str__()


# Global prompt manager instance
_prompt_manager = None

def get_prompt_manager(prompts_path: Optional[str] = None) -> PromptManager:
    """
    Get the global prompt manager instance.
    
    Args:
        prompts_path: Path to prompts file. Only used on first call.
        
    Returns:
        PromptManager instance.
    """
    global _prompt_manager
    if _prompt_manager is None:
        _prompt_manager = PromptManager(prompts_path)
    return _prompt_manager

def reload_prompts():
    """Reload the global prompt templates."""
    global _prompt_manager
    if _prompt_manager is not None:
        _prompt_manager.reload_prompts()
