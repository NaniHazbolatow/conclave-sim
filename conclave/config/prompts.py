"""
Prompt management for the Conclave Simulation.
Loads and formats prompt templates from external files.
"""

import yaml
import logging
from typing import Dict, Any, Optional
from pathlib import Path
from .prompt_variables import PromptVariableGenerator

logger = logging.getLogger(__name__)

class PromptManager:
    """Manages loading and formatting of prompt templates."""
    
    def __init__(self, prompts_path: Optional[str] = None, env=None):
        """
        Initialize the prompt manager.
        
        Args:
            prompts_path: Path to the prompts YAML file. If None, looks for prompts.yaml
                         in the data directory.
            env: Environment instance for variable generation
        """
        if prompts_path is None:
            # Look for prompts.yaml in the data directory
            prompts_path = Path("data") / "prompts.yaml"
            
        self.prompts_path = Path(prompts_path)
        self.prompts = self._load_prompts()
        self.env = env
        self.variable_generator = None
        
        if env is not None:
            self.variable_generator = PromptVariableGenerator(env)
        
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
    
    def set_environment(self, env):
        """Set the environment and initialize variable generator."""
        self.env = env
        self.variable_generator = PromptVariableGenerator(env)
    
    def get_prompt(self, prompt_name: str, agent_id: int = None, **extra_vars) -> str:
        """
        Get a prompt template with full variable substitution.
        
        Args:
            prompt_name: Name of the prompt template
            agent_id: Agent ID for generating agent-specific variables
            **extra_vars: Additional variables to substitute
            
        Returns:
            Formatted prompt with all variables substituted
        """
        if prompt_name not in self.prompts:
            raise ValueError(f"Prompt template '{prompt_name}' not found")
        
        template = self.prompts[prompt_name]
        variables = {}

        # Generate agent-specific variables
        if agent_id is not None and self.variable_generator is not None:
            agent_vars = self.variable_generator.generate_agent_variables(agent_id, prompt_name)
            variables.update(agent_vars)

        # Handle group-specific variables
        if self.variable_generator:
            participant_ids_source = extra_vars.get('participant_ids')
            if participant_ids_source is None and self.env and hasattr(self.env, '_current_discussion_speakers') and self.env._current_discussion_speakers:
                participant_ids_source = list(self.env._current_discussion_speakers)

            processed_participant_ids = []
            if participant_ids_source:
                try:
                    if not all(isinstance(pid, int) for pid in participant_ids_source):
                        logger.debug(f"Attempting conversion of participant IDs for prompt '{prompt_name}': {participant_ids_source}")
                        processed_participant_ids = [int(pid) for pid in participant_ids_source]
                    else:
                        processed_participant_ids = list(participant_ids_source) # Ensure it's a list
                except (ValueError, TypeError) as e:
                    logger.warning(f"Invalid participant IDs provided for prompt '{prompt_name}': {participant_ids_source}. Error: {e}. Group variables will use an empty participant list.")
                    # processed_participant_ids remains []
            
            # For discussion prompts, always attempt to generate group_vars.
            # generate_group_variables internally calls generate_group_profiles, which handles empty lists.
            # For other prompts, only generate if processed_participant_ids is actually populated.
            if prompt_name.startswith("discussion_") or processed_participant_ids:
                try:
                    group_vars = self.variable_generator.generate_group_variables(processed_participant_ids)
                    variables.update(group_vars)
                    if prompt_name.startswith("discussion_") and not processed_participant_ids:
                        logger.info(f"No valid participant IDs were available for discussion prompt '{prompt_name}'. 'group_profiles' was generated with a default message by PromptVariableGenerator.")
                except Exception as e:
                    logger.error(f"Error generating group variables for prompt '{prompt_name}' with processed IDs {processed_participant_ids}: {e}")
                    if prompt_name.startswith("discussion_"):
                        logger.warning(f"Group variables (like group_profiles) might be missing for '{prompt_name}' due to an error during their generation. Fallback will be used if 'group_profiles' is not in template.")
                        # Ensure 'group_profiles' is not in variables if its generation failed, to trigger fallback for sure.
                        variables.pop('group_profiles', None) 
        elif prompt_name.startswith("discussion_"): # This case means self.variable_generator is None
            logger.warning(f"Variable generator not available. Group variables (like group_profiles) for prompt '{prompt_name}' will rely on fallbacks.")
            # 'group_profiles' will be missing, and the existing fallback logic will handle it.

        variables.update(extra_vars) # Allow extra_vars to override any generated variables
        
        try:
            return template.format(**variables)
        except KeyError as e:
            original_missing_key = str(e).strip("'\"") 
            logger.error(f"Initial formatting failed. Missing variable '{original_missing_key}' for prompt '{prompt_name}'.")
            logger.info(f"Variables available before fallback attempt: {list(variables.keys())}")

            logger.info(f"Attempting to add fallback variables for prompt '{prompt_name}'.")
            # Note: _add_basic_fallback_variables expects agent_id: int. If agent_id is None,
            # f'Agent {agent_id}' in that method will become 'Agent None'.
            # The type hint in _add_basic_fallback_variables should ideally be Optional[int].
            self._add_basic_fallback_variables(variables, agent_id, prompt_name) # Modifies 'variables' in-place
            
            if original_missing_key not in variables:
                logger.warning(f"Fallback variable for the initially missing key '{original_missing_key}' was NOT set by _add_basic_fallback_variables for prompt '{prompt_name}'.")
            
            try:
                logger.info(f"Retrying formatting for prompt '{prompt_name}' with fallbacks included. Final variables for formatting: {list(variables.keys())}")
                return template.format(**variables)
            except KeyError as final_e:
                final_missing_key = str(final_e).strip("'\"")
                logger.error(f"Formatting failed even after fallback for prompt '{prompt_name}'. Missing variable '{final_missing_key}'.")
                logger.error(f"Final available variables after fallback attempt: {list(variables.keys())}")
                
                # For debugging, log what fallbacks _add_basic_fallback_variables would have set
                temp_fallback_check = {}
                # Use a placeholder for agent_id if it's None, to satisfy type hint for debug check if necessary,
                # though _add_basic_fallback_variables should ideally handle None.
                check_agent_id_for_debug = agent_id if agent_id is not None else -1 # Using -1 as a dummy non-None ID
                self._add_basic_fallback_variables(temp_fallback_check, check_agent_id_for_debug, prompt_name)
                logger.error(f"Expected fallbacks that _add_basic_fallback_variables would set for '{prompt_name}' (debug check with agent_id {check_agent_id_for_debug}): {list(temp_fallback_check.keys())}")
                raise final_e

    def _get_prompt_by_role(self, base_prompt_name: str, agent_id: int = None, **kwargs) -> str:
        """
        Helper to get a prompt based on agent's role (candidate or elector).
        
        Args:
            base_prompt_name: The base name for the prompt (e.g., "voting", "discussion").
                              The actual template name will be e.g. "voting_candidate" or "voting_elector".
            agent_id: ID of the agent requesting the prompt.
            **kwargs: Variables to substitute in the prompt template.
            
        Returns:
            Formatted prompt string.
        """
        template_name = f"{base_prompt_name}_elector"  # Default to elector
        if agent_id is not None and self.env is not None:
            # Check if the agent is a candidate
            # Assuming env has a method like get_candidates_list() or is_candidate(agent_id)
            try:
                # Adapt this check to your environment's actual method for identifying candidates
                if hasattr(self.env, 'get_candidates_list') and agent_id in self.env.get_candidates_list():
                    template_name = f"{base_prompt_name}_candidate"
                elif hasattr(self.env, 'is_candidate') and self.env.is_candidate(agent_id):
                     template_name = f"{base_prompt_name}_candidate"
            except Exception as e:
                logger.warning(f"Could not determine agent role for agent_id {agent_id}: {e}. Defaulting to elector prompt.")

        return self.get_prompt(template_name, agent_id, **kwargs)

    def get_voting_prompt(self, agent_id=None, **kwargs) -> str:
        """Get the voting prompt, dynamically selecting candidate/elector version."""
        return self._get_prompt_by_role("voting", agent_id, **kwargs)
    
    def get_discussion_prompt(self, agent_id=None, **kwargs) -> str:
        """Get the discussion prompt, dynamically selecting candidate/elector version."""
        return self._get_prompt_by_role("discussion", agent_id, **kwargs)
    
    def get_internal_stance_prompt(self, agent_id=None, **kwargs) -> str:
        """Get the internal stance prompt."""
        return self.get_prompt("stance", agent_id, **kwargs)
    
    def _add_basic_fallback_variables(self, variables: dict, agent_id: int, template_name: str):
        """Add basic fallback variables when variable generation fails."""
        # Common variables for all templates
        variables.setdefault('voting_round', self.env.votingRound)
        variables.setdefault('discussion_round', self.env.discussionRound)
        variables.setdefault('agent_name', f'Agent {agent_id}')
        variables.setdefault('threshold', '77')
        
        # Template-specific variables
        if template_name.startswith('voting_'):
            variables.setdefault('persona_internal', 'Cardinal profile')
            variables.setdefault('stance_digest', 'Initial stance')
            variables.setdefault('compact_scoreboard', 'No voting history yet')
            variables.setdefault('candidate_id_mapping', 'Cardinal 0: Agent 0\\nCardinal 1: Agent 1')
            
        elif template_name.startswith('discussion_'):
            variables.setdefault('persona_internal', 'Cardinal profile')
            variables.setdefault('stance_digest', 'Initial stance')
            variables.setdefault('compact_scoreboard', 'No voting history yet')
            variables.setdefault('group_profiles', 'Fellow cardinals participating in discussion')
            variables.setdefault('visible_candidates', 'All cardinals are potential candidates')
            
        elif template_name == 'stance':
            variables.setdefault('role_tag', 'ELECTOR')
            variables.setdefault('persona_internal', 'Cardinal profile')
            variables.setdefault('compact_scoreboard', 'No voting history yet')
            variables.setdefault('reflection_digest', 'Initial stance formation')
            variables.setdefault('visible_candidates', 'All cardinals are candidates')
    

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

def get_prompt_manager(prompts_path: Optional[str] = None, env=None) -> PromptManager:
    """
    Get the global prompt manager instance.
    
    Args:
        prompts_path: Path to prompts file. Only used on first call.
        env: Environment instance. Only used on first call.
        
    Returns:
        PromptManager instance.
    """
    global _prompt_manager
    if _prompt_manager is None:
        _prompt_manager = PromptManager(prompts_path, env)
    elif env is not None and _prompt_manager.env is None:
        _prompt_manager.set_environment(env)
    return _prompt_manager

def reload_prompts():
    """Reload the global prompt templates."""
    global _prompt_manager
    if _prompt_manager is not None:
        _prompt_manager.reload_prompts()
