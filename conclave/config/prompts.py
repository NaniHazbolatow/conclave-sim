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
        
        # Generate variables if we have an environment and agent
        variables = {}
        if self.variable_generator and agent_id is not None:
            variables.update(self.variable_generator.generate_agent_variables(agent_id))
        
        # Add extra variables
        variables.update(extra_vars)
        
        try:
            return template.format(**variables)
        except KeyError as e:
            logger.error(f"Missing variable {e} for prompt '{prompt_name}'")
            logger.error(f"Available variables: {list(variables.keys())}")
            raise
    
    def get_internal_persona_extractor_prompt(self, agent_name: str, biography: str) -> str:
        """Get the internal persona extractor prompt."""
        return self.get_prompt("internal_persona_extractor", 
                             agent_name=agent_name, 
                             biography=biography)
    
    def get_external_profile_generator_prompt(self, agent_name: str, persona_internal: str) -> str:
        """Get the external profile generator prompt."""
        return self.get_prompt("external_profile_generator",
                             agent_name=agent_name,
                             persona_internal=persona_internal)
    
    def get_discussion_candidate_prompt(self, agent_id: int, group_participants: list = None) -> str:
        """Get the discussion prompt for a candidate."""
        extra_vars = {}
        if group_participants and self.variable_generator:
            extra_vars.update(self.variable_generator.generate_group_variables(group_participants))
        
        return self.get_prompt("discussion_candidate", agent_id, **extra_vars)
    
    def get_discussion_elector_prompt(self, agent_id: int, group_participants: list = None) -> str:
        """Get the discussion prompt for an elector."""
        extra_vars = {}
        if group_participants and self.variable_generator:
            extra_vars.update(self.variable_generator.generate_group_variables(group_participants))
        
        return self.get_prompt("discussion_elector", agent_id, **extra_vars)
    
    def get_discussion_analyzer_prompt(self, round_id: int, group_transcript: str) -> str:
        """Get the discussion analyzer prompt."""
        return self.get_prompt("discussion_analyzer",
                             round_id=round_id,
                             group_transcript=group_transcript)
    
    def get_discussion_reflection_prompt(self, agent_id: int, group_summary: str, agent_last_utterance: str) -> str:
        """Get the discussion reflection prompt."""
        return self.get_prompt("discussion_reflection", agent_id,
                             group_summary=group_summary,
                             agent_last_utterance=agent_last_utterance)
    
    def get_stance_prompt(self, agent_id: int, reflection_digest: str = None) -> str:
        """Get the stance generation prompt."""
        extra_vars = {}
        if reflection_digest:
            extra_vars['reflection_digest'] = reflection_digest
        
        return self.get_prompt("stance", agent_id, **extra_vars)
    
    def get_voting_candidate_prompt(self, agent_id: int) -> str:
        """Get the voting prompt for a candidate."""
        return self.get_prompt("voting_candidate", agent_id)
    
    def get_voting_elector_prompt(self, agent_id: int) -> str:
        """Get the voting prompt for an elector."""
        return self.get_prompt("voting_elector", agent_id)
    
    def get_voting_prompt(self, agent_id=None, **kwargs) -> str:
        """
        Get the voting prompt with variable substitution.
        Uses the enhanced voting prompts (voting_candidate/voting_elector).
        
        Args:
            agent_id: ID of the agent requesting the prompt
            **kwargs: Variables to substitute in the prompt template
            
        Returns:
            Formatted voting prompt
        """
        # Check if agent is a candidate by looking at available candidates
        if agent_id is not None and self.env is not None:
            # Check if this agent is in the candidates list
            candidates = self.env.get_candidates_list()
            if agent_id in candidates:
                # Use candidate voting prompt
                template = self.prompts.get("voting_candidate", "")
            else:
                # Use elector voting prompt  
                template = self.prompts.get("voting_elector", "")
        else:
            # Fallback - try voting_elector as default
            template = self.prompts.get("voting_elector", "")
            
        # If we have a variable generator, use it to populate template variables
        if self.variable_generator and agent_id is not None:
            try:
                variables = self.variable_generator.generate_agent_variables(agent_id)
                # Merge with provided kwargs (kwargs take precedence)
                variables.update(kwargs)
                return template.format(**variables)
            except Exception as e:
                logger.warning(f"Error generating variables for agent {agent_id}: {e}")
                # Fall back to basic variables without complex ones
                basic_vars = {k: v for k, v in kwargs.items()}
                # Add missing basic variables for voting prompts
                if self.env:
                    basic_vars.setdefault('voting_round', self.env.votingRound)
                    basic_vars.setdefault('agent_name', f'Agent {agent_id}')
                    basic_vars.setdefault('persona_internal', 'Cardinal profile')
                    basic_vars.setdefault('stance_digest', 'Initial stance')
                    basic_vars.setdefault('compact_scoreboard', 'No voting history yet')
                    basic_vars.setdefault('threshold', '77')
                    basic_vars.setdefault('visible_candidates_ids', '[0, 1, 2, 3, 4]')
                return template.format(**basic_vars)
        else:
            # Ensure basic variables are present
            if self.env:
                kwargs.setdefault('voting_round', self.env.votingRound)
                kwargs.setdefault('threshold', '77')
            return template.format(**kwargs)
    

    def get_discussion_prompt(self, agent_id=None, **kwargs) -> str:
        """
        Get the discussion prompt with variable substitution.
        Uses the enhanced discussion prompts (discussion_candidate/discussion_elector).
        
        Args:
            agent_id: ID of the agent requesting the prompt
            **kwargs: Variables to substitute in the prompt template
            
        Returns:
            Formatted discussion prompt
        """
        # Check if agent is a candidate by looking at available candidates
        if agent_id is not None and self.env is not None:
            # Check if this agent is in the candidates list
            candidates = self.env.get_candidates_list()
            if agent_id in candidates:
                # Use candidate prompt
                template = self.prompts.get("discussion_candidate", "")
            else:
                # Use elector prompt  
                template = self.prompts.get("discussion_elector", "")
        else:
            # Fallback - try discussion_elector as default
            template = self.prompts.get("discussion_elector", "")
            
        # If we have a variable generator, use it to populate template variables
        if self.variable_generator and agent_id is not None:
            try:
                variables = self.variable_generator.generate_agent_variables(agent_id)
                # Merge with provided kwargs (kwargs take precedence)
                variables.update(kwargs)
                return template.format(**variables)
            except Exception as e:
                logger.warning(f"Error generating variables for agent {agent_id}: {e}")
                # Fall back to basic variables without complex ones
                basic_vars = {k: v for k, v in kwargs.items()}
                # Add missing basic variables for discussion prompts
                if self.env:
                    basic_vars.setdefault('discussion_round', self.env.discussionRound)
                    basic_vars.setdefault('voting_round', self.env.votingRound)
                    basic_vars.setdefault('agent_name', f'Agent {agent_id}')
                    basic_vars.setdefault('persona_internal', 'Cardinal profile')
                    basic_vars.setdefault('stance_digest', 'Initial stance')
                    basic_vars.setdefault('compact_scoreboard', 'No voting history yet')
                    basic_vars.setdefault('threshold', '77')
                    basic_vars.setdefault('group_profiles', 'Fellow cardinals participating in discussion')
                    # Generate proper visible candidates even in fallback
                    if 'visible_candidates' not in basic_vars:
                        try:
                            from .prompt_variables import PromptVariableGenerator
                            temp_gen = PromptVariableGenerator(self.env)
                            basic_vars['visible_candidates'] = temp_gen.generate_visible_candidates()
                        except:
                            basic_vars['visible_candidates'] = 'All cardinals are potential candidates'
                return template.format(**basic_vars)
        else:
            # Ensure basic variables are present
            if self.env:
                kwargs.setdefault('discussion_round', self.env.discussionRound)
                kwargs.setdefault('voting_round', self.env.votingRound)
            return template.format(**kwargs)
    

    def get_internal_stance_prompt(self, agent_id=None, **kwargs) -> str:
        """
        Get the internal stance prompt with variable substitution.
        Uses the enhanced stance prompt.
        
        Args:
            agent_id: ID of the agent requesting the prompt
            **kwargs: Variables to substitute in the prompt template
            
        Returns:
            Formatted internal stance prompt
        """
        template = self.prompts.get("stance", "")
        
        # If we have a variable generator, use it to populate template variables
        if self.variable_generator and agent_id is not None:
            try:
                variables = self.variable_generator.generate_agent_variables(agent_id)
                # Merge with provided kwargs (kwargs take precedence)
                variables.update(kwargs)
                return template.format(**variables)
            except Exception as e:
                logger.warning(f"Error generating variables for agent {agent_id}: {e}")
                # Fall back to basic variables without complex ones
                basic_vars = {k: v for k, v in kwargs.items()}
                # Add missing basic variables
                if self.env:
                    basic_vars.setdefault('voting_round', self.env.votingRound)
                    basic_vars.setdefault('threshold', '77')  # Default threshold
                    basic_vars.setdefault('agent_name', f'Agent {agent_id}')
                    basic_vars.setdefault('role_tag', 'ELECTOR')
                    basic_vars.setdefault('persona_internal', 'Cardinal profile')
                    basic_vars.setdefault('compact_scoreboard', 'No voting history yet')
                    basic_vars.setdefault('reflection_digest', 'Initial stance formation')
                    # Generate proper visible candidates even in fallback
                    if 'visible_candidates' not in basic_vars:
                        try:
                            from .prompt_variables import PromptVariableGenerator
                            temp_gen = PromptVariableGenerator(self.env)
                            basic_vars['visible_candidates'] = temp_gen.generate_visible_candidates()
                        except:
                            basic_vars['visible_candidates'] = 'All cardinals are candidates'
                return template.format(**basic_vars)
        else:
            # Ensure basic variables are present
            if self.env:
                kwargs.setdefault('voting_round', self.env.votingRound)
                kwargs.setdefault('threshold', '77')
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
