"""
Simplified Prompt Variable Generator

This is the new unified prompt variable generator that uses decomposed generators
to provide all variables needed for prompt templates. It replaces the monolithic
PromptVariableGenerator with a cleaner, more maintainable approach.
"""

import json
import logging
import math
from typing import Dict, List, Optional, Any
from datetime import datetime

from .generators import BaseVariableGenerator, AgentVariableGenerator, VotingVariableGenerator, GroupVariableGenerator

logger = logging.getLogger("conclave.prompting.unified")


class UnifiedPromptVariableGenerator:
    """
    Unified generator that combines all specialized generators.
    
    This class provides a clean interface for generating all prompt variables
    while delegating to specialized generators for better maintainability.
    """
    
    def __init__(self, env, prompt_loader):
        """Initialize with environment and prompt loader."""
        self.env = env
        self.prompt_loader = prompt_loader
        self.logger = logging.getLogger(self.__class__.__name__)
        
        # Initialize specialized generators
        self.agent_generator = AgentVariableGenerator(env, prompt_loader)
        self.voting_generator = VotingVariableGenerator(env, prompt_loader)
        self.group_generator = GroupVariableGenerator(env, prompt_loader)
        self.base_generator = BaseVariableGenerator(env, prompt_loader)
    
    def generate_prompt(
        self,
        prompt_name: str,
        agent_id: Optional[int] = None,
        discussion_round: Optional[int] = None,
        discussion_group_ids: Optional[List[int]] = None,
        target_agent_id: Optional[int] = None,
        include_id_for_visible_candidates: bool = False,
        **kwargs
    ) -> str:
        """
        Generate a formatted prompt using the appropriate variables.
        
        Args:
            prompt_name: Name of the prompt template
            agent_id: ID of the agent for agent-specific prompts
            discussion_round: Current discussion round number
            discussion_group_ids: List of participant agent IDs for group prompts
            target_agent_id: ID of target agent for interaction prompts
            include_id_for_visible_candidates: Whether to include cardinal IDs in candidate lists
            **kwargs: Additional variables to include in the prompt
        
        Returns:
            Formatted prompt string
        """
        try:
            template_str = self.prompt_loader.get_prompt(prompt_name)
            
            if template_str is None:
                self.logger.error(f"Template '{prompt_name}' not found in prompt configuration")
                raise ValueError(f"Template '{prompt_name}' not found in prompt configuration")
            
            # Generate all variables
            variables = self.generate_all_variables(
                agent_id=agent_id,
                discussion_round=discussion_round,
                discussion_group_ids=discussion_group_ids,
                target_agent_id=target_agent_id,
                include_id_for_visible_candidates=include_id_for_visible_candidates
            )
            
            # Add any additional kwargs
            variables.update(kwargs)
            
            # Format the template
            formatted_prompt = template_str.format(**variables)
            
            self.logger.debug(f"Generated prompt '{prompt_name}' for agent {agent_id}")
            return formatted_prompt
            
        except KeyError as e:
            self.logger.error(f"Missing variable {e} for prompt '{prompt_name}'")
            raise ValueError(f"Missing required variable {e} for prompt '{prompt_name}'")
        except Exception as e:
            self.logger.error(f"Error generating prompt '{prompt_name}': {e}")
            raise
    
    def generate_all_variables(
        self,
        agent_id: Optional[int] = None,
        discussion_round: Optional[int] = None,
        discussion_group_ids: Optional[List[int]] = None,
        target_agent_id: Optional[int] = None,
        include_id_for_visible_candidates: bool = False
    ) -> Dict[str, Any]:
        """Generate all available variables for prompt templates."""
        variables = {}
        
        # Base environment variables
        variables.update(self.generate_environment_variables())
        
        # Voting variables
        variables.update(self.voting_generator.generate_voting_variables(
            include_cardinal_id_for_visible=include_id_for_visible_candidates
        ))
        
        # Agent-specific variables
        if agent_id is not None:
            agent_vars = self.agent_generator.generate_agent_variables(agent_id)
            variables.update(agent_vars)
            
            # Add variable mappings for legacy template compatibility
            variables.update({
                'role_tag': agent_vars.get('agent_role_tag', 'ELECTOR'),
                'persona_internal': agent_vars.get('agent_persona', 'No persona available.'),
                'reflection_digest': agent_vars.get('agent_reflection', 'No reflection available.'),
                'stance_digest': agent_vars.get('agent_stance', 'No stance available.'),
            })
        
        # Target agent variables
        if target_agent_id is not None:
            target_variables = self.agent_generator.generate_agent_variables(target_agent_id)
            # Prefix with 'target_' to avoid conflicts
            for key, value in target_variables.items():
                variables[f"target_{key}"] = value
        
        # Group/discussion variables
        if discussion_group_ids is not None:
            variables.update(self.group_generator.generate_group_variables(
                discussion_group_ids=discussion_group_ids,
                discussion_round=discussion_round
            ))
        
        # Add computed variables
        variables.update(self.generate_computed_variables())
        
        return variables
    
    def generate_environment_variables(self) -> Dict[str, Any]:
        """Generate basic environment variables."""
        return {
            'current_timestamp': datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            'total_agents': len(self.env.agents),
            'total_electors': len([a for a in self.env.agents if getattr(a, 'role_tag', 'ELECTOR') == 'ELECTOR']),
            'elector_count': len(self.env.agents),  # Total number of electors
            'current_round': getattr(self.env, 'votingRound', 0),  # Current voting round
            'current_round_number': getattr(self.env, 'votingRound', 0),  # Alias for current_round
            'max_rounds': getattr(self.env, 'max_election_rounds', 10),  # Maximum election rounds
            'total_rounds': getattr(self.env, 'max_election_rounds', 10),  # Alias for max_rounds
            'discussion_round': getattr(self.env, 'discussionRound', 0),  # Current discussion round
            'simulation_phase': getattr(self.env, 'current_phase', 'Unknown'),
        }
    
    def generate_computed_variables(self) -> Dict[str, Any]:
        """Generate computed variables based on current state."""
        variables = {}
        
        # Voting progress
        if self.env.votingHistory:
            current_round = len(self.env.votingHistory)
            variables['current_voting_round'] = current_round
            variables['voting_rounds_completed'] = current_round
            
            # Election progress
            current_votes = self.env.votingHistory[-1]
            if current_votes:
                max_votes = max(current_votes.values())
                required_majority = self.voting_generator.get_required_majority()
                progress_percentage = min(100, (max_votes / required_majority) * 100)
                variables['election_progress_percentage'] = f"{progress_percentage:.1f}%"
        else:
            variables['current_voting_round'] = 0
            variables['voting_rounds_completed'] = 0
            variables['election_progress_percentage'] = "0.0%"
        
        return variables
    
    def generate_prompt_variables_for_group_analysis(
        self,
        group_id: int,
        group_agent_ids: List[int],
        discussion_transcript: str,
        current_round: Optional[int] = None
    ) -> Dict[str, Any]:
        """
        Generate prompt variables for group discussion analysis.
        
        This is a compatibility method for the old API.
        """
        # Use the existing generate_all_variables method
        variables = self.generate_all_variables(
            discussion_round=current_round or self.env.discussionRound,
            discussion_group_ids=group_agent_ids
        )
        
        # Add group-specific variables
        variables.update({
            'group_id': group_id,
            'group_transcript_text': discussion_transcript,
            'group_size': len(group_agent_ids),
            'group_participants': ", ".join([self.env.agents[aid].name for aid in group_agent_ids if aid < len(self.env.agents)])
        })
        
        return variables

    # Legacy compatibility methods - delegate to specialized generators
    
    def generate_visible_candidates(self, include_cardinal_id: bool = False) -> str:
        """Legacy method - use voting generator."""
        return self.voting_generator.generate_candidate_list(include_cardinal_id=include_cardinal_id)
    
    def generate_viable_papabili(self, include_cardinal_id: bool = False) -> str:
        """Legacy method - use voting generator with votes."""
        return self.voting_generator.generate_candidate_list(
            include_cardinal_id=include_cardinal_id, 
            include_votes=True
        )
    
    def generate_agent_persona(self, agent_id: int) -> str:
        """Legacy method - use agent generator."""
        variables = self.agent_generator.generate_agent_variables(agent_id)
        return variables.get('agent_persona', 'No persona available.')
    
    def generate_discussion_participants(self, participant_ids: List[int]) -> str:
        """Legacy method - use group generator."""
        return self.group_generator.generate_participant_list(participant_ids)
    
    def get_agent_by_id(self, agent_id: int):
        """Delegate to base generator."""
        return self.base_generator.get_agent_by_id(agent_id)
    
    def validate_participant_ids(self, participant_ids: List[int]) -> List[int]:
        """Delegate to base generator."""
        return self.base_generator.validate_participant_ids(participant_ids)
