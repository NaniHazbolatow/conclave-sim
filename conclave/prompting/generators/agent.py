"""
Agent-specific variable generator.

This module handles all agent-related variables like personas, stances, 
reflections, and agent attributes.
"""

import logging
from typing import Dict, List, Optional, Any
from datetime import datetime
from .base import BaseVariableGenerator

logger = logging.getLogger("conclave.prompting.generators.agent")


class AgentVariableGenerator(BaseVariableGenerator):
    """Generates agent-specific variables for prompt templates."""
    
    def generate_agent_variables(self, agent_id: int) -> Dict[str, Any]:
        """Generate all agent-specific variables for a given agent."""
        agent = self.get_agent_by_id(agent_id)
        if not agent:
            return self._get_fallback_agent_variables()
        
        return {
            'agent_id': agent_id,
            'agent_name': agent.name,
            'cardinal_id': getattr(agent, 'cardinal_id', f'ID_{agent_id}'),
            'agent_persona': self.get_agent_attribute(agent, 'internal_persona'),
            'agent_public_profile': self.get_agent_attribute(agent, 'public_profile'),
            'agent_stance': self.generate_current_stance(agent),
            'agent_stance_history': self.generate_stance_history(agent),
            'agent_reflection': self.get_agent_attribute(agent, 'current_reflection_digest', 'No reflection available.'),
            'agent_vote_history': self.generate_vote_history(agent),
            'agent_role_tag': self.get_agent_attribute(agent, 'role_tag', 'ELECTOR'),
            'stance_digest': self.generate_stance_digest(agent),
            'reflection_digest': self.generate_reflection_digest(agent),
            
            # CSV-based attributes for compatibility
            'biography': getattr(agent, 'background_csv', 'No background available'),
            'internal_persona': getattr(agent, 'internal_persona_csv', getattr(agent, 'internal_persona', 'No internal persona available')),
            'persona_internal': getattr(agent, 'internal_persona_csv', getattr(agent, 'internal_persona', 'No internal persona available')),
            'public_profile': getattr(agent, 'public_profile_csv', getattr(agent, 'public_profile', 'No public profile available')),
            'persona_tag': getattr(agent, 'persona_tag_csv', 'No persona tag available'),
            
            # Generated digest variables
            'stance_digest': self.generate_stance_digest(agent),
            'reflection_digest': self.generate_reflection_digest(agent),
        }
    
    def generate_current_stance(self, agent) -> str:
        """Generate the current stance of an agent."""
        if hasattr(agent, 'internal_stance') and agent.internal_stance:
            return agent.internal_stance
        elif hasattr(agent, 'internal_persona') and agent.internal_persona:
            return agent.internal_persona
        else:
            return "No stance available."
    
    def generate_stance_history(self, agent) -> str:
        """Generate the stance history for an agent."""
        if not hasattr(agent, 'stance_history') or not agent.stance_history:
            return "No stance history available."
        
        history_entries = []
        for i, stance in enumerate(agent.stance_history, 1):
            history_entries.append(f"Round {i}: {stance}")
        
        return "\n".join(history_entries)
    
    def generate_vote_history(self, agent) -> str:
        """Generate the voting history for an agent."""
        if not hasattr(agent, 'vote_history') or not agent.vote_history:
            return "No voting history available."
        
        history_entries = []
        for i, vote in enumerate(agent.vote_history, 1):
            # Handle different vote formats
            if isinstance(vote, dict):
                voted_for = vote.get('voted_for', 'Unknown')
            elif isinstance(vote, (int, str)):
                voted_for = str(vote)
            else:
                voted_for = str(vote)
            
            history_entries.append(f"Round {i}: Voted for {voted_for}")
        
        return "\n".join(history_entries)
    
    def generate_stance_digest(self, agent) -> str:
        """Generate stance digest for an agent."""
        stance = getattr(agent, 'internal_stance', None)
        return stance if stance else "No current stance"
    
    def generate_reflection_digest(self, agent) -> str:
        """Generate reflection digest for an agent."""
        reflection = getattr(agent, 'current_reflection_digest', None)
        return reflection if reflection else "No reflection available"

    def generate_agent_context(self, agent_id: int) -> str:
        """Generate a comprehensive context string for an agent."""
        agent = self.get_agent_by_id(agent_id)
        if not agent:
            return f"Agent {agent_id}: Information not available."
        
        context_parts = [
            f"Name: {agent.name}",
            f"Cardinal ID: {getattr(agent, 'cardinal_id', f'ID_{agent_id}')}",
            f"Role: {self.get_agent_attribute(agent, 'role_tag', 'ELECTOR')}",
        ]
        
        # Add persona if available
        persona = self.get_agent_attribute(agent, 'internal_persona')
        if persona and persona != "N/A":
            context_parts.append(f"Persona: {persona}")
        
        # Add current stance if different from persona
        stance = self.generate_current_stance(agent)
        if stance and stance != persona and stance != "No stance available.":
            context_parts.append(f"Current Stance: {stance}")
        
        return "\n".join(context_parts)
    
    def generate_agent_list(self, agent_ids: List[int], include_details: bool = False) -> str:
        """Generate a formatted list of agents."""
        if not agent_ids:
            return "No agents specified."
        
        agent_lines = []
        for agent_id in agent_ids:
            agent = self.get_agent_by_id(agent_id)
            if not agent:
                agent_lines.append(f"Agent {agent_id}: Information not available")
                continue
            
            if include_details:
                line = self.generate_agent_context(agent_id).replace('\n', ', ')
            else:
                line = f"{agent.name}"
                cardinal_id = getattr(agent, 'cardinal_id', None)
                if cardinal_id:
                    line = f"Cardinal {cardinal_id} - {line}"
            
            agent_lines.append(line)
        
        return "\n".join(agent_lines)
    
    def _get_fallback_agent_variables(self) -> Dict[str, Any]:
        """Return fallback variables when agent is not found."""
        return {
            'agent_id': -1,
            'agent_name': 'Unknown Agent',
            'cardinal_id': 'Unknown',
            'agent_persona': 'No persona available.',
            'agent_public_profile': 'No profile available.',
            'agent_stance': 'No stance available.',
            'agent_stance_history': 'No stance history available.',
            'agent_reflection': 'No reflection available.',
            'agent_vote_history': 'No voting history available.',
            'agent_role_tag': 'UNKNOWN',
        }
