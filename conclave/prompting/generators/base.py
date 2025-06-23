"""
Base variable generator with common utilities.

This module provides shared functionality for all variable generators,
eliminating redundant code and providing consistent interfaces.
"""

from typing import Dict, List, Optional, Any
from conclave.utils import get_logger, validate_agent_id, format_candidate_info

logger = get_logger("conclave.prompting.generators")

class BaseVariableGenerator:
    """Base class for all variable generators with common utilities."""
    
    def __init__(self, env, prompt_loader):
        self.env = env
        self.prompt_loader = prompt_loader
        self.logger = get_logger(self.__class__.__name__)
    
    def get_agent_by_id(self, agent_id: int):
        """Get agent by ID with error handling."""
        try:
            validate_agent_id(agent_id, len(self.env.agents), context="get_agent_by_id")
            return self.env.agents[agent_id]
        except Exception as e:
            self.logger.warning(f"Invalid agent ID {agent_id}: {e}")
            return None
    
    def get_agent_attribute(self, agent, attribute_name: str, default: str = "N/A") -> str:
        """Get agent attribute with fallback to default."""
        return getattr(agent, attribute_name, default)
    
    def format_candidate_line(self, agent, include_cardinal_id: bool = False, include_votes: bool = False, format_spec: str = 'full') -> str:
        """Format a standardized candidate line for listings."""
        name = agent.name
        cardinal_id_str = f"Cardinal {getattr(agent, 'cardinal_id', 'ID N/A')} - " if include_cardinal_id else ""
        
        public_profile = getattr(agent, 'public_profile_csv', 'Public Profile N/A')

        if format_spec == 'tags':
            details = public_profile.split('|')[0].strip()
        else:  # 'full'
            details = public_profile

        line = f"{cardinal_id_str}{name} – {details}"
        
        if include_votes and self.env.votingHistory:
            current_votes = self.env.votingHistory[-1]
            agent_votes = current_votes.get(agent.agent_id, 0)
            
            # Calculate momentum
            momentum_status = "stable"
            if len(self.env.votingHistory) > 1:
                previous_votes = self.env.votingHistory[-2]
                previous_agent_votes = previous_votes.get(agent.agent_id, 0)
                delta = agent_votes - previous_agent_votes
                
                if delta >= 2:
                    momentum_status = "gaining"
                elif delta <= -2:
                    momentum_status = "losing"
                elif agent_votes == 0 and previous_agent_votes > 0:
                    momentum_status = "dropped"
            elif agent_votes > 0:
                momentum_status = "gaining"
            
            line += f", {agent_votes} votes ({momentum_status})"
        
        return line
    
    def get_visible_candidate_agents(self) -> List[Any]:
        """
        Get the list of visible candidate agent objects for the current voting round.
        Returns top 5 candidates if votes have been cast, otherwise all designated candidates.
        """
        if not self.env.votingHistory:
            # Return all designated candidates (as agent objects)
            candidate_agent_ids = self.env.get_candidates_list()
            return [self.env.agents[aid] for aid in candidate_agent_ids if aid < len(self.env.agents)]

        # Votes have been cast, determine top 5
        current_votes = self.env.votingHistory[-1]
        designated_candidate_agent_ids = self.env.get_candidates_list()
        
        # Filter current_votes to only include designated candidates and get their vote counts
        candidate_votes = {
            agent_id: current_votes.get(agent_id, 0) 
            for agent_id in designated_candidate_agent_ids
        }

        # Sort these designated candidates by their votes (descending), then by agent_id (ascending) for tie-breaking
        sorted_candidate_agent_ids = sorted(
            candidate_votes.keys(),
            key=lambda aid: (candidate_votes[aid], -aid),
            reverse=True
        )
        
        # Take top 5
        top_5_agent_ids = sorted_candidate_agent_ids[:5]
        return [self.env.agents[aid] for aid in top_5_agent_ids if aid < len(self.env.agents)]
    
    def validate_participant_ids(self, participant_ids: List[int]) -> List[int]:
        """Validate and filter participant IDs, removing invalid ones."""
        if not participant_ids:
            self.logger.info("validate_participant_ids called with empty participant_ids.")
            return []
        
        valid_ids = []
        for pid in participant_ids:
            try:
                validate_agent_id(pid, len(self.env.agents), context="validate_participant_ids")
                valid_ids.append(pid)
            except Exception as e:
                self.logger.warning(f"Invalid participant ID {pid}: {e}. Skipping.")
        
        if not valid_ids:
            self.logger.info("validate_participant_ids found no valid participants from the provided IDs.")
        
        return valid_ids
