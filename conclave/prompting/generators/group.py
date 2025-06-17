"""
Group-specific variable generator.

This module handles all group and discussion-related variables including
participant lists, group dynamics, and discussion context.
"""

import logging
from typing import Dict, List, Optional, Any
from .base import BaseVariableGenerator

logger = logging.getLogger("conclave.prompting.generators.group")


class GroupVariableGenerator(BaseVariableGenerator):
    """Generates group and discussion-specific variables for prompt templates."""
    
    def generate_group_variables(self, 
                                discussion_group_ids: Optional[List[int]] = None,
                                discussion_round: Optional[int] = None) -> Dict[str, Any]:
        """Generate all group-related variables."""
        if discussion_group_ids:
            participant_ids = self.validate_participant_ids(discussion_group_ids)
        else:
            participant_ids = []
        
        return {
            'discussion_participants': self.generate_participant_list(participant_ids),
            'discussion_group_size': len(participant_ids),
            'discussion_round': discussion_round or 0,
            'group_dynamics': self.generate_group_dynamics(participant_ids),
            'discussion_context': self.generate_discussion_context(participant_ids, discussion_round),
            'room_support_counts': self.generate_room_support_counts(participant_ids),
        }
    
    def generate_participant_list(self, participant_ids: List[int]) -> str:
        """Generate a formatted list of discussion participants."""
        if not participant_ids:
            return "No participants specified for this discussion."
        
        valid_ids = self.validate_participant_ids(participant_ids)
        if not valid_ids:
            return "No valid participants found for this discussion."
        
        participant_lines = []
        for participant_id in valid_ids:
            agent = self.get_agent_by_id(participant_id)
            if agent:
                # Include name, cardinal ID, and basic persona info
                cardinal_id = getattr(agent, 'cardinal_id', f'ID_{participant_id}')
                name = agent.name
                persona_snippet = self.get_agent_attribute(agent, 'internal_persona', 'No persona available')
                
                # Truncate persona if too long
                if len(persona_snippet) > 100:
                    persona_snippet = persona_snippet[:97] + "..."
                
                line = f"Cardinal {cardinal_id} - {name}: {persona_snippet}"
                participant_lines.append(line)
        
        return "\n".join(participant_lines)
    
    def generate_group_dynamics(self, participant_ids: List[int]) -> str:
        """Generate information about group dynamics and composition."""
        if not participant_ids:
            return "No group dynamics available - no participants."
        
        valid_ids = self.validate_participant_ids(participant_ids)
        if not valid_ids:
            return "No valid participants for group dynamics analysis."
        
        dynamics_parts = [
            f"Group size: {len(valid_ids)} participants"
        ]
        
        # Analyze role distribution
        role_counts = {}
        for participant_id in valid_ids:
            agent = self.get_agent_by_id(participant_id)
            if agent:
                role = self.get_agent_attribute(agent, 'role_tag', 'ELECTOR')
                role_counts[role] = role_counts.get(role, 0) + 1
        
        if role_counts:
            role_info = ", ".join([f"{count} {role}" for role, count in role_counts.items()])
            dynamics_parts.append(f"Composition: {role_info}")
        
        # Add voting behavior analysis if voting history exists
        if self.env.votingHistory and len(valid_ids) > 1:
            voting_diversity = self.analyze_voting_diversity(valid_ids)
            if voting_diversity:
                dynamics_parts.append(f"Voting diversity: {voting_diversity}")
        
        return "\n".join(dynamics_parts)
    
    def analyze_voting_diversity(self, participant_ids: List[int]) -> str:
        """Analyze voting diversity among participants."""
        if not self.env.votingHistory or not participant_ids:
            return "No voting data available"
        
        current_votes = self.env.votingHistory[-1]
        
        # Get votes cast by participants
        participant_votes = {}
        for participant_id in participant_ids:
            agent = self.get_agent_by_id(participant_id)
            if agent and hasattr(agent, 'vote_history') and agent.vote_history:
                # Get most recent vote
                last_vote = agent.vote_history[-1]
                if isinstance(last_vote, dict):
                    voted_for = last_vote.get('voted_for')
                else:
                    voted_for = last_vote
                
                if voted_for is not None:
                    participant_votes[participant_id] = voted_for
        
        if not participant_votes:
            return "No voting data available for participants"
        
        # Count unique candidates voted for
        unique_votes = len(set(participant_votes.values()))
        total_participants = len(participant_votes)
        
        if unique_votes == 1:
            return "Unanimous voting pattern"
        elif unique_votes == total_participants:
            return "Highly diverse voting - no consensus"
        else:
            return f"Moderate diversity - {unique_votes} different choices among {total_participants} voters"
    
    def generate_discussion_context(self, 
                                  participant_ids: List[int], 
                                  discussion_round: Optional[int] = None) -> str:
        """Generate context for a discussion session."""
        if not participant_ids:
            return "No discussion context available - no participants specified."
        
        valid_ids = self.validate_participant_ids(participant_ids)
        if not valid_ids:
            return "No valid participants for discussion context."
        
        context_parts = []
        
        # Basic context
        round_info = f"Discussion Round {discussion_round}" if discussion_round else "Discussion Session"
        context_parts.append(f"{round_info} with {len(valid_ids)} participants")
        
        # Current election state
        if self.env.votingHistory:
            voting_round = len(self.env.votingHistory)
            context_parts.append(f"Following Voting Round {voting_round}")
            
            # Add brief voting summary
            current_votes = self.env.votingHistory[-1]
            if current_votes:
                max_votes = max(current_votes.values())
                required_majority = self.get_required_majority()
                
                if max_votes >= required_majority:
                    context_parts.append("Election may be concluded - majority achieved")
                else:
                    context_parts.append(f"No majority yet - highest candidate has {max_votes} votes, need {required_majority}")
        else:
            context_parts.append("Prior to first voting round")
        
        return "\n".join(context_parts)
    
    def get_required_majority(self) -> int:
        """Calculate the required majority for election (helper method)."""
        total_electors = len([agent for agent in self.env.agents if getattr(agent, 'role_tag', 'ELECTOR') == 'ELECTOR'])
        return (2 * total_electors + 2) // 3
    
    def generate_participant_stances(self, participant_ids: List[int]) -> str:
        """Generate a summary of participant stances."""
        if not participant_ids:
            return "No participants specified."
        
        valid_ids = self.validate_participant_ids(participant_ids)
        if not valid_ids:
            return "No valid participants found."
        
        stance_lines = []
        for participant_id in valid_ids:
            agent = self.get_agent_by_id(participant_id)
            if agent:
                name = agent.name
                
                # Get current stance
                if hasattr(agent, 'internal_stance') and agent.internal_stance:
                    stance = agent.internal_stance
                elif hasattr(agent, 'internal_persona') and agent.internal_persona:
                    stance = agent.internal_persona
                else:
                    stance = "No stance available"
                
                # Truncate if too long
                if len(stance) > 150:
                    stance = stance[:147] + "..."
                
                stance_lines.append(f"{name}: {stance}")
        
        return "\n".join(stance_lines)
    
    def generate_group_summary(self, 
                             participant_ids: List[int],
                             discussion_round: Optional[int] = None) -> str:
        """Generate a comprehensive group summary."""
        if not participant_ids:
            return "No group summary available - no participants."
        
        summary_parts = [
            self.generate_discussion_context(participant_ids, discussion_round),
            "",
            "Participants:",
            self.generate_participant_list(participant_ids),
            "",
            "Group Dynamics:",
            self.generate_group_dynamics(participant_ids),
        ]
        
        return "\n".join(summary_parts)
    
    def generate_room_support_counts(self, participant_ids: List[int]) -> str:
        """
        Generate room support counts showing declared supporters in this discussion group.
        Format: "Parolin 2 · Tagle 1 · Arinze 0 · Undecided 4"
        """
        if not participant_ids:
            return "No participants in room"
        
        valid_ids = self.validate_participant_ids(participant_ids)
        if not valid_ids:
            return "No valid participants in room"
        
        try:
            # Use the latest individual votes if available
            if not hasattr(self.env, 'individual_votes_history') or not self.env.individual_votes_history:
                # Fallback: no voting data available yet
                return f"Room composition: {len(valid_ids)} participants (no voting data yet)"
            
            latest_individual_votes = self.env.individual_votes_history[-1] if self.env.individual_votes_history else {}
            
            # Count support for each visible candidate among room participants
            support_counts = {}
            undecided_count = 0
            
            visible_candidates = self.base_generator.get_visible_candidate_agents() if hasattr(self, 'base_generator') else []
            
            # Count votes from room participants
            for participant_id in valid_ids:
                vote = latest_individual_votes.get(participant_id)
                if vote is not None:
                    # Find the candidate name
                    candidate_agent = self.get_agent_by_id(vote) if isinstance(vote, int) else None
                    candidate_name = candidate_agent.name if candidate_agent else f"Agent_{vote}"
                    support_counts[candidate_name] = support_counts.get(candidate_name, 0) + 1
                else:
                    undecided_count += 1
            
            # Format the result
            count_parts = []
            for candidate_name, count in sorted(support_counts.items()):
                count_parts.append(f"{candidate_name} {count}")
            
            if undecided_count > 0:
                count_parts.append(f"Undecided {undecided_count}")
            
            return " · ".join(count_parts) if count_parts else f"{len(valid_ids)} participants (no declared votes)"
            
        except Exception as e:
            self.logger.warning(f"Error generating room support counts: {e}")
            return f"Room composition: {len(valid_ids)} participants"
