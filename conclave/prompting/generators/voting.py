"""
Voting-specific variable generator.

This module handles all voting-related variables including candidate lists,
vote tallies, and election status information.
"""

import logging
from typing import Dict, List, Optional, Any
from .base import BaseVariableGenerator

logger = logging.getLogger("conclave.prompting.generators.voting")


class VotingVariableGenerator(BaseVariableGenerator):
    """Generates voting-specific variables for prompt templates."""
    
    def generate_voting_variables(self, include_cardinal_id_for_visible: bool = False) -> Dict[str, Any]:
        """Generate all voting-related variables."""
        return {
            'visible_candidates': self.generate_candidate_list(include_cardinal_id=include_cardinal_id_for_visible),
            'candidate_list': self.generate_candidate_list(include_cardinal_id=include_cardinal_id_for_visible),
            'viable_papabili': self.generate_candidate_list(include_cardinal_id=True, include_votes=True),
            'vote_tallies': self.generate_vote_tallies(),
            'voting_round': self.get_current_voting_round(),
            'total_votes_cast': self.get_total_votes_cast(),
            'required_majority': self.get_required_majority(),
            'election_status': self.get_election_status(),
            'compact_scoreboard': self.generate_compact_scoreboard(),
            'threshold': self.get_threshold(),
            'candidate_id_mapping': self.generate_candidate_id_mapping(),
            'visible_candidates_ids': [getattr(agent, 'cardinal_id', agent.agent_id) for agent in self.get_visible_candidate_agents()],
        }
    
    def generate_candidate_list(self, include_cardinal_id: bool = False, include_votes: bool = False) -> str:
        """
        Generate a standardized candidate list.
        
        This replaces both generate_visible_candidates and generate_viable_papabili
        with a unified approach that can handle both use cases.
        """
        candidate_agents = self.get_visible_candidate_agents()
        
        if not candidate_agents:
            return "No candidates available at this time."
        
        candidate_lines = []
        for agent in candidate_agents:
            line = self.format_candidate_line(
                agent, 
                include_cardinal_id=include_cardinal_id,
                include_votes=include_votes
            )
            candidate_lines.append(line)
        
        return "\n".join(candidate_lines)
    
    def generate_vote_tallies(self) -> str:
        """Generate current vote tallies for all candidates."""
        if not self.env.votingHistory:
            return "No votes have been cast yet."
        
        current_votes = self.env.votingHistory[-1]
        if not current_votes:
            return "No votes recorded for current round."
        
        # Sort by vote count (descending), then by agent ID for tie-breaking
        sorted_votes = sorted(
            current_votes.items(),
            key=lambda x: (x[1], -x[0]),  # votes descending, agent_id ascending for ties
            reverse=True
        )
        
        tally_lines = []
        for agent_id, vote_count in sorted_votes:
            if vote_count > 0:  # Only show candidates with votes
                agent = self.get_agent_by_id(agent_id)
                agent_name = agent.name if agent else f"Agent {agent_id}"
                tally_lines.append(f"{agent_name}: {vote_count} votes")
        
        if not tally_lines:
            return "No votes cast in current round."
        
        return "\n".join(tally_lines)
    
    def get_current_voting_round(self) -> int:
        """Get the current voting round number."""
        return len(self.env.votingHistory) if self.env.votingHistory else 0
    
    def get_total_votes_cast(self) -> int:
        """Get the total number of votes cast in the current round."""
        if not self.env.votingHistory:
            return 0
        
        current_votes = self.env.votingHistory[-1]
        return sum(current_votes.values()) if current_votes else 0
    
    def get_required_majority(self) -> int:
        """Calculate the required majority for election."""
        total_electors = len([agent for agent in self.env.agents if getattr(agent, 'role_tag', 'ELECTOR') == 'ELECTOR'])
        # Two-thirds majority: electors * (2/3), rounded up
        import math
        threshold = math.ceil(total_electors * (2/3))
        self.logger.debug(f"Calculated threshold: {threshold} for {total_electors} electors (total_electors * 2/3)")
        return threshold
    
    def get_election_status(self) -> str:
        """Get the current election status."""
        if not self.env.votingHistory:
            return "Election has not yet begun."
        
        current_votes = self.env.votingHistory[-1]
        if not current_votes:
            return "No votes recorded for current round."
        
        # Find highest vote count
        max_votes = max(current_votes.values()) if current_votes.values() else 0
        required_majority = self.get_required_majority()
        
        if max_votes >= required_majority:
            # Find winner(s)
            winners = [agent_id for agent_id, votes in current_votes.items() if votes == max_votes]
            if len(winners) == 1:
                winner_agent = self.get_agent_by_id(winners[0])
                winner_name = winner_agent.name if winner_agent else f"Agent {winners[0]}"
                return f"Election concluded. Winner: {winner_name} with {max_votes} votes."
            else:
                return f"Tie detected among {len(winners)} candidates with {max_votes} votes each."
        else:
            return f"No candidate has achieved required majority ({required_majority} votes). Highest: {max_votes} votes."
    
    def get_candidate_momentum(self, agent_id: int) -> str:
        """Get the voting momentum for a specific candidate."""
        if len(self.env.votingHistory) < 2:
            current_votes = self.env.votingHistory[-1] if self.env.votingHistory else {}
            agent_votes = current_votes.get(agent_id, 0)
            return "gaining" if agent_votes > 0 else "no momentum"
        
        current_votes = self.env.votingHistory[-1]
        previous_votes = self.env.votingHistory[-2]
        
        current_agent_votes = current_votes.get(agent_id, 0)
        previous_agent_votes = previous_votes.get(agent_id, 0)
        
        delta = current_agent_votes - previous_agent_votes
        
        if delta >= 2:
            return "gaining"
        elif delta <= -2:
            return "losing"
        elif current_agent_votes == 0 and previous_agent_votes > 0:
            return "dropped"
        else:
            return "stable"
    
    def generate_voting_summary(self) -> str:
        """Generate a comprehensive voting summary."""
        if not self.env.votingHistory:
            return "No voting has taken place yet."
        
        summary_parts = [
            f"Voting Round: {self.get_current_voting_round()}",
            f"Total Votes Cast: {self.get_total_votes_cast()}",
            f"Required Majority: {self.get_required_majority()}",
            f"Status: {self.get_election_status()}",
            "",
            "Current Vote Tallies:",
            self.generate_vote_tallies()
        ]
        
        return "\n".join(summary_parts)
    
    def generate_compact_scoreboard(self) -> str:
        """Generate compact scoreboard as defined in variable glossary."""
        if not self.env.votingHistory:
            return "No votes cast yet"
        
        current_votes = self.env.votingHistory[-1]
        previous_votes = self.env.votingHistory[-2] if len(self.env.votingHistory) > 1 else {}
        
        scoreboard_items = []
        sorted_candidates = sorted(current_votes.items(), key=lambda x: x[1], reverse=True)
        
        for candidate_id, votes in sorted_candidates:
            if candidate_id >= len(self.env.agents):
                continue
            
            candidate_name = self.env.agents[candidate_id].name
            
            if previous_votes:
                delta = votes - previous_votes.get(candidate_id, 0)
                if delta >= 2:
                    tag = "gaining"
                elif delta <= -2:
                    tag = "losing"
                elif votes == 0:
                    tag = "dropped"
                else:
                    tag = "stable"
                scoreboard_items.append(f"{candidate_name}:{votes} ({tag})")
            else:
                scoreboard_items.append(f"{candidate_name}:{votes}")
        
        return ", ".join(scoreboard_items)
    
    def get_threshold(self) -> int:
        """Get the voting threshold (same as required majority)."""
        return self.get_required_majority()
    
    def generate_candidate_id_mapping(self) -> str:
        """
        Generate a mapping showing Agent ID -> Cardinal ID for voting.
        
        This helps agents understand which IDs to use when voting.
        """
        candidate_agents = self.get_visible_candidate_agents()
        
        if not candidate_agents:
            return "No candidates available for voting."
        
        mapping_lines = []
        for agent in candidate_agents:
            cardinal_id = getattr(agent, 'cardinal_id', agent.agent_id)
            mapping_lines.append(f"{agent.name} (Cardinal ID: {cardinal_id})")
        
        return "\n".join(mapping_lines)
