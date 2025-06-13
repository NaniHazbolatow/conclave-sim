"""
Prompt Variable Generator

This module generates all the variables defined in the variable glossary
for use in the literature-grounded prompt templates.
"""

import json
import logging
from typing import Dict, List, Optional, Any
from datetime import datetime

logger = logging.getLogger(__name__)

class PromptVariableGenerator:
    """Generates all variables from the variable glossary for prompt templates."""
    
    def __init__(self, env):
        self.env = env
        self.logger = logging.getLogger(self.__class__.__name__)
    
    def generate_agent_variables(self, agent_id: int, prompt_name: str) -> Dict[str, Any]: # Added prompt_name
        """Generate all agent-specific variables for prompts."""
        agent = self.env.agents[agent_id]
        
        # Determine if Cardinal ID should be included in visible_candidates
        # Show IDs for internal stance and voting, not for discussions.
        # Assuming voting prompts might also use 'visible_candidates' or a similar setup.
        # For now, being explicit: show ID unless it's a discussion prompt.
        include_id_for_visible_candidates = prompt_name not in ["discussion_elector", "discussion_candidate"]
        
        variables = {
            'agent_name': agent.name,
            'cardinal_id': agent.cardinal_id, 
            'role_tag': getattr(agent, 'role_tag', 'ELECTOR'),
            'biography': agent.background,
            'persona_internal': agent.internal_persona,
            'profile_public': agent.public_profile,
            'persona_tag': agent.persona_tag or 'N/A',
            'profile_blurb': agent.profile_blurb or 'N/A',
            'discussion_round': self.env.discussionRound,
            'voting_round': self.env.votingRound,
            'threshold': self.env._calculate_voting_threshold(),
            'compact_scoreboard': self.generate_compact_scoreboard(),
            'visible_candidates': self.generate_visible_candidates(include_cardinal_id=include_id_for_visible_candidates), 
            'visible_candidates_ids': self.generate_visible_candidate_ids(), # Restored
            'candidate_id_mapping': self.generate_candidate_id_mapping(), # Restored
            'stance_digest': self.generate_stance_digest(agent_id),
            'reflection_digest': self.generate_reflection_digest(agent_id),
            'recent_speech_snippets': self.generate_recent_speech_snippets(agent_id),
        }
        return variables

    def generate_group_variables(self, participant_ids: List[int]) -> Dict[str, Any]:
        """Generate group-specific variables for discussion prompts."""
        return {
            'group_profiles': self.generate_group_profiles(participant_ids),
        }

    def get_visible_candidates_ids(self) -> List[int]:
        """Get the list of visible candidate IDs for the current voting round."""
        if self.env.votingRound == 1 or not self.env.votingHistory:
            return self.env.get_candidates_list()
        current_votes = self.env.votingHistory[-1]
        sorted_candidates = sorted(current_votes.items(), key=lambda x: x[1], reverse=True)
        valid_candidates = self.env.get_candidates_list()
        return [cid for cid, _ in sorted_candidates[:5] if cid in valid_candidates]

    def get_visible_candidates_cardinal_ids(self) -> List[int]:
        """Get the list of visible candidate Cardinal IDs for the current voting round."""
        return [getattr(self.env.agents[aid], 'cardinal_id', aid) for aid in self.get_visible_candidates_ids()]

    def generate_compact_scoreboard(self) -> str:
        """Generate compact_scoreboard variable as defined in glossary."""
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
        return "; ".join(scoreboard_items)

    def generate_visible_candidates(self, include_cardinal_id: bool) -> str: # Added include_cardinal_id flag
        """Generate visible_candidates variable with names, (optional) Cardinal IDs, profile blurbs, votes, and momentum."""
        lines = []
        visible_candidate_agents = self.get_visible_candidates() # List of agent objects

        if not self.env.votingHistory:
            # No votes cast yet, so don't show vote count or momentum
            for agent in visible_candidate_agents:
                name = agent.name
                cardinal_id_str = f"Cardinal {agent.cardinal_id or 'ID N/A'} - " if include_cardinal_id else ""
                profile_blurb = agent.profile_blurb or 'Blurb N/A'
                lines.append(f"{cardinal_id_str}{name} – {profile_blurb}")
            return "\\\\n".join(lines)

        current_votes_map = self.env.votingHistory[-1]
        previous_votes_map = self.env.votingHistory[-2] if len(self.env.votingHistory) > 1 else {}

        for agent in visible_candidate_agents:
            name = agent.name
            agent_id_key = agent.agent_id
            cardinal_id_str = f"Cardinal {agent.cardinal_id or 'ID N/A'} - " if include_cardinal_id else ""
            profile_blurb = agent.profile_blurb or 'Blurb N/A'
            
            current_agent_votes = current_votes_map.get(agent_id_key, 0)
            
            momentum_status = "stable"
            if not previous_votes_map: 
                if current_agent_votes > 0:
                    momentum_status = "gaining"
            else:
                previous_agent_votes = previous_votes_map.get(agent_id_key, 0)
                delta = current_agent_votes - previous_agent_votes
                if delta >= 2:
                    momentum_status = "gaining"
                elif delta <= -2:
                    momentum_status = "losing"
                elif current_agent_votes == 0 and previous_agent_votes > 0:
                    momentum_status = "dropped"

            lines.append(f"{cardinal_id_str}{name} – {profile_blurb}, {current_agent_votes} votes ({momentum_status})")
        return "\\\\n".join(lines)

    def generate_stance_digest(self, agent_id: int) -> str:
        """Generate stance_digest variable as defined in glossary."""
        stance = getattr(self.env.agents[agent_id], 'internal_stance', None)
        return stance if stance else "No current stance"

    def generate_group_profiles(self, participant_ids: List[int]) -> str:
        """Generate group_profiles variable as defined in glossary."""
        if not participant_ids:
            self.logger.info("generate_group_profiles called with empty participant_ids. Returning default message.")
            return "No participants currently specified for this discussion."

        profiles = []
        for pid in participant_ids:
            if not isinstance(pid, int) or pid < 0 or pid >= len(self.env.agents):
                self.logger.warning(f"Invalid participant ID {pid} (type: {type(pid)}, value: {pid}). Max agent index: {len(self.env.agents)-1 if self.env.agents else 'N/A'}). Skipping.")
                continue
            
            agent = self.env.agents[pid]
            name = agent.name
            persona_tag = getattr(agent, 'persona_tag', 'Tag N/A') # Using 'Tag N/A' for clarity
            
            # Changed format to be more explicit about the tag and removed "()"
            profiles.append(f"• {name} – Tag: {persona_tag}") 

        if not profiles:
            self.logger.info("generate_group_profiles found no valid participants from the provided IDs. Returning default message.")
            return "No valid participants found from the provided IDs for this discussion."
            
        return "\\\\n".join(profiles)

    def generate_relation_for_participant(self, participant_id: int, all_participants: List[int]) -> str:
        """Generate relation label for a participant (sympathetic, neutral, opposed)."""
        # Placeholder: always neutral
        return "neutral"

    def generate_group_transcript(self, discussion_round: int) -> str:
        """Generate group_transcript variable as defined in glossary."""
        if discussion_round <= len(self.env.discussionHistory):
            round_comments = self.env.discussionHistory[discussion_round - 1]
            return "\n".join(f"{self.env.agents[c['agent_id']].name}: {c['message']}" for c in round_comments)
        return "No transcript available"

    def generate_group_summary(self, discussion_round: int) -> str:
        """Generate group_summary variable as JSON object as defined in glossary."""
        if discussion_round <= len(self.env.discussionHistory):
            round_comments = self.env.discussionHistory[discussion_round - 1]
            key_points = []
            speakers = []
            for c in round_comments:
                speaker_name = self.env.agents[c['agent_id']].name
                message = c['message']
                speakers.append(speaker_name)
                first_sentence = message.split('.')[0].strip()
                words = first_sentence.split()
                if len(words) > 25:
                    first_sentence = " ".join(words[:25]) + "..."
                key_points.append(first_sentence)
            while len(key_points) < 3:
                key_points.append("N/A")
            while len(speakers) < 3:
                speakers.append("N/A")
            summary = {
                "key_points": key_points[:3],
                "speakers": speakers[:3],
                "overall_tone": "mixed"
            }
            return json.dumps(summary)
        return json.dumps({"key_points": ["N/A"]*3, "speakers": ["N/A"]*3, "overall_tone": "mixed"})

    def generate_agent_last_utterance(self, agent_id: int) -> str:
        """Generate agent_last_utterance variable as defined in glossary."""
        for round_comments in reversed(self.env.discussionHistory):
            for comment in round_comments:
                if comment['agent_id'] == agent_id:
                    return comment['message']
        return "No previous utterance"

    def generate_reflection_digest(self, agent_id: int) -> str:
        """Generate reflection_digest variable as defined in glossary."""
        agent = self.env.agents[agent_id]
        if self.env.discussionRound == 0:
            # return "" # Return a more informative default for initial stance
            return "Initial assessment of the conclave situation and potential candidates."
        stance = getattr(agent, 'internal_stance', None)
        if not stance:
            return "Analyzing current situation and candidate options."
        words = stance.split()
        reflection = " ".join(words[:25]) if len(words) > 25 else stance
        return reflection.replace('\n', ' ').replace('"', '').replace("'", '').strip()

    def generate_visible_candidate_ids(self) -> str:
        """Generate list of Cardinal IDs for voting (JSON format)."""
        return json.dumps(self.get_visible_candidates_cardinal_ids())

    def generate_candidate_id_mapping(self) -> str: # Restored
        """Generate candidate ID mapping for voting prompts.""" # Restored
        lines = [] # Restored
        for agent in self.get_visible_candidates(): # Restored
            cardinal_id = getattr(agent, 'cardinal_id', getattr(agent, 'agent_id', '?')) # Restored
            name = agent.name # Restored
            lines.append(f"Cardinal {cardinal_id}: {name}") # Restored
        return "\\n".join(lines) # Restored

    def get_visible_candidates(self) -> list:
        """Get list of visible candidate agent objects."""
        return [self.env.agents[aid] for aid in self.get_visible_candidates_ids() if aid < len(self.env.agents)]

    def generate_recent_speech_snippets(self, agent_id: int) -> str:
        """Extract last 2-3 speech snippets (~30 words each) for memory cue."""
        agent_speeches = []
        for round_comments in self.env.discussionHistory:
            for comment in round_comments:
                if comment['agent_id'] == agent_id:
                    words = comment['message'].split()
                    snippet = ' '.join(words[:30]) + ("..." if len(words) > 30 else "")
                    agent_speeches.append(snippet)
        if not agent_speeches:
            return "No previous speeches"
        return " | ".join(agent_speeches[-3:])
