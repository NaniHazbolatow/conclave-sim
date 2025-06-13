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
    
    def generate_agent_variables(self, agent_id: int) -> Dict[str, Any]:
        """Generate all agent-specific variables for prompts."""
        agent = self.env.agents[agent_id]
        variables = {
            'agent_name': agent.name,
            'cardinal_id': getattr(agent, 'cardinal_id', agent_id),
            'role_tag': getattr(agent, 'role_tag', 'ELECTOR'),
            'biography': agent.background,
            'persona_internal': getattr(agent, 'persona_internal', ''),
            'profile_public': getattr(agent, 'profile_public', ''),
            'discussion_round': self.env.discussionRound,
            'voting_round': self.env.votingRound,
            'threshold': self.env._calculate_voting_threshold(),
            'compact_scoreboard': self.generate_compact_scoreboard(),
            'visible_candidates': self.generate_visible_candidates(),
            'visible_candidates_ids': self.generate_visible_candidate_ids(),
            'candidate_id_mapping': self.generate_candidate_id_mapping(),
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

    def generate_visible_candidates(self) -> str:
        """Generate visible_candidates variable with names and external personas."""
        lines = []
        for agent in self.get_visible_candidates():
            cardinal_id = getattr(agent, 'cardinal_id', getattr(agent, 'agent_id', '?'))
            name = agent.name
            profile = getattr(agent, 'profile_public', 'Profile not available')
            lines.append(f"Cardinal {cardinal_id} ({name}): {profile}")
        return "\n".join(lines)

    def generate_stance_digest(self, agent_id: int) -> str:
        """Generate stance_digest variable as defined in glossary."""
        stance = getattr(self.env.agents[agent_id], 'internal_stance', None)
        return stance if stance else "No current stance"

    def generate_group_profiles(self, participant_ids: List[int]) -> str:
        """Generate group_profiles variable as defined in glossary."""
        profiles = []
        for pid in participant_ids:
            if pid >= len(self.env.agents):
                continue
            agent = self.env.agents[pid]
            name = agent.name
            profile_public = getattr(agent, 'profile_public', 'Profile not available')
            relation = self.generate_relation_for_participant(pid, participant_ids)
            profiles.append(f"{name} – {profile_public} ({relation})")
        return "\n".join(profiles)

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
            return ""
        stance = getattr(agent, 'internal_stance', None)
        if not stance:
            return "Analyzing current situation and candidate options."
        words = stance.split()
        reflection = " ".join(words[:25]) if len(words) > 25 else stance
        return reflection.replace('\n', ' ').replace('"', '').replace("'", '').strip()

    def generate_visible_candidate_ids(self) -> str:
        """Generate list of Cardinal IDs for voting (JSON format)."""
        return json.dumps(self.get_visible_candidates_cardinal_ids())

    def generate_candidate_id_mapping(self) -> str:
        """Generate candidate ID mapping for voting prompts."""
        lines = []
        for agent in self.get_visible_candidates():
            cardinal_id = getattr(agent, 'cardinal_id', getattr(agent, 'agent_id', '?'))
            name = agent.name
            lines.append(f"Cardinal {cardinal_id}: {name}")
        return "\n".join(lines)

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
