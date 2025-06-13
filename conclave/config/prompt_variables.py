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
            # Basic agent information
            'agent_name': agent.name,
            'cardinal_id': getattr(agent, 'cardinal_id', agent_id),
            'role_tag': getattr(agent, 'role_tag', 'ELECTOR'),
            
            # Persona information
            'biography': agent.background,
            'persona_internal': getattr(agent, 'persona_internal', ''),
            'profile_public': getattr(agent, 'profile_public', ''),
            
            # Round tracking
            'discussion_round': self.env.discussionRound,
            'voting_round': self.env.votingRound,
            
            # Election mechanics
            'threshold': self.env._calculate_voting_threshold(),
            
            # Dynamic variables (generated fresh each time)
            'compact_scoreboard': self.generate_compact_scoreboard(),
            'visible_candidates': self.generate_visible_candidates(),
            'visible_candidates_ids': self.generate_visible_candidate_ids(),
            'candidate_id_mapping': self.generate_candidate_id_mapping(),
            'stance_digest': self.generate_stance_digest(agent_id),
            'reflection_digest': self.generate_reflection_digest(agent_id),
        }
        
        return variables
    
    def generate_group_variables(self, participant_ids: List[int]) -> Dict[str, Any]:
        """Generate group-specific variables for discussion prompts."""
        variables = {
            'group_profiles': self.generate_group_profiles(participant_ids),
        }
        
        return variables
    
    def get_visible_candidates_ids(self) -> List[int]:
        """Get the list of visible candidate IDs for the current voting round."""
        if self.env.votingRound == 1:
            # Round 1 = use designated candidates only (not all agents)
            if self.env.testing_groups_enabled:
                return [agent_id for agent_id in self.env.candidate_ids if agent_id < len(self.env.agents)]
            else:
                # In normal mode, use the actual candidates list instead of all agents
                return self.env.get_candidates_list()
        else:
            # Subsequent rounds = top-N (e.g. 5) by votes, filtered to valid candidates
            if not self.env.votingHistory:
                if self.env.testing_groups_enabled:
                    return [agent_id for agent_id in self.env.candidate_ids if agent_id < len(self.env.agents)]
                else:
                    # In normal mode, use the actual candidates list instead of all agents
                    return self.env.get_candidates_list()
            else:
                current_votes = self.env.votingHistory[-1]
                sorted_candidates = sorted(current_votes.items(), key=lambda x: x[1], reverse=True)
                top_candidates = sorted_candidates[:5]
                # Filter to only include valid candidates
                valid_candidates = self.env.get_candidates_list()
                return [candidate_id for candidate_id, _ in top_candidates 
                       if candidate_id < len(self.env.agents) and candidate_id in valid_candidates]

    def get_visible_candidates_cardinal_ids(self) -> List[int]:
        """Get the list of visible candidate Cardinal IDs for the current voting round."""
        visible_agent_ids = self.get_visible_candidates_ids()
        cardinal_ids = []
        
        for agent_id in visible_agent_ids:
            if agent_id < len(self.env.agents):
                cardinal_id = getattr(self.env.agents[agent_id], 'cardinal_id', agent_id)
                cardinal_ids.append(cardinal_id)
        
        return cardinal_ids

    def generate_compact_scoreboard(self) -> str:
        """Generate compact_scoreboard variable as defined in glossary."""
        if not self.env.votingHistory:
            return "No votes cast yet"
        
        current_votes = self.env.votingHistory[-1]
        previous_votes = self.env.votingHistory[-2] if len(self.env.votingHistory) > 1 else {}
        
        # Calculate momentum tags
        scoreboard_items = []
        
        # Sort by vote count descending
        sorted_candidates = sorted(current_votes.items(), key=lambda x: x[1], reverse=True)
        
        for candidate_id, votes in sorted_candidates:
            if candidate_id >= len(self.env.agents):
                continue
                
            candidate_name = self.env.agents[candidate_id].name
            
            # Only show momentum tags if there's a previous round to compare with
            if len(self.env.votingHistory) > 1:
                previous_count = previous_votes.get(candidate_id, 0)
                delta = votes - previous_count
                
                # Apply tag algorithm: More sensitive thresholds for better momentum detection
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
                # First round - no momentum tags
                scoreboard_items.append(f"{candidate_name}:{votes}")
        
        return "; ".join(scoreboard_items)
    
    def generate_visible_candidates(self) -> str:
        """Generate visible_candidates variable with names and external personas."""
        candidate_info = []
        
        # Get visible candidate IDs using the new helper function
        visible_candidate_ids = self.get_visible_candidates_ids()
        candidate_agents = [self.env.agents[agent_id] for agent_id in visible_candidate_ids]
        
        for agent in candidate_agents:
            cardinal_id = getattr(agent, 'cardinal_id', agent.agent_id)
            name = agent.name
            profile = getattr(agent, 'profile_public', 'Profile not available')
            candidate_info.append(f"Cardinal {cardinal_id} ({name}): {profile}")
        
        return "\n".join(candidate_info)
    
    def generate_stance_digest(self, agent_id: int) -> str:
        """Generate stance_digest variable as defined in glossary."""
        agent = self.env.agents[agent_id]
        
        # Get current stance without triggering generation to avoid circular calls
        stance = agent.internal_stance
        if not stance:
            return "No current stance"
        
        return stance
    
    def generate_group_profiles(self, participant_ids: List[int]) -> str:
        """Generate group_profiles variable as defined in glossary."""
        if len(participant_ids) != 5:
            self.logger.warning(f"Expected 5 participants, got {len(participant_ids)}")
        
        profiles = []
        for participant_id in participant_ids:
            if participant_id >= len(self.env.agents):
                continue
                
            agent = self.env.agents[participant_id]
            name = agent.name
            profile_public = getattr(agent, 'profile_public', 'Profile not available')
            
            # Generate relation (for now, using placeholder)
            relation = self.generate_relation_for_participant(participant_id, participant_ids)
            
            profile_line = f"{name} – {profile_public} ({relation})"
            profiles.append(profile_line)
        
        return "\n".join(profiles)
    
    def generate_participant_relation_list(self, participant_ids: List[int]) -> str:
        """Generate participant_relation_list variable as defined in glossary."""
        relations = []
        
        for participant_id in participant_ids:
            if participant_id >= len(self.env.agents):
                continue
                
            agent = self.env.agents[participant_id]
            relation = self.generate_relation_for_participant(participant_id, participant_ids)
            relations.append(f"{agent.name.split()[-1]}-{relation}")  # Use last name
        
        return "; ".join(relations)
    
    def generate_relation_for_participant(self, participant_id: int, all_participants: List[int]) -> str:
        """Generate relation label for a participant (sympathetic, neutral, opposed)."""
        # For now, using a simple heuristic based on voting history
        # In a full implementation, this could use embeddings or more sophisticated analysis
        
        if not self.env.votingHistory:
            return "neutral"
        
        # Simple heuristic: check if they voted for similar candidates
        current_votes = self.env.votingHistory[-1]
        
        # Get who this participant voted for
        participant_vote = None
        for candidate_id, voters in current_votes.items():
            # This is simplified - in reality we'd need to track individual votes
            pass
        
        # For now, return random distribution
        import random
        return random.choice(["sympathetic", "neutral", "opposed"])
    
    def generate_group_transcript(self, discussion_round: int) -> str:
        """Generate group_transcript variable as defined in glossary."""
        if discussion_round <= len(self.env.discussionHistory):
            round_comments = self.env.discussionHistory[discussion_round - 1]
            
            transcript_lines = []
            for comment in round_comments:
                speaker_name = self.env.agents[comment['agent_id']].name
                message = comment['message']
                transcript_lines.append(f"{speaker_name}: {message}")
            
            return "\n".join(transcript_lines)
        
        return "No transcript available"
    
    def generate_group_summary(self, discussion_round: int) -> str:
        """Generate group_summary variable as JSON object as defined in glossary."""
        if discussion_round <= len(self.env.discussionHistory):
            round_comments = self.env.discussionHistory[discussion_round - 1]
            
            # Extract key points (simplified)
            key_points = []
            speakers = []
            
            for comment in round_comments:
                speaker_name = self.env.agents[comment['agent_id']].name
                message = comment['message']
                
                speakers.append(speaker_name)
                
                # Extract key point (first sentence, truncated to ≤25 words)
                sentences = message.split('.')
                if sentences:
                    first_sentence = sentences[0].strip()
                    words = first_sentence.split()
                    if len(words) > 25:
                        first_sentence = " ".join(words[:25]) + "..."
                    key_points.append(first_sentence)
            
            # Determine overall tone (simplified)
            overall_tone = "mixed"  # Could be "harmonious", "mixed", or "tense"
            
            # Ensure exactly 3 items in lists (pad with "N/A" if necessary)
            while len(key_points) < 3:
                key_points.append("N/A")
            while len(speakers) < 3:
                speakers.append("N/A")
            
            key_points = key_points[:3]
            speakers = speakers[:3]
            
            summary = {
                "key_points": key_points,
                "speakers": speakers,
                "overall_tone": overall_tone
            }
            
            return json.dumps(summary)
        
        return json.dumps({"key_points": ["N/A", "N/A", "N/A"], "speakers": ["N/A", "N/A", "N/A"], "overall_tone": "mixed"})
    
    def generate_agent_last_utterance(self, agent_id: int) -> str:
        """Generate agent_last_utterance variable as defined in glossary."""
        # Find the most recent speech by this agent
        for round_comments in reversed(self.env.discussionHistory):
            for comment in round_comments:
                if comment['agent_id'] == agent_id:
                    return comment['message']
        
        return "No previous utterance"
    
    def generate_reflection_digest(self, agent_id: int) -> str:
        """Generate reflection_digest variable as defined in glossary."""
        agent = self.env.agents[agent_id]
        
        # If this is initial stance generation (no discussions yet), return empty
        if self.env.discussionRound == 0:
            return ""
        
        # This would typically be generated by the DISCUSSION_REFLECTION prompt
        # For now, create a placeholder based on current stance
        stance = getattr(agent, 'internal_stance', None)
        if not stance:
            return "Analyzing current situation and candidate options."
        
        # Truncate to ≤25 words, single sentence, no bullet
        words = stance.split()
        if len(words) > 25:
            reflection = " ".join(words[:25])
        else:
            reflection = stance
        
        # Ensure it's a single sentence (no line breaks or quotes)
        reflection = reflection.replace('\n', ' ').replace('"', '').replace("'", '').strip()
        
        return reflection
    
    def test_variable_generation(self, agent_id: int = 0):
        """Test the generation of all variables for debugging."""
        self.logger.info(f"=== TESTING VARIABLE GENERATION FOR AGENT {agent_id} ===")
        
        # Generate agent variables
        agent_vars = self.generate_agent_variables(agent_id)
        
        for var_name, value in agent_vars.items():
            self.logger.info(f"{var_name}: {value}")
            print(f"{var_name}: {value}")
        
        # Test group variables with a sample group
        sample_group = list(range(min(5, len(self.env.agents))))
        group_vars = self.generate_group_variables(sample_group)
        
        for var_name, value in group_vars.items():
            self.logger.info(f"{var_name}: {value}")
            print(f"{var_name}: {value}")
        
        self.logger.info("=== END VARIABLE GENERATION TEST ===")
    
    def generate_visible_candidate_ids(self) -> str:
        """Generate list of Cardinal IDs for voting (JSON format)."""
        cardinal_ids = self.get_visible_candidates_cardinal_ids()
        return json.dumps(cardinal_ids)
