# filepath: /Users/kbverlaan/Library/Mobile Documents/com~apple~CloudDocs/Universiteit/Computational Science/ABM/conclave-sim/conclave/config/prompt_variable_generator.py
"""
Prompt Variable Generator

This module generates all the variables defined in the variable glossary
for use in the literature-grounded prompt templates.
"""

import json
import logging
import math # Add import math
from typing import Dict, List, Optional, Any
from datetime import datetime

logger = logging.getLogger(__name__)

class PromptVariableGenerator:
    """Generates all variables from the variable glossary for prompt templates."""
    
    def __init__(self, env, prompt_loader): # Added prompt_loader argument
        self.env = env
        self.logger = logging.getLogger(self.__class__.__name__)
        self.prompt_loader = prompt_loader # Use passed prompt_loader instance

    def generate_prompt(
        self,
        prompt_name: str,
        agent_id: Optional[int] = None,
        discussion_round: Optional[int] = None,
        discussion_group_ids: Optional[List[int]] = None,
        target_agent_id: Optional[int] = None,
        include_id_for_visible_candidates: bool = False,  # Added parameter with default
        **kwargs
    ) -> str:
        """
        Retrieves a prompt template by name and formats it with relevant variables.

        Args:
            prompt_name: The name of the prompt template (e.g., "stance", "discussion_candidate").
            agent_id: The ID of the agent for whom the prompt is being generated (if agent-specific).
            participant_ids: A list of agent IDs for group-specific prompts (e.g., discussion participants).
            **kwargs: Additional ad-hoc variables to be passed to the prompt.

        Returns:
            The formatted prompt string.
        
        Raises:
            ValueError: If the prompt template is not found.
        """
        template_str = self.prompt_loader.get_prompt(prompt_name)
        if template_str is None:
            self.logger.error(f"Prompt template '{prompt_name}' not found.")
            raise ValueError(f"Prompt template '{prompt_name}' not found.")

        variables = {}
        if agent_id is not None:
            # Pass include_id_for_visible_candidates to generate_agent_variables
            variables.update(self.generate_agent_variables(agent_id, prompt_name, include_id_for_visible_candidates)) # Pass prompt_name
        
        # Calculate votes needed for supermajority
        num_total_agents = self.env.num_agents
        supermajority_fraction = self.env.supermajority_threshold
        # Use math.ceil() for closest integer, ensuring it reflects votes needed to win
        votes_needed = math.ceil(supermajority_fraction * num_total_agents) 
        variables['threshold'] = int(votes_needed)

        if discussion_group_ids is not None:
            variables.update(self.generate_group_variables(discussion_group_ids))
            # Add specific variables for discussion prompts if needed
            if prompt_name in ["discussion_candidate", "discussion_elector"] and discussion_round is not None:
                variables['discussion_round'] = discussion_round # Use the passed discussion_round

        variables.update(kwargs) # Add any ad-hoc variables

        try:
            # Ensure all required variables are present or handle missing ones gracefully
            # For now, we rely on Python's string formatting to raise KeyError if a variable is missing.
            # A more robust solution might involve checking for all expected keys in the template.
            formatted_prompt = template_str.format(**variables)
            return formatted_prompt
        except KeyError as e:
            self.logger.error(f"Missing variable {e} in prompt template '{prompt_name}'. Available variables: {variables.keys()}")
            # Fallback or re-raise, depending on desired behavior
            # For now, returning a message indicating the error
            return f"[ERROR: Missing variable {e} for prompt '{prompt_name}']"
        except Exception as e:
            self.logger.error(f"Error formatting prompt '{prompt_name}': {e}", exc_info=True)
            return f"[ERROR: Could not format prompt '{prompt_name}']"

    def generate_agent_variables(self, agent_id: int, prompt_name: str, include_id_for_visible_candidates: bool = False) -> Dict[str, Any]: # Added parameter
        """Generate variables specific to an agent."""
        agent = self.env.agents[agent_id]
        
        # Determine role_tag
        role_tag = "elector"
        if hasattr(self.env, 'candidate_ids') and agent_id in self.env.candidate_ids:
            role_tag = "candidate"
        elif hasattr(self.env, 'current_candidates') and agent_id in self.env.current_candidates: # Fallback for different candidate tracking
            role_tag = "candidate"

        # Determine if the agent itself is a candidate for certain prompts
        agent_is_candidate_for_prompt = False
        if prompt_name in ["discussion_candidate", "vote_candidate", "stance_candidate"]: # Add other relevant prompt names
            if hasattr(self.env, 'candidate_ids') and agent_id in self.env.candidate_ids:
                agent_is_candidate_for_prompt = True
            elif hasattr(self.env, 'current_candidates') and agent_id in self.env.current_candidates:
                agent_is_candidate_for_prompt = True
        
        # For 'visible_candidates', decide whether to include Cardinal ID based on the prompt type or agent role
        # For stance generation, always include Cardinal ID. For other prompts, use the passed flag.
        include_cardinal_id_for_visible = True if prompt_name.startswith("stance") else include_id_for_visible_candidates


        variables = {
            'agent_name': agent.name,
            'agent_id': agent.agent_id,
            'cardinal_id': getattr(agent, 'cardinal_id', 'N/A'), # Added cardinal_id
            'biography': agent.background_csv if hasattr(agent, 'background_csv') else "No background available",
            'internal_persona': agent.internal_persona_csv if hasattr(agent, 'internal_persona_csv') else "No internal persona available", # Retained for compatibility if used elsewhere
            'persona_internal': agent.internal_persona_csv if hasattr(agent, 'internal_persona_csv') else "No internal persona available", # Added persona_internal
            'public_profile': agent.public_profile_csv if hasattr(agent, 'public_profile_csv') else "No public profile available",
            'persona_tag': agent.persona_tag_csv if hasattr(agent, 'persona_tag_csv') else "No persona tag available", # Added persona_tag
            'current_round': self.env.votingRound, # Use votingRound as it's incremented
            'voting_round': self.env.votingRound, # Add voting_round explicitly
            'max_rounds': self.env.max_election_rounds,
            'visible_candidates': self.generate_visible_candidates(include_cardinal_id=include_cardinal_id_for_visible), # Use determined flag
            'visible_candidates_ids': self.get_visible_candidates_cardinal_ids(), # Changed to get_visible_candidates_cardinal_ids
            'candidate_list': self.generate_visible_candidates(include_cardinal_id=include_cardinal_id_for_visible), # Changed to generate_visible_candidates and use flag
            'stance_digest': self.generate_stance_digest(agent_id),
            'reflection_digest': self.generate_reflection_digest(agent_id),
            'role_tag': role_tag,  # Added role_tag
            'compact_scoreboard': self.generate_compact_scoreboard(), # Added compact_scoreboard
            'current_round_number': self.env.votingRound, # Added current_round_number
            'total_rounds': self.env.max_election_rounds, # Added total_rounds
            'candidate_id_mapping': self._generate_candidate_id_mapping(), # Added candidate_id_mapping
            'viable_papabili': self.generate_viable_papabili(include_cardinal_id=True), # Always include cardinal_id for viable_papabili
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
        """Generate visible_candidates variable with names, (optional) Cardinal IDs, public profiles, votes, and momentum."""
        lines = []
        visible_candidate_agents = self.get_visible_candidates() # List of agent objects

        if not self.env.votingHistory:
            # No votes cast yet, so don't show vote count or momentum
            for agent in visible_candidate_agents:
                name = agent.name
                cardinal_id_str = f"Cardinal {getattr(agent, 'cardinal_id', 'ID N/A')} - " if include_cardinal_id else ""
                public_profile = getattr(agent, 'public_profile_csv', 'Public Profile N/A') # Use public_profile_csv
                lines.append(f"{cardinal_id_str}{name} – {public_profile}")
            return "\n".join(lines) # Use actual newline

        current_votes_map = self.env.votingHistory[-1]
        previous_votes_map = self.env.votingHistory[-2] if len(self.env.votingHistory) > 1 else {}

        for agent in visible_candidate_agents:
            name = agent.name
            agent_id_key = agent.agent_id
            cardinal_id_str = f"Cardinal {getattr(agent, 'cardinal_id', 'ID N/A')} - " if include_cardinal_id else ""
            public_profile = getattr(agent, 'public_profile_csv', 'Public Profile N/A') # Use public_profile_csv
            
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

            lines.append(f"{cardinal_id_str}{name} – {public_profile}, {current_agent_votes} votes ({momentum_status})")
        return "\n".join(lines) # Use actual newline

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
            persona_tag = getattr(agent, 'persona_tag_csv', 'Tag N/A') # Changed to persona_tag_csv
            
            # Changed format to be more explicit about the tag and removed "()"
            profiles.append(f"• {name} – Tag: {persona_tag}") 

        if not profiles:
            self.logger.info("generate_group_profiles found no valid participants from the provided IDs. Returning default message.")
            return "No valid participants found from the provided IDs for this discussion."
            
        return "\n".join(profiles) # Use actual newline

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
        if self.env.discussionRound == 0: # Changed from self.env.votingRound == 0
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

    def get_visible_candidates(self) -> List[Any]: # Changed from List[Agent] to List[Any] for flexibility
        """
        Get the list of visible candidate agent objects for the current voting round.
        Returns top 5 candidates if votes have been cast, otherwise all designated candidates.
        """
        if not self.env.votingHistory: # No votes cast yet
            # Return all designated candidates (as agent objects)
            candidate_agent_ids = self.env.get_candidates_list() # This returns agent_ids (indices)
            return [self.env.agents[aid] for aid in candidate_agent_ids if aid < len(self.env.agents)]

        # Votes have been cast, determine top 5
        current_votes = self.env.votingHistory[-1] # {candidate_agent_id: num_votes}
        
        # Sort by votes, then by original candidate order (e.g., by agent_id) for tie-breaking
        # This ensures somewhat stable ordering if votes are tied.
        # We need to ensure that only actual candidates are considered here.
        
        designated_candidate_agent_ids = self.env.get_candidates_list()
        
        # Filter current_votes to only include designated candidates and get their vote counts
        candidate_votes = {
            agent_id: current_votes.get(agent_id, 0) 
            for agent_id in designated_candidate_agent_ids
        }

        # Sort these designated candidates by their votes (descending), then by agent_id (ascending) for tie-breaking
        sorted_candidate_agent_ids = sorted(
            candidate_votes.keys(),
            key=lambda aid: (candidate_votes[aid], -aid), # Sort by votes (desc), then -agent_id (asc for agent_id)
            reverse=True
        )
        
        # Take top 5
        top_5_agent_ids = sorted_candidate_agent_ids[:5]
        
        return [self.env.agents[aid] for aid in top_5_agent_ids if aid < len(self.env.agents)]

    def _generate_candidate_id_mapping(self) -> str:
        """
        Generates a multi-line string listing visible candidates
        with their Cardinal ID (from CSV) and full name.
        This is used for the 'Available candidates' section in the prompt.
        """
        visible_candidate_agents = self.get_visible_candidates() # Returns list of agent objects
        if not visible_candidate_agents:
            return "No candidates currently available for voting."
        
        lines = []
        for agent in visible_candidate_agents:
            name = getattr(agent, 'name', 'Unknown Name') # Agent's full name
            cardinal_id = getattr(agent, 'cardinal_id', 'N/A') # Cardinal ID from CSV
            
            lines.append(f"Cardinal {cardinal_id} - {name}")
            
        return "\\n".join(lines)

    def generate_viable_papabili(self, include_cardinal_id: bool) -> str:
        """
        Generate a list of viable papabili (candidates) based on current voting round.
        Format: "Cardinal [Cardinal ID] - [Name] – [Profile Blurb], [Votes] votes ([Momentum])"
        or "Cardinal [Cardinal ID] - [Name] – [Profile Blurb]" if no votes yet.
        Always includes Cardinal ID.
        """
        return self.generate_visible_candidates(include_cardinal_id=True)


    # Helper methods (potentially move to a utils module or keep private)
    def _get_agent_profile_blurb(self, agent_id: int) -> str:
        """Helper to get an agent\'s profile blurb, defaulting if not found."""
        agent = self.env.agents[agent_id]
        return getattr(agent, 'profile_blurb_csv', "No profile blurb available.")

    def _get_agent_cardinal_id(self, agent_id: int) -> str:
        """Helper to get an agent\'s cardinal_id, defaulting if not found."""
        agent = self.env.agents[agent_id]
        return getattr(agent, 'cardinal_id', "N/A")

    def _get_agent_name(self, agent_id: int) -> str:
        """Helper to get an agent\'s name."""
        return self.env.agents[agent_id].name
