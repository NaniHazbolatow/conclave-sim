import logging
import random
import threading
import logging
import math
import pandas as pd
from concurrent.futures import ThreadPoolExecutor
from tqdm import tqdm
from typing import Dict, List, Optional
from ..config.manager import get_config

logger = logging.getLogger(__name__)

class ConclaveEnv:
    def __init__(self, num_agents: int = 3):
        self.num_agents = num_agents
        self.agents = []
        self.votingRound = 0
        self.votingHistory = []
        self.votingBuffer = {}
        self.voting_lock = threading.Lock()
        self.winner = None
        self.discussionHistory = []
        self.discussionRound = 0
        # Track which agents participated in which discussion rounds
        self.agent_discussion_participation = {}
        # Track which agents voted in each round
        self.voting_participation = {}
        # Track which agents have already spoken in the current election round
        self.agents_spoken_in_current_election = set()
        
        # Get configuration for simulation parameters
        self.config = get_config()
        self.simulation_config = self.config.get_simulation_config()
        self.max_speakers_per_round = self.simulation_config.get("max_speakers_per_round", 5)
        
        # Apply testing group overrides if enabled
        self.config.apply_testing_group_overrides()
        
        # Get voting configuration (after potential overrides)
        voting_config = self.simulation_config.get("voting", {})
        self.supermajority_threshold = self.config.get_supermajority_threshold()
        
        # Testing groups support
        self.testing_groups_enabled = self.config.is_testing_groups_enabled()
        self.candidate_ids = []
        if self.testing_groups_enabled:
            self.candidate_ids = self.config.get_testing_group_candidate_ids()
            logger.info(f"Testing groups enabled. Candidates: {self.candidate_ids}")
    
    def freeze_agent_count(self):
        """Freeze the agent count to match the loaded roster. Call after loading all agents."""
        self.num_agents = len(self.agents)

    # Testing groups candidate/elector helper methods
    def is_candidate(self, agent_id: int) -> bool:
        """Check if an agent is a candidate in testing groups mode."""
        if not self.testing_groups_enabled:
            return True  # In normal mode, all agents can be candidates
        return agent_id in self.candidate_ids

    def get_candidates_list(self) -> List[int]:
        """Get list of candidate agent IDs."""
        if not self.testing_groups_enabled:
            return list(range(self.num_agents))  # All agents are candidates in normal mode
        return self.candidate_ids.copy()

    def get_candidates_description(self) -> str:
        """Get a formatted description of candidates for prompts."""
        candidates = self.get_candidates_list()
        if not candidates:
            return "No candidates designated."
        
        candidate_names = []
        for agent_id in candidates:
            if agent_id < len(self.agents):
                agent = self.agents[agent_id]
                cardinal_id = getattr(agent, 'cardinal_id', agent_id)
                candidate_names.append(f"Cardinal {cardinal_id} - {agent.name} (Agent ID: {agent_id})")
            else:
                candidate_names.append(f"Agent {agent_id} - (Not loaded)")
        
        return "Designated candidates:\n" + "\n".join(candidate_names)

    def is_valid_vote_candidate(self, candidate_id: int) -> bool:
        """Check if a candidate ID is valid for voting in current mode."""
        # Basic range check
        if not (0 <= candidate_id < self.num_agents):
            return False
        
        # In testing groups mode, only designated candidates can receive votes
        if self.testing_groups_enabled:
            return candidate_id in self.candidate_ids
        
        return True

    def get_role_description(self, agent_id: int) -> str:
        """Get the role description for an agent (candidate or elector)."""
        if self.is_candidate(agent_id):
            return "candidate"
        else:
            return "elector"

    def load_agents_from_config(self) -> None:
        """Load agents based on testing groups configuration or all agents in normal mode."""
        # Import here to avoid circular imports
        from ..agents.base import Agent
        
        # Read cardinals from CSV file
        cardinals_df = pd.read_csv('data/cardinal_electors_2025.csv')
        
        # Read persona data from separate CSV file
        personas_df = pd.read_csv('data/cardinal_electors_personas.csv')
        
        # Merge the dataframes on Cardinal_ID to get both background and persona data
        merged_df = cardinals_df.merge(personas_df, on='Cardinal_ID', how='left', suffixes=('', '_persona'))
        
        if self.testing_groups_enabled:
            # Get the cardinal IDs for the current testing group
            cardinal_ids = self.config.get_testing_group_cardinal_ids()
            logger.info(f"Loading testing group with {len(cardinal_ids)} cardinals")
            
            # Filter dataframe to only include cardinals in the testing group
            merged_df = merged_df[merged_df['Cardinal_ID'].isin(cardinal_ids)]
            
            if len(merged_df) != len(cardinal_ids):
                loaded_ids = set(merged_df['Cardinal_ID'].tolist())
                missing_ids = set(cardinal_ids) - loaded_ids
                logger.warning(f"Some cardinal IDs not found in CSV: {missing_ids}")
        else:
            # In normal mode, optionally limit number of cardinals based on config
            num_cardinals = self.config.get_num_cardinals()
            if num_cardinals and num_cardinals < len(merged_df):
                merged_df = merged_df.head(num_cardinals)
                logger.info(f"Loading first {num_cardinals} cardinals from CSV")
        
        # Create Agent instances and add them to env.agents
        for list_index, (idx, row) in enumerate(merged_df.iterrows()):
            agent = Agent(
                agent_id=list_index,  # Use list index as agent_id for consistent indexing
                name=row['Name'],
                background=row['Background'],
                env=self
            )
            # Store the original Cardinal_ID for reference
            agent.cardinal_id = row['Cardinal_ID']
            
            # Store persona data
            agent.persona_internal = row.get('Internal_Persona', '')
            agent.profile_public = row.get('External_Persona', '')
            
            self.agents.append(agent)
        
        # Update candidate_ids to use list indices instead of Cardinal_IDs
        if self.testing_groups_enabled:
            original_candidate_ids = self.config.get_testing_group_candidate_ids()
            # Map Cardinal_IDs to list indices
            cardinal_id_to_index = {row['Cardinal_ID']: idx for idx, (_, row) in enumerate(merged_df.iterrows())}
            self.candidate_ids = [cardinal_id_to_index[cid] for cid in original_candidate_ids if cid in cardinal_id_to_index]
            logger.info(f"Mapped candidate Cardinal_IDs {original_candidate_ids} to list indices {self.candidate_ids}")
        
        # Freeze agent count after loading
        self.freeze_agent_count()
        
        # Set up the prompt manager with the environment after agents are loaded
        from ..config.prompts import get_prompt_manager
        prompt_manager = get_prompt_manager()
        prompt_manager.set_environment(self)
        logger.info("Prompt manager environment updated with ConclaveEnv instance")
        
        logger.info(f"Loaded {self.num_agents} agents")
        if self.testing_groups_enabled:
            logger.info(f"Testing group candidates: {self.candidate_ids}")
            logger.info(f"Testing group setup:\n{self.get_candidates_description()}")
        
        logger.info(f"\n{self.list_candidates_for_prompt(randomize=False)}")
    
    def reset_discussion_speakers_for_new_election_round(self):
        """Reset the set of agents who have spoken in the current election round."""
        self.agents_spoken_in_current_election = set()
        logger.info("Reset discussion speakers for new election round")

    def cast_vote(self, candidate_id: int, voter_id: int = None) -> None:
        with self.voting_lock:
            # Debug: Log vote before adding
            old_count = self.votingBuffer.get(candidate_id, 0)
            self.votingBuffer[candidate_id] = self.votingBuffer.get(candidate_id, 0) + 1
            new_count = self.votingBuffer[candidate_id]
            
            logger.debug(f"Vote aggregation: Candidate {candidate_id} votes: {old_count} -> {new_count}")
            logger.debug(f"Current voting buffer: {dict(self.votingBuffer)}")
            
            # Track which agents have voted this round
            if voter_id is not None:
                if self.votingRound not in self.voting_participation:
                    self.voting_participation[self.votingRound] = set()
                self.voting_participation[self.votingRound].add(voter_id)
                logger.debug(f"Agent {voter_id} recorded as voting in round {self.votingRound}")

    def _safe_cast(self, agent) -> bool:
        """
        Safely execute an agent's cast_vote method with proper exception handling and retry logic.
        
        Args:
            agent: The agent instance to cast vote for
            
        Returns:
            bool: True if vote was cast successfully, False if failed after all retries
        """
        import time
        
        max_r = self.config.get_llm_max_retries()
        back = self.config.get_llm_backoff()
        
        for attempt in range(1, max_r + 1):
            try:
                agent.cast_vote()
                logger.debug(f"✅ {agent.name} successfully cast vote (attempt {attempt})")
                return True
            except Exception as e:
                logger.warning(f"Vote fail {attempt}/{max_r} for {agent.name}: {e}")
                if attempt == max_r:
                    logger.error(f"❌ giving up on {agent.name}")
                    return False
                time.sleep(back * attempt)  # simple exponential back-off
        
        return False

    def run_voting_round(self) -> bool:
        self.votingBuffer.clear()
        
        # Increment voting round BEFORE launching vote threads to prevent race conditions
        self.votingRound += 1
        
        # Initialize voting participation tracking for the current round
        self.voting_participation[self.votingRound] = set()
        
        with ThreadPoolExecutor(max_workers=min(8, self.num_agents)) as executor:
            futures = []
            
            # Submit voting tasks using safe casting
            for agent in self.agents:
                futures.append(executor.submit(self._safe_cast, agent))
            
            # Wait for all futures to complete and log results
            vote_results = []
            success_count = 0
            failed_count = 0
            
            for future in tqdm(futures, desc="Collecting Votes", total=len(futures), disable=True):
                result = future.result()  # This blocks until the task completes (True/False)
                if result:
                    success_count += 1
                    vote_results.append("SUCCESS")
                else:
                    failed_count += 1
                    vote_results.append("FAILED")
            
            # Log summary of vote execution
            max_r = self.config.get_llm_max_retries()
            logger.info(f"Vote execution summary: {success_count} successful, "
                       f"{failed_count} permanent failures "
                       f"(after {max_r} retry attempts each)")
        self.votingHistory.append(self.votingBuffer.copy())
        
        # SANITY CHECK: Ensure vote count matches agent count
        total_votes = sum(self.votingBuffer.values())
        logger.info(f"=== VOTING SANITY CHECK (Round {self.votingRound}) ===")
        logger.info(f"Total votes cast: {total_votes}")
        logger.info(f"Expected votes (num_agents): {self.num_agents}")
        logger.info(f"Vote count matches: {'✅ YES' if total_votes == self.num_agents else '❌ NO'}")
        
        # Handle vote count mismatches
        if total_votes != self.num_agents:
            logger.error(f"VOTE COUNT MISMATCH: Expected {self.num_agents}, got {total_votes}")
            
            if total_votes < self.num_agents:
                missing_votes = self.num_agents - total_votes
                logger.warning(f"Missing {missing_votes} vote(s)")
                
                # Identify which agents didn't vote
                voted_agents = self.voting_participation.get(self.votingRound, set())
                all_agent_ids = set(range(self.num_agents))
                non_voting_agents = all_agent_ids - voted_agents
                
                if non_voting_agents:
                    non_voting_names = [f"Cardinal {agent_id} - {self.agents[agent_id].name}" for agent_id in non_voting_agents]
                    logger.error(f"Agents who didn't vote: {', '.join(non_voting_names)}")
                    # No forced votes - let the round end with fewer votes
            
            elif total_votes > self.num_agents:
                extra_votes = total_votes - self.num_agents
                logger.critical(f"CRITICAL: {extra_votes} extra vote(s) detected!")
                logger.critical("This indicates a serious bug in vote counting or thread safety")
        else:
            logger.info("✅ Vote count sanity check PASSED")
        
        logger.info(f"=== END SANITY CHECK ===")
        
        # Debug: Log the voting buffer before processing results
        logger.debug(f"=== VOTE AGGREGATION DEBUG ===")
        logger.debug(f"Final voting buffer: {dict(self.votingBuffer)}")
        logger.debug(f"Buffer copy added to history: {self.votingBuffer.copy()}")
        
        voting_results = sorted(self.votingBuffer.items(), key=lambda x: x[1], reverse=True)
        logger.debug(f"Sorted voting results: {voting_results}")
        
        voting_results_str = "\n".join([f"Cardinal {getattr(self.agents[i], 'cardinal_id', i)} - {self.agents[i].name}: {votes}" for i, votes in voting_results])
        logger.debug(f"=== END VOTE AGGREGATION DEBUG ===")
        
        logger.info(f"Voting round {self.votingRound} completed.\n{voting_results_str}")
        
        # Simplified voting summary for console
        import math
        threshold = self._calculate_voting_threshold()
        
        # Check if any votes were cast
        if not voting_results:
            print("No votes were cast in this round.")
            self.votingBuffer.clear()
            return False
            
        # Show only top 3 candidates to reduce clutter
        top_candidates = voting_results[:3]
        top_results_str = ", ".join([f"{self.agents[i].name}: {votes}" for i, votes in top_candidates])
        
        # Show all candidates who received votes
        all_results_str = ", ".join([f"{self.agents[i].name}: {votes}" for i, votes in voting_results])
        print(f"🗳️  Voting Round {self.votingRound}: {all_results_str} | Threshold: {threshold} votes")

        # if the top candidate has more than or equal to the supermajority threshold of the votes
        if voting_results[0][1] >= threshold:
            top_candidate = voting_results[0][0]
            self.winner = top_candidate
            print(f"🎉 Cardinal {self.agents[top_candidate].name} elected Pope!")
            return True

        self.votingBuffer.clear()
        return False

    def run_discussion_round(self, num_speakers: int = 5) -> None:
        """
        Run a discussion round where agents can speak about candidates or their own position.
        Ensures that each cardinal speaks only once per election round across all discussion cycles.

        Args:
            num_speakers: Number of speakers to include in the discussion.
                          If greater than available agents, all available agents will participate.
        """
        self.discussionRound += 1
        round_comments = []

        # Get all agent IDs that haven't spoken yet in this election round
        all_agent_ids = list(range(len(self.agents)))
        available_agent_ids = [agent_id for agent_id in all_agent_ids 
                              if agent_id not in self.agents_spoken_in_current_election]

        # If no agents are available (all have spoken), reset and use all agents
        if not available_agent_ids:
            logger.warning(f"All agents have already spoken in this election round. "
                          f"Resetting speakers to allow participation.")
            available_agent_ids = all_agent_ids
            self.agents_spoken_in_current_election.clear()

        # Check if we should randomize speaking order
        randomize_order = self.config.get_randomize_speaking_order()
        
        if randomize_order:
            # Select speakers randomly from available agents
            logger.info(f"Using random selection for discussion round {self.discussionRound}")
            # Randomly shuffle the available IDs
            random.shuffle(available_agent_ids)
        else:
            # Use sequential order (agent ID order) from available agents
            logger.info(f"Using sequential order for discussion round {self.discussionRound}")

        # Select the specified number of speakers from available agents
        if num_speakers > len(available_agent_ids):
            selected_agent_ids = available_agent_ids
            logger.info(f"Requested {num_speakers} speakers but only {len(available_agent_ids)} available. "
                       f"Using all available agents.")
        else:
            selected_agent_ids = available_agent_ids[:num_speakers]

        # Mark these agents as having spoken in this election round
        self.agents_spoken_in_current_election.update(selected_agent_ids)

        # Store current discussion speakers for reference during the discussion
        self._current_discussion_speakers = selected_agent_ids

        # Get the corresponding agent objects
        speakers = [self.agents[agent_id] for agent_id in selected_agent_ids]

        # Log the selection
        selected_str = "\n".join([
            f"Cardinal {agent_id} - {self.agents[agent_id].name}"
            for agent_id in selected_agent_ids
        ])
        logger.info(f"Selected speakers for discussion round {self.discussionRound}:\n{selected_str}")
        logger.info(f"Agents already spoken in this election round: {sorted(self.agents_spoken_in_current_election)}")

        logger.info(f"Starting discussion round {self.discussionRound} with {len(speakers)} speakers")

        # Collect discussions from selected speakers
        futures = []
        with ThreadPoolExecutor(max_workers=min(8, len(speakers))) as executor:
            # Submit the discuss task for each speaker
            for agent in speakers:
                futures.append(executor.submit(agent.discuss))

            # Wait for all futures to complete
            for future in tqdm(futures, desc="Collecting Discussion", total=len(futures), disable=True):
                result = future.result()  # This blocks until the task completes
                if result:
                    round_comments.append(result)

        self.discussionHistory.append(round_comments)

        # Track which agents participated in this discussion round
        participating_agent_ids = [comment['agent_id'] for comment in round_comments]
        for agent_id in participating_agent_ids:
            if agent_id not in self.agent_discussion_participation:
                self.agent_discussion_participation[agent_id] = []
            self.agent_discussion_participation[agent_id].append(self.discussionRound - 1)  # -1 because we already incremented

        # Log the discussion
        discussion_str = "\n\n".join([
            f"Cardinal {getattr(self.agents[comment['agent_id']], 'cardinal_id', comment['agent_id'])} - {self.agents[comment['agent_id']].name} "
            f"{comment['message']}"
            for comment in round_comments
        ])
        logger.info(f"Discussion round {self.discussionRound} completed.\n{discussion_str}")
        
        # Simplified discussion summary for console
        print(f"\n💬 Discussion Round {self.discussionRound} ({len(round_comments)} speakers):")
        print("─" * 80)
        for comment in round_comments:
            agent_name = self.agents[comment['agent_id']].name
            # Show full message without truncation
            message = comment['message']
            print(f"\n🔸 {agent_name}:")
            print(f"{message}")
        print("─" * 80)
        
        # Generate internal stances after discussion
        self.update_internal_stances()

    def list_candidates_for_prompt(self, randomize: bool = True) -> str:
        # For voting prompts, don't randomize and show clean ID→Name mapping
        candidates = []
        
        # In testing groups mode, only show designated candidates
        if self.testing_groups_enabled:
            for agent_id in self.candidate_ids:
                if agent_id < len(self.agents):
                    agent = self.agents[agent_id]
                    cardinal_id = getattr(agent, 'cardinal_id', agent_id)
                    candidates.append(f"{agent.agent_id}\t{agent.name} (Cardinal {cardinal_id}) (CANDIDATE)")
        else:
            # Normal mode: show all agents
            for agent in self.agents:
                cardinal_id = getattr(agent, 'cardinal_id', agent.agent_id)
                candidates.append(f"{agent.agent_id}\t{agent.name} (Cardinal {cardinal_id})")
        
        result = "\n".join(candidates)
        return result

    def get_discussion_history(self, agent_id: Optional[int] = None) -> str:
        """Return formatted discussion history for prompts.
        
        Args:
            agent_id: If provided, only return discussions this agent participated in.
                     If None, return all discussions (original behavior).
        """
        if not self.discussionHistory:
            return ""

        # If no agent_id provided, return all discussions (backward compatibility)
        if agent_id is None:
            history_str = ""
            for round_num, comments in enumerate(self.discussionHistory):
                round_str = f"Discussion Round {round_num + 1}:\n"
                for comment in comments:
                    comment_agent_id = comment['agent_id']
                    round_str += f"Cardinal {comment_agent_id} - {self.agents[comment_agent_id].name}:\n{comment['message']}\n\n"
                history_str += round_str + "\n"
            return history_str

        # Return only discussions this agent participated in
        if agent_id not in self.agent_discussion_participation:
            return ""
        
        participated_rounds = self.agent_discussion_participation[agent_id]
        if not participated_rounds:
            return ""

        history_str = ""
        for round_index in participated_rounds:
            if round_index < len(self.discussionHistory):
                comments = self.discussionHistory[round_index]
                round_str = f"Discussion Round {round_index + 1}:\n"
                for comment in comments:
                    comment_agent_id = comment['agent_id']
                    round_str += f"Cardinal {comment_agent_id} - {self.agents[comment_agent_id].name}:\n{comment['message']}\n\n"
                history_str += round_str + "\n"

        return history_str
    
    def update_internal_stances(self) -> None:
        """
        Update internal stances for all agents who should update them.
        """
        from concurrent.futures import ThreadPoolExecutor
        from tqdm import tqdm
        
        # Find agents that should update their stance
        agents_to_update = [agent for agent in self.agents if agent.should_update_stance()]
        
        if not agents_to_update:
            return
        
        logger.info(f"Updating internal stances for {len(agents_to_update)} agents...")
        
        # Update stances in parallel
        with ThreadPoolExecutor(max_workers=min(4, len(agents_to_update))) as executor:
            futures = [executor.submit(agent.generate_internal_stance) for agent in agents_to_update]
            # Wait for all futures to complete
            for future in tqdm(futures, desc="Generating Stances", total=len(futures), disable=True):
                future.result()  # This blocks until the task completes
        
        # Log completion and summary
        updated_count = sum(1 for agent in agents_to_update if agent.internal_stance)
        logger.info(f"Internal stance update completed: {updated_count}/{len(agents_to_update)} agents updated")
        
        # Log stance summary for this round
        logger.info(f"=== STANCE UPDATE SUMMARY (V{self.votingRound}.D{self.discussionRound}) ===")
        for agent in agents_to_update:
            if agent.internal_stance:
                stance_preview = agent.internal_stance[:100].replace('\n', ' ')
                logger.info(f"{agent.name}: {stance_preview}...")
        logger.info("=== END STANCE SUMMARY ===")
    
    def get_all_stances(self) -> Dict[str, str]:
        """Get all current internal stances from agents."""
        stances = {}
        for agent in self.agents:
            if agent.internal_stance:
                stances[agent.name] = agent.internal_stance
        return stances

    def generate_initial_stances(self) -> None:
        """
        Generate initial internal stances for all agents before the simulation begins.
        This should be called once after all agents are added to the environment.
        """
        logger.info(f"Generating initial internal stances for {len(self.agents)} agents...")
        
        # Generate initial stances in parallel
        with ThreadPoolExecutor(max_workers=min(4, len(self.agents))) as executor:
            futures = [executor.submit(agent.generate_internal_stance) for agent in self.agents]
            # Wait for all futures to complete
            for future in tqdm(futures, desc="Generating Initial Stances", total=len(futures), disable=True):
                future.result()  # This blocks until the task completes
        
        logger.info("Initial internal stances generated for all agents")
        
        # Log initial stance summary
        stances = self.get_all_stances()
        logger.info(f"=== INITIAL STANCES SUMMARY ({len(stances)} agents) ===")
        for name, stance in stances.items():
            stance_preview = stance[:100].replace('\n', ' ')
            logger.info(f"{name}: {stance_preview}...")
        logger.info("=== END INITIAL STANCES SUMMARY ===")
    
    def _calculate_voting_threshold(self) -> int:
        """
        Calculate voting threshold with special handling for small groups.
        
        For small groups (≤5 cardinals), use rounding to avoid requiring unanimity.
        For larger groups, use ceiling as originally intended.
        
        Returns:
            The minimum number of votes required to elect a pope
        """
        import math
        raw_threshold = self.num_agents * self.supermajority_threshold
        
        # For very small groups, use rounding instead of ceiling
        # This ensures 3 cardinals need 2 votes (67%), not 3 votes (100%)
        if self.num_agents <= 5:
            return max(1, round(raw_threshold))
        else:
            return math.ceil(raw_threshold)
    
    def get_current_discussion_participants(self) -> str:
        """
        Get information about who is participating in the current discussion round.
        
        Returns:
            Formatted string with current discussion participants
        """
        if not hasattr(self, '_current_discussion_speakers') or not self._current_discussion_speakers:
            return "No current discussion participants."
        
        participant_names = []
        for agent_id in sorted(self._current_discussion_speakers):
            if agent_id < len(self.agents):
                participant_names.append(f"Cardinal {self.agents[agent_id].name}")
        
        if len(participant_names) == 1:
            return f"You are the only speaker in this discussion round."
        else:
            return f"Current discussion participants: {', '.join(participant_names)}"
    
    def get_current_scoreboard(self) -> str:
        """
        Generate a one-line scoreboard showing the top three candidates with their vote counts.
        Makes stagnation visually obvious in discussions.
        
        Returns:
            Formatted scoreboard string like "Top three: Smith=3, Jones=2, Brown=1"
        """
        if not self.votingHistory:
            return "Top three: No votes cast yet"
        
        # Get the most recent voting results
        latest_results = self.votingHistory[-1]
        
        if not latest_results:
            return "Top three: No votes cast yet"
        
        # Sort by vote count (descending) and take top 3
        sorted_results = sorted(latest_results.items(), key=lambda x: x[1], reverse=True)
        top_three = sorted_results[:3]
        
        # Format as "Name=votes" pairs
        scoreboard_parts = []
        for candidate_id, vote_count in top_three:
            candidate_name = self.agents[candidate_id].name.split()[-1]  # Use last name only for brevity
            scoreboard_parts.append(f"{candidate_name}={vote_count}")
        
        return f"Top three: {', '.join(scoreboard_parts)}"
    
    def get_valid_candidates_for_stance(self) -> str:
        """Get a formatted list of valid candidates for internal stance generation."""
        candidates = self.get_candidates_list()
        if not candidates:
            return "All cardinals in the conclave are potential candidates."
        
        candidate_lines = []
        for agent_id in candidates:
            if agent_id < len(self.agents):
                agent = self.agents[agent_id]
                cardinal_id = getattr(agent, 'cardinal_id', agent_id)
                candidate_lines.append(f"- {agent.name} (Cardinal {cardinal_id})")
            else:
                candidate_lines.append(f"- Agent {agent_id} (Not loaded)")
        
        if self.testing_groups_enabled:
            return "Only these cardinals may receive votes:\n" + "\n".join(candidate_lines)
        else:
            return "All cardinals in the conclave:\n" + "\n".join(candidate_lines)
    
    def get_threshold(self) -> int:
        """
        Public method to get the current voting threshold.
        
        Returns:
            The minimum number of votes required to elect a pope
        """
        return self._calculate_voting_threshold()
