import logging
import random
import threading
import logging
import math
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
        
        # Get configuration for simulation parameters
        self.config = get_config()
        self.simulation_config = self.config.get_simulation_config()
        self.max_speakers_per_round = self.simulation_config.get("max_speakers_per_round", 5)
        
        # Get voting configuration
        voting_config = self.simulation_config.get("voting", {})
        self.supermajority_threshold = voting_config.get("supermajority_threshold", 0.667)

    def cast_vote(self, candidate_id: int) -> None:
        with self.voting_lock:
            self.votingBuffer[candidate_id] = self.votingBuffer.get(candidate_id, 0) + 1

    def run_voting_round(self) -> bool:
        self.votingBuffer.clear()
        with ThreadPoolExecutor(max_workers=min(8, self.num_agents)) as executor:
            futures = [executor.submit(agent.cast_vote) for agent in self.agents]
            # Wait for all futures to complete
            for future in tqdm(futures, desc="Collecting Votes", total=len(futures), disable=True):
                future.result()  # This blocks until the task completes

        self.votingRound += 1
        self.votingHistory.append(self.votingBuffer.copy())
        voting_results = sorted(self.votingBuffer.items(), key=lambda x: x[1], reverse=True)
        voting_results_str = "\n".join([f"Cardinal {i} - {self.agents[i].name}: {votes}" for i, votes in voting_results])
        logger.info(f"Voting round {self.votingRound} completed.\n{voting_results_str}")
        logger.info(f"Total votes: {sum(self.votingBuffer.values())}")
        
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
        print(f"🗳️  Voting Round {self.votingRound}: {top_results_str} | Threshold: {threshold} votes")

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

        Args:
            num_speakers: Number of speakers to include in the discussion.
                          If greater than total agents, all agents will participate.
        """
        self.discussionRound += 1
        round_comments = []

        # Get all agent IDs
        all_agent_ids = list(range(len(self.agents)))

        # Check if we should randomize speaking order
        randomize_order = self.config.get_randomize_speaking_order()
        
        if randomize_order:
            # Select speakers randomly
            logger.info(f"Using random selection for discussion round {self.discussionRound}")
            # Randomly shuffle the IDs
            random.shuffle(all_agent_ids)
        else:
            # Use sequential order (agent ID order)
            logger.info(f"Using sequential order for discussion round {self.discussionRound}")

        # Select the specified number of speakers
        if num_speakers > self.num_agents:
            selected_agent_ids = all_agent_ids
        else:
            selected_agent_ids = all_agent_ids[:num_speakers]

        # Get the corresponding agent objects
        speakers = [self.agents[agent_id] for agent_id in selected_agent_ids]

        # Log the random selection
        selected_str = "\n".join([
            f"Cardinal {agent_id} - {self.agents[agent_id].name}"
            for agent_id in selected_agent_ids
        ])
        logger.info(f"Randomly selected speakers for round {self.discussionRound}:\n{selected_str}")

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
            f"Cardinal {comment['agent_id']} - {self.agents[comment['agent_id']].name} "
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
        indices = list(range(self.num_agents))
        if randomize:
            random.shuffle(indices)
        
        candidates = []
        for display_order, actual_id in enumerate(indices):
            # Check if we can load ideological stance from CSV
            ideological_stance = None
            try:
                # Try to load the CSV to get ideological stance
                import pandas as pd
                import os
                if os.path.exists('data/cardinal_electors_2025.csv'):
                    df = pd.read_csv('data/cardinal_electors_2025.csv')
                    agent_row = df[df['Name'] == self.agents[actual_id].name]
                    if not agent_row.empty and 'Ideological_Stance' in df.columns:
                        ideological_stance = agent_row['Ideological_Stance'].iloc[0]
                        if pd.notna(ideological_stance):
                            ideological_stance = str(ideological_stance).strip()
                        else:
                            ideological_stance = None
            except Exception:
                ideological_stance = None
            
            # Use the actual agent ID, not the display order, for consistent mapping
            if ideological_stance:
                candidates.append(f"Cardinal {actual_id}: {self.agents[actual_id].name}\nIdeological Stance: {ideological_stance}\n")
            else:
                # Just show the name if no ideological stance available
                candidates.append(f"Cardinal {actual_id}: {self.agents[actual_id].name}\n")
        
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
        """
        Get all current internal stances from agents.
        
        Returns:
            Dictionary mapping agent names to their current stances
        """
        stances = {}
        for agent in self.agents:
            if agent.internal_stance:
                stances[agent.name] = agent.internal_stance
        return stances
    
    def get_stance_embeddings(self) -> Dict[str, str]:
        """
        Get all current internal stances for embedding analysis.
        
        Returns:
            Dictionary mapping agent names to their current stances (for embedding)
        """
        return self.get_all_stances()

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
