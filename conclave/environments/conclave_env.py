import logging
import random
import threading
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

    def cast_vote(self, candidate_id: int) -> None:
        with self.voting_lock:
            self.votingBuffer[candidate_id] = self.votingBuffer.get(candidate_id, 0) + 1

    def run_voting_round(self) -> bool:
        self.votingBuffer.clear()
        with ThreadPoolExecutor(max_workers=min(8, self.num_agents)) as executor:
            futures = [executor.submit(agent.cast_vote) for agent in self.agents]
            # Wait for all futures to complete
            for future in tqdm(futures, desc="Collecting Votes", total=len(futures)):
                future.result()  # This blocks until the task completes

        self.votingRound += 1
        self.votingHistory.append(self.votingBuffer.copy())
        voting_results = sorted(self.votingBuffer.items(), key=lambda x: x[1], reverse=True)
        voting_results_str = "\n".join([f"Cardinal {i} - {self.agents[i].name}: {votes}" for i, votes in voting_results])
        logger.info(f"Voting round {self.votingRound} completed.\n{voting_results_str}")
        logger.info(f"Total votes: {sum(self.votingBuffer.values())}")
        print(f"Voting round {self.votingRound} completed.\n{voting_results_str}")
        print(f"Total votes: {sum(self.votingBuffer.values())}")
        threshold = self.num_agents * self.supermajority_threshold
        
        # Check if any votes were cast
        if not voting_results:
            print("No votes were cast in this round.")
            self.votingBuffer.clear()
            return False
            
        print(f"most votes: {voting_results[0][1]}, threshold: {threshold} (supermajority: {self.supermajority_threshold*100:.1f}%)")

        # if the top candidate has more than the supermajority threshold of the votes
        if voting_results[0][1] > threshold:
            top_candidate = voting_results[0][0]
            self.winner = top_candidate
            print(f"Cardinal {top_candidate} wins!")
            return True

        self.votingBuffer.clear()
        return False

    def run_discussion_round(self, num_speakers: int = 5, random_selection: bool = False) -> None:
        """
        Run a discussion round where agents can speak about candidates or their own position.

        Args:
            num_speakers: Optional number of speakers to include in the discussion.
                          If None, all agents will participate.
            random: If True, selects speakers randomly instead of based on urgency.
                   If False, agents with higher speaking urgency are prioritized.
        """
        self.discussionRound += 1
        round_comments = []

        # Select speakers randomly or based on urgency
        if random_selection:
            # Log random selection mode
            logger.info(f"Using random selection for discussion round {self.discussionRound}")

            # Get all agent IDs
            all_agent_ids = list(range(len(self.agents)))

            # Randomly shuffle the IDs
            random.shuffle(all_agent_ids)

            # Select the specified number of speakers randomly
            if num_speakers > self.num_agents:
                selected_agent_ids = all_agent_ids
            else:
                selected_agent_ids = all_agent_ids[:num_speakers]

            # Create empty urgency scores for compatibility with the rest of the function
            urgency_scores = [{'agent_id': agent_id, 'urgency_score': 'Random', 'reasoning': 'Random selection'}
                             for agent_id in selected_agent_ids]

            # Get the corresponding agent objects
            speakers = [self.agents[agent_id] for agent_id in selected_agent_ids]

            # Log the random selection
            selected_str = "\n".join([
                f"Cardinal {agent_id} - {self.agents[agent_id].name}"
                for agent_id in selected_agent_ids
            ])
            logger.info(f"Randomly selected speakers for round {self.discussionRound}:\n{selected_str}")

        else:
            # Use original urgency-based selection
            logger.info(f"Evaluating speaking urgency for discussion round {self.discussionRound}")

            # Collect speaking urgency from all agents
            urgency_scores = []
            with ThreadPoolExecutor(max_workers=min(8, self.num_agents)) as executor:
                futures = [executor.submit(agent.speaking_urgency) for agent in self.agents]
                # Wait for all futures to complete
                for future in tqdm(futures, desc="Evaluating Speaking Urgency", total=len(futures)):
                    result = future.result()  # This blocks until the task completes
                    if result:
                        urgency_scores.append(result)

            # Sort agents by urgency score (highest to lowest)
            sorted_agents = sorted(urgency_scores, key=lambda x: x['urgency_score'], reverse=True)

            # Log the urgency scores
            urgency_str = "\n".join([
                f"Cardinal {score['agent_id']} - {self.agents[score['agent_id']].name}: {score['urgency_score']}/100\n"
                f"Reasoning: {score['reasoning']}"
                for score in sorted_agents
            ])
            logger.info(f"Speaking urgency for round {self.discussionRound}:\n{urgency_str}")

            # Determine which agents will speak in this round based on urgency
            if num_speakers > self.num_agents:
                selected_agent_ids = [score['agent_id'] for score in sorted_agents]
            else:
                # Take the top N agents by urgency score
                selected_agent_ids = [score['agent_id'] for score in sorted_agents[:num_speakers]]

            # Get the corresponding agent objects
            speakers = [self.agents[agent_id] for agent_id in selected_agent_ids]

        logger.info(f"Starting discussion round {self.discussionRound} with {len(speakers)} speakers")

        # Collect discussions from selected speakers
        futures = []
        with ThreadPoolExecutor(max_workers=min(8, len(speakers))) as executor:
            # Create a list to store agent-urgency pairs
            agent_urgency_pairs = []

            # Match each selected agent with their urgency data
            for agent in speakers:
                agent_id = agent.agent_id
                # Find the corresponding urgency data
                urgency_data = None
                for score in urgency_scores:
                    if score['agent_id'] == agent_id:
                        urgency_data = score
                        break

                # Submit the discuss task with the urgency data
                futures.append(executor.submit(agent.discuss, urgency_data))

            # Wait for all futures to complete
            for future in tqdm(futures, desc="Collecting Discussion", total=len(futures)):
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

        # Log the discussion with urgency included (or random selection note)
        discussion_str = "\n\n".join([
            f"Cardinal {comment['agent_id']} - {self.agents[comment['agent_id']].name} "
            f"{comment['message']}"
            for comment in round_comments
        ])
        logger.info(f"Discussion round {self.discussionRound} completed.\n{discussion_str}")
        print(f"\nDiscussion round {self.discussionRound} completed:")
        print("=" * 60)
        for comment in round_comments:
            agent_name = self.agents[comment['agent_id']].name
            print(f"\nCardinal {comment['agent_id']} - {agent_name}:")
            print(f"{comment['message']}")
        print("=" * 60)

    def list_candidates_for_prompt(self, randomize: bool = True) -> str:
        indices = list(range(self.num_agents))
        if randomize:
            random.shuffle(indices)
        candidates = [f"Cardinal {i}: {self.agents[i].name}" for i in indices]
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
