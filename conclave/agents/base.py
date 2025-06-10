from ..environments.conclave_env import ConclaveEnv
import json
from typing import Dict, List, Optional
import logging
import os
import threading
from ..config.manager import get_config
from ..config.prompts import get_prompt_manager

# Set tokenizer parallelism to avoid warnings in multiprocessing
os.environ["TOKENIZERS_PARALLELISM"] = "false"

logger = logging.getLogger(__name__)

# Shared LLM client manager to avoid multiple model instances
class SharedLLMManager:
    _instance = None
    _local_client = None
    _remote_client = None
    _lock = threading.Lock()
    
    def __new__(cls):
        if cls._instance is None:
            cls._instance = super().__new__(cls)
        return cls._instance
    
    def get_client(self, config):
        """Get shared LLM client instance based on configuration."""
        if config.is_local_backend():
            with self._lock:  # Thread-safe client creation
                if self._local_client is None:
                    from ..llm.client import HuggingFaceClient
                    client_kwargs = config.get_llm_client_kwargs()
                    client_kwargs = {k: v for k, v in client_kwargs.items() if v is not None}
                    self._local_client = HuggingFaceClient(**client_kwargs)
                    print(f"Created shared local LLM client: {self._local_client.model_name}")
                return self._local_client
        else:
            # For remote clients, create separate instances (they're stateless)
            from ..llm.client import RemoteLLMClient
            client_kwargs = config.get_llm_client_kwargs()
            client_kwargs = {k: v for k, v in client_kwargs.items() if v is not None}
            return RemoteLLMClient(**client_kwargs)

# Global shared manager instance
_shared_llm_manager = SharedLLMManager()

class Agent:
    def __init__(self, agent_id: int, name: str, background: str, env: ConclaveEnv):
        self.agent_id = agent_id
        self.name = name
        self.background = background
        self.env = env
        self.vote_history = []
        self.logger = logging.getLogger(name)
        
        # Internal stance tracking
        self.internal_stance = None  # Current internal stance as plain text
        self.stance_history = []     # History of stance updates with timestamps
        self.last_stance_update = None  # When stance was last generated
        
        # Get configuration and prompt manager
        self.config = get_config()
        self.prompt_manager = get_prompt_manager()
        
        # Use shared LLM client to avoid multiple model instances
        try:
            self.llm_client = _shared_llm_manager.get_client(self.config)
            
            # Initialize robust tool caller (import here to avoid circular imports)
            from ..llm.robust_tools import RobustToolCaller
            self.tool_caller = RobustToolCaller(self.llm_client, self.logger, config=self.config)
            
            self.logger.info(f"Agent {self.name} initialized with {self.config.get_backend_type()} backend, model: {self.llm_client.model_name}")
            
        except Exception as e:
            self.logger.error(f"Failed to initialize LLM client for agent {self.name}: {e}")
            raise

    def cast_vote(self) -> None:
        personal_vote_history = self.promptize_vote_history()
        ballot_results_history = self.promptize_voting_results_history()
        
        # Get current internal stance (generate if needed)
        internal_stance = self.get_internal_stance()
        
        # Use prompt manager to get formatted prompt
        prompt = self.prompt_manager.get_voting_prompt(
            agent_name=self.name,
            internal_stance=internal_stance,
            personal_vote_history=personal_vote_history,
            ballot_results_history=ballot_results_history
        )
        
        # Add candidate list to help with voting decision
        candidates_list = self.env.list_candidates_for_prompt(randomize=False)
        prompt += f"\n\nAvailable candidates:\n{candidates_list}\n"
        prompt += "IMPORTANT: Use the Cardinal ID number (0, 1, 2, etc.) as shown above when casting your vote."
        
        # Only log the prompt for debugging, don't print to console
        self.logger.debug(f"Voting prompt for {self.name}: {prompt}")
        
        # Define vote tool
        tools = [
            {
                "type": "function",
                "function": {
                    "name": "cast_vote",
                    "description": "Cast a vote for a candidate",
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "candidate": {
                                "type": "integer",
                                "description": "The Cardinal ID number (0, 1, 2, etc.) of the candidate to vote for. Must match the Cardinal ID shown in the candidate list."
                            },
                            "explanation": {
                                "type": "string",
                                "description": "Explain why you chose this candidate"
                            }
                        },
                        "required": ["candidate", "explanation"]
                    }
                }
            }
        ]
        try:
            # Use robust tool calling
            messages = [{"role": "user", "content": prompt}]
            result = self.tool_caller.call_tool(messages, tools, tool_choice="cast_vote")
            
            if result.success and result.arguments:
                vote = result.arguments.get("candidate")
                reasoning = result.arguments.get("explanation", "No explanation provided.")

                # Save vote reasoning
                self.vote_history.append({
                    "vote": vote,
                    "reasoning": reasoning
                })

                if vote is not None and isinstance(vote, int):
                    # Validate vote is within range
                    if 0 <= vote < len(self.env.agents):
                        voted_candidate_name = self.env.agents[vote].name
                        
                        # Check for potential reasoning-vote mismatch
                        if reasoning and voted_candidate_name.lower() not in reasoning.lower():
                            # Check if reasoning mentions a different candidate name
                            for i, agent in enumerate(self.env.agents):
                                if i != vote and agent.name.lower() in reasoning.lower():
                                    self.logger.warning(f"POTENTIAL VOTE MISMATCH: {self.name} voted for {voted_candidate_name} (ID {vote}) but reasoning mentions {agent.name}")
                                    break
                        
                        self.env.cast_vote(vote)
                        self.logger.info(f"{self.name} ({self.agent_id}) voted for {voted_candidate_name} ({vote}) because\n{reasoning}")
                        return
                    else:
                        self.logger.error(f"Invalid vote ID {vote}: must be between 0 and {len(self.env.agents)-1}")
                        raise ValueError(f"Invalid vote ID {vote}")
                else:
                    raise ValueError("Invalid vote")
            else:
                self.logger.error(f"Tool calling failed: {result.error}")
                raise ValueError(f"Tool calling failed: {result.error}")

        except Exception as e:
            # Default vote if there's an error
            self.logger.error(f"Error in LlmAgent {self.agent_id} voting: {e}")
            # Default to voting for the first candidate (ID 0)
            default_vote = 0
            default_candidate_name = self.env.agents[default_vote].name
            self.env.cast_vote(default_vote)
            self.logger.warning(f"{self.name} ({self.agent_id}) defaulted to voting for {default_candidate_name} ({default_vote}) due to error")



    def discuss(self) -> Optional[Dict]:
        """
        Generate a discussion contribution about the conclave proceedings.

        Returns:
            Dict with agent_id and message if successful, None otherwise
        """
        personal_vote_history = self.promptize_vote_history()
        ballot_results_history = self.promptize_voting_results_history()
        discussion_history = self.env.get_discussion_history(self.agent_id)
        
        # Get current internal stance (generate if needed)
        internal_stance = self.get_internal_stance()

        # Use prompt manager to get formatted prompt
        prompt = self.prompt_manager.get_discussion_prompt(
            agent_name=self.name,
            internal_stance=internal_stance,
            personal_vote_history=personal_vote_history,
            ballot_results_history=ballot_results_history,
            discussion_history=discussion_history,
            discussion_min_words=self.config.get_discussion_min_words(),
            discussion_max_words=self.config.get_discussion_max_words()
        )

        # Define speak tool
        min_words = self.config.get_discussion_min_words()
        max_words = self.config.get_discussion_max_words()
        tools = [
            {
                "type": "function",
                "function": {
                    "name": "speak_message",
                    "description": "Contribute a message to the conclave discussion",
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "message": {
                                "type": "string",
                                "description": f"Your contribution to the discussion ({min_words}-{max_words} words)"
                            }
                        },
                        "required": ["message"]
                    }
                }
            }
        ]

        try:
            # Use robust tool calling
            messages = [{"role": "user", "content": prompt}]
            result = self.tool_caller.call_tool(messages, tools, tool_choice="speak_message")
            
            if result.success and result.arguments:
                message = result.arguments.get("message", "")

                # Log the discussion contribution
                self.logger.info(f"{self.name} ({self.agent_id}) speaks:\n{message}")
                # Don't print individual speaking to console - it will be shown in the summary

                # Return the discussion contribution
                return {
                    "agent_id": self.agent_id,
                    "message": message
                }
            else:
                self.logger.warning(f"Discussion tool calling failed: {result.error}")
                return {
                    "agent_id": self.agent_id,
                    "message": f"Tool calling failed: {result.error}"
                }

        except Exception as e:
            # Log the error and return None if there's a problem
            self.logger.error(f"Error in LlmAgent {self.agent_id} discussion: {e}")
            return None

    def promptize_vote_history(self) -> str:
        if self.vote_history:
            vote_history_entries = []
            for i, vote in enumerate(self.vote_history):
                try:
                    # Ensure vote is a dictionary and has the expected keys
                    if isinstance(vote, dict) and 'vote' in vote and 'reasoning' in vote:
                        vote_idx = vote['vote']
                        if isinstance(vote_idx, int) and 0 <= vote_idx < len(self.env.agents):
                            candidate_name = self.env.agents[vote_idx].name
                            reasoning = vote['reasoning']
                            vote_history_entries.append(f"In round {i+1}, you voted for {candidate_name} for the following reason:\n{reasoning}")
                        else:
                            vote_history_entries.append(f"In round {i+1}, you cast an invalid vote: {vote_idx}")
                    else:
                        vote_history_entries.append(f"In round {i+1}, vote data was malformed: {vote}")
                except Exception as e:
                    logger.error(f"Error processing vote history entry {i}: {vote}, error: {e}")
                    vote_history_entries.append(f"In round {i+1}, error processing vote")
            
            vote_history_str = "\n".join(vote_history_entries)
            return f"Your vote history:\n{vote_history_str}\n"
        else:
            return ""

    def promptize_voting_results_history(self) -> str:
        def promptize_voting_results(results: Dict[str, int]) -> str:
            voting_results = sorted(results.items(), key=lambda x: x[1], reverse=True)
            if results:
                voting_results_str = "\n".join([f"Cardinal {i} - {self.env.agents[i].name}: {votes}" for i, votes in voting_results])
                return f"\n{voting_results_str}\n"
            else:
                return ""

        if self.env.votingHistory:
            voting_results_history_str = "\n".join([f"Round {i+1}: {promptize_voting_results(result)}" for i,result in enumerate(self.env.votingHistory)])
            return f"Previous ballot results:\n{voting_results_history_str}"
        else:
            return ""

    def generate_internal_stance(self) -> str:
        """
        Generate an internal stance based on the agent's background, discussions, and voting history.
        
        Returns:
            The generated internal stance as plain text
        """
        import datetime
        
        personal_vote_history = self.promptize_vote_history()
        ballot_results_history = self.promptize_voting_results_history()
        discussion_history = self.env.get_discussion_history(self.agent_id)
        
        # Use prompt manager to get formatted prompt
        prompt = self.prompt_manager.get_internal_stance_prompt(
            agent_name=self.name,
            background=self.background,
            candidates_list=self.env.list_candidates_for_prompt(randomize=False),
            personal_vote_history=personal_vote_history,
            ballot_results_history=ballot_results_history,
            discussion_history=discussion_history
        )
        
        try:
            # Define stance generation tool with strict word limit
            tools = [
                {
                    "type": "function", 
                    "function": {
                        "name": "generate_stance",
                        "description": "Generate a concise internal stance on the papal election",
                        "parameters": {
                            "type": "object",
                            "properties": {
                                "stance": {
                                    "type": "string",
                                    "description": "Your internal stance (75-125 words ONLY). Must be direct, concrete, and avoid diplomatic language. Focus on specific positions and candidate preferences."
                                }
                            },
                            "required": ["stance"]
                        }
                    }
                }
            ]
            
            # Generate stance using tool calling
            messages = [{"role": "user", "content": prompt}]
            result = self.tool_caller.call_tool(messages, tools, tool_choice="generate_stance")
            
            if result.success and result.arguments:
                stance = result.arguments.get("stance", "")
                
                # Validate word count (strict enforcement)
                word_count = len(stance.split())
                if word_count > 150:
                    self.logger.warning(f"Stance too long ({word_count} words), truncating to 125 words...")
                    words = stance.split()[:125]
                    stance = " ".join(words)
                elif word_count < 50:
                    self.logger.warning(f"Stance too short ({word_count} words). This may affect embedding quality.")
                
                # Update internal state
                timestamp = datetime.datetime.now()
                self.internal_stance = stance
                self.last_stance_update = timestamp
                
                # Add to stance history
                stance_entry = {
                    "timestamp": timestamp,
                    "stance": stance,
                    "voting_round": self.env.votingRound,
                    "discussion_round": self.env.discussionRound
                }
                self.stance_history.append(stance_entry)
                
                # Enhanced logging with full stance content
                self.logger.info(f"{self.name} generated internal stance (V{self.env.votingRound}.D{self.env.discussionRound})")
                self.logger.info(f"=== INTERNAL STANCE: {self.name} ===")
                self.logger.info(f"Round: Voting {self.env.votingRound}, Discussion {self.env.discussionRound}")
                self.logger.info(f"Word count: {word_count}")
                self.logger.info(f"Stance Content:")
                self.logger.info(stance)
                self.logger.info(f"=== END STANCE: {self.name} ===")
                
                return stance
            else:
                self.logger.error(f"Failed to generate stance: {result.error}")
                return ""
            
        except Exception as e:
            self.logger.error(f"Error generating internal stance for {self.name}: {e}")
            return ""
    
    def get_internal_stance(self) -> str:
        """
        Get the current internal stance, generating one if it doesn't exist.
        
        Returns:
            The agent's current internal stance
        """
        if self.internal_stance is None:
            return self.generate_internal_stance()
        return self.internal_stance
    
    def should_update_stance(self) -> bool:
        """
        Determine if the agent should update their internal stance.
        
        Returns:
            True if stance should be updated, False otherwise
        """
        # Update stance if:
        # 1. No stance exists yet
        # 2. There's been new voting or discussion activity since last update
        if self.internal_stance is None:
            return True
            
        if self.last_stance_update is None:
            return True
            
        # Check if there's been new activity since last stance update
        current_activity = (self.env.votingRound, self.env.discussionRound)
        
        # If we have stance history, compare with last recorded activity
        if self.stance_history:
            last_entry = self.stance_history[-1]
            last_activity = (last_entry["voting_round"], last_entry["discussion_round"])
            return current_activity != last_activity
        
        return True