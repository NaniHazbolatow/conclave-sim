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
        discussion_history = self.env.get_discussion_history(self.agent_id)
        
        # Use prompt manager to get formatted prompt
        prompt = self.prompt_manager.get_voting_prompt(
            agent_name=self.name,
            background=self.background,
            candidates_list=self.env.list_candidates_for_prompt(),
            personal_vote_history=personal_vote_history,
            ballot_results_history=ballot_results_history,
            discussion_history=discussion_history
        )
        
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
                                "description": "The ID of the candidate to vote for"
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
                    self.env.cast_vote(vote)
                    self.logger.info(f"{self.name} ({self.agent_id}) voted for {self.env.agents[vote].name} ({vote}) because\n{reasoning}")
                    return
                else:
                    raise ValueError("Invalid vote")
            else:
                self.logger.error(f"Tool calling failed: {result.error}")
                raise ValueError(f"Tool calling failed: {result.error}")

        except Exception as e:
            # Default vote if there's an error
            self.logger.error(f"Error in LlmAgent {self.agent_id} voting: {e}")

    def speaking_urgency(self) -> Dict[str, any]:
        """
        Calculate how urgently the agent wants to speak in the next discussion round.

        Returns:
            Dict with urgency_score (1-100) and reasoning
        """
        personal_vote_history = self.promptize_vote_history()
        ballot_results_history = self.promptize_voting_results_history()
        discussion_history = self.env.get_discussion_history(self.agent_id)

        # Use prompt manager to get formatted prompt
        prompt = self.prompt_manager.get_speaking_urgency_prompt(
            agent_name=self.name,
            background=self.background,
            candidates_list=self.env.list_candidates_for_prompt(),
            personal_vote_history=personal_vote_history,
            ballot_results_history=ballot_results_history,
            discussion_history=discussion_history
        )

        # Define urgency evaluation tool
        tools = [
            {
                "type": "function",
                "function": {
                    "name": "evaluate_speaking_urgency",
                    "description": "Evaluate how urgently you want to speak",
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "urgency_score": {
                                "type": "integer",
                                "description": "Your urgency score (1-100)"
                            },
                            "reasoning": {
                                "type": "string",
                                "description": "Explain why you rated your urgency at this level"
                            }
                        },
                        "required": ["urgency_score", "reasoning"]
                    }
                }
            }
        ]

        try:
            # Use robust tool calling
            messages = [{"role": "user", "content": prompt}]
            result = self.tool_caller.call_tool(messages, tools, tool_choice="evaluate_speaking_urgency")
            
            if result.success and result.arguments:
                urgency_score = result.arguments.get("urgency_score", 50)
                reasoning = result.arguments.get("reasoning", "No reasoning provided.")
                
                # Ensure score is in range 1-100
                urgency_score = max(1, min(100, int(urgency_score)))
                
                return {
                    "agent_id": self.agent_id,
                    "urgency_score": urgency_score,
                    "reasoning": reasoning
                }
            else:
                self.logger.warning(f"Speaking urgency tool calling failed: {result.error}")
                return {
                    "agent_id": self.agent_id,
                    "urgency_score": 50,
                    "reasoning": f"Tool calling failed: {result.error}"
                }

        except Exception as e:
            self.logger.error(f"Error in LlmAgent {self.agent_id} speaking urgency: {e}")
            return {
                "agent_id": self.agent_id,
                "urgency_score": 50,
                "reasoning": f"Error during urgency evaluation: {e}"
            }

    def discuss(self, urgency_data: Optional[Dict] = None) -> Optional[Dict]:
        """
        Generate a discussion contribution about the conclave proceedings.

        Args:
            urgency_data: Optional dictionary containing urgency score and reasoning

        Returns:
            Dict with agent_id and message if successful, None otherwise
        """
        personal_vote_history = self.promptize_vote_history()
        ballot_results_history = self.promptize_voting_results_history()
        discussion_history = self.env.get_discussion_history(self.agent_id)

        # Include speaking urgency information if available
        urgency_context = ""
        if urgency_data and 'urgency_score' in urgency_data and 'reasoning' in urgency_data:
            urgency_context = self.prompt_manager.get_urgency_context(
                urgency_score=urgency_data['urgency_score'],
                urgency_reasoning=urgency_data['reasoning']
            )

        # Use prompt manager to get formatted prompt
        prompt = self.prompt_manager.get_discussion_prompt(
            agent_name=self.name,
            background=self.background,
            candidates_list=self.env.list_candidates_for_prompt(),
            personal_vote_history=personal_vote_history,
            ballot_results_history=ballot_results_history,
            discussion_history=discussion_history,
            urgency_context=urgency_context,
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
            vote_history_str = "\n".join([f"In round {i+1}, you voted for {self.env.agents[vote['vote']].name} for the following reason:\n{vote['reasoning']}" for i,vote in enumerate(self.vote_history)])
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