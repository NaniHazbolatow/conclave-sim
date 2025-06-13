from ..environments.conclave_env import ConclaveEnv
import json
import re
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
        self.background = background  # This serves as 'biography' from glossary
        self.env = env
        self.vote_history = []
        self.logger = logging.getLogger(name)
        
        # Load persona data from CSV
        self.cardinal_id = agent_id  # Cardinal ID from glossary
        self.role_tag = "ELECTOR"    # Default role, can be "CANDIDATE" or "ELECTOR"
        
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
        """Cast a vote using the new variable-based voting prompts."""
        # Get role information for testing groups and set role_tag
        if self.env.testing_groups_enabled:
            self.role_tag = "CANDIDATE" if self.env.is_candidate(self.agent_id) else "ELECTOR"
        else:
            self.role_tag = "ELECTOR"  # In normal mode, all can vote, but we treat as electors
        
        # Use the updated prompt manager method with agent_id
        voting_prompt = self.prompt_manager.get_voting_prompt(agent_id=self.agent_id)
        
        # Debug log the prompt
        self.logger.debug(f"Voting prompt for {self.name}: {voting_prompt}")
        
        # Define vote tool that expects Cardinal ID
        tools = [
            {
                "type": "function",
                "function": {
                    "name": "cast_vote",
                    "description": "Cast a vote for a candidate using their Cardinal ID",
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "vote_cardinal_id": {
                                "type": "integer",
                                "description": "Cardinal ID of the candidate to vote for"
                            }
                        },
                        "required": ["vote_cardinal_id"]
                    }
                }
            }
        ]
        
        try:
            # Use robust tool calling
            messages = [{"role": "user", "content": voting_prompt}]
            result = self.tool_caller.call_tool(messages, tools, tool_choice="cast_vote")
            
            # Debug: Log the full tool calling result
            self.logger.debug(f"=== VOTE TOOL CALLING RESULT FOR {self.name} ===")
            self.logger.debug(f"Success: {result.success}")
            self.logger.debug(f"Arguments: {result.arguments}")
            self.logger.debug(f"Error: {result.error}")
            self.logger.debug(f"=== END VOTE RESULT ===")
            
            if result.success and result.arguments:
                cardinal_id = result.arguments.get("vote_cardinal_id")

                # Validate and convert Cardinal ID to agent ID
                if cardinal_id is not None:
                    try:
                        cardinal_id = int(cardinal_id)
                        
                        # Find the agent with this cardinal_id
                        agent_id = None
                        for i, agent in enumerate(self.env.agents):
                            if getattr(agent, 'cardinal_id', i) == cardinal_id:
                                agent_id = i
                                break
                        
                        if agent_id is None:
                            available_cardinals = [getattr(agent, 'cardinal_id', i) for i, agent in enumerate(self.env.agents)]
                            raise ValueError(f"Cardinal ID {cardinal_id} not found. Available: {available_cardinals}")
                        
                        # Check if candidate is valid for voting in current mode
                        if not self.env.is_valid_vote_candidate(agent_id):
                            available_candidates = [getattr(self.env.agents[cid], 'cardinal_id', cid) for cid in self.env.get_candidates_list()]
                            raise ValueError(f"Cardinal {cardinal_id} is not eligible to receive votes. Available candidates: {available_candidates}")
                        
                        self.logger.debug(f"Valid Cardinal ID {cardinal_id} (Agent {agent_id}) selected")
                        
                        # Save vote reasoning (simplified since new prompts don't include explanation)
                        self.vote_history.append({
                            "vote": agent_id,
                            "reasoning": f"Voted for Cardinal {cardinal_id} based on internal stance and voting prompt analysis"
                        })

                        # Cast the vote using agent_id
                        self.env.cast_vote(agent_id, self.agent_id)
                        voted_candidate_name = self.env.agents[agent_id].name
                        self.logger.info(f"{self.name} voted for Cardinal {cardinal_id} - {voted_candidate_name}")
                        return
                        
                    except (ValueError, TypeError) as e:
                        self.logger.error(f"Invalid Cardinal ID '{cardinal_id}': {e}")
                        raise ValueError(f"Invalid Cardinal ID: {cardinal_id}")
                else:
                    self.logger.error("No Cardinal ID provided in vote")
                    raise ValueError("No Cardinal ID provided")
            else:
                self.logger.error(f"Tool calling failed: {result.error}")
                raise ValueError(f"Tool calling failed: {result.error}")

        except Exception as e:
            # Log the error and re-raise to fail hard
            self.logger.error(f"Error in {self.name} voting: {e}")
            self.logger.debug(f"Voting failed for {self.name}", exc_info=True)
            raise  # Re-raise the exception to fail hard



    def discuss(self) -> Optional[Dict]:
        """
        Generate a discussion contribution about the conclave proceedings.

        Returns:
            Dict with agent_id and message if successful, None otherwise
        """
        personal_vote_history = self.promptize_vote_history()
        ballot_results_history = self.promptize_voting_results_history()
        discussion_history = self.env.get_discussion_history(self.agent_id)
        short_term_memory = self.get_short_term_memory()
        recent_speech_snippets = self.get_recent_speech_snippets()
        
        # Get current internal stance (generate if needed)
        internal_stance = self.get_internal_stance()

        # Get current discussion participants information
        current_discussion_participants = self.env.get_current_discussion_participants()
        
        # Get current scoreboard for visual context
        current_scoreboard = self.env.get_current_scoreboard()
        
        # Get role information for testing groups
        role_description = self.env.get_role_description(self.agent_id)
        candidates_description = self.env.get_candidates_description()

        # Use prompt manager to get formatted prompt
        prompt = self.prompt_manager.get_discussion_prompt(
            agent_id=self.agent_id,
            current_scoreboard=current_scoreboard,
            agent_name=self.name,
            background=self.background,
            internal_stance=internal_stance,
            personal_vote_history=personal_vote_history,
            ballot_results_history=ballot_results_history,
            discussion_history=discussion_history,
            short_term_memory=short_term_memory,
            recent_speech_snippets=recent_speech_snippets,
            current_discussion_participants=current_discussion_participants,
            discussion_round=self.env.discussionRound,
            voting_round=self.env.votingRound,
            discussion_min_words=self.config.get_discussion_min_words(),
            discussion_max_words=self.config.get_discussion_max_words(),
            role_description=role_description,
            candidates_description=candidates_description
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
                # Map agent indices to Cardinal_ID for display
                voting_results_str = "\n".join([
                    f"Cardinal {getattr(self.env.agents[i], 'cardinal_id', i)} - {self.env.agents[i].name}: {votes}" 
                    for i, votes in voting_results
                ])
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
        
        # Get the last stance for continuity
        last_stance = self.get_last_stance()
        
        # Get valid candidates list for stance generation
        valid_candidates_list = self.env.get_valid_candidates_for_stance()
        
        # Use prompt manager to get formatted prompt
        prompt = self.prompt_manager.get_internal_stance_prompt(
            agent_id=self.agent_id,
            agent_name=self.name,
            background=self.background,
            last_stance=last_stance,
            valid_candidates_list=valid_candidates_list,
            candidates_list=self.env.list_candidates_for_prompt(randomize=False),
            personal_vote_history=personal_vote_history,
            ballot_results_history=ballot_results_history,
            discussion_history=discussion_history,
            voting_round=self.env.votingRound,
            discussion_round=self.env.discussionRound
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
                
                return stance
            else:
                self.logger.warning(f"Tool calling failed for stance generation: {result.error}")
                self.logger.info("Attempting fallback direct prompting for stance generation...")
                
                # Fallback: Use direct prompting without tool calling
                return self._generate_stance_fallback(prompt)
            
        except Exception as e:
            self.logger.error(f"Error generating internal stance for {self.name}: {e}")
            self.logger.info("Attempting fallback direct prompting for stance generation...")
            return self._generate_stance_fallback(prompt)
    
    def _generate_stance_fallback(self, original_prompt: str) -> str:
        """Fallback method for stance generation when tool calling fails."""
        try:
            # Create a simple prompt that asks for direct text output
            fallback_prompt = f"""{original_prompt}

RESPOND WITH YOUR INTERNAL STANCE DIRECTLY (NO JSON):
Write exactly 75-125 words covering:
- Your preferred candidate and why
- What the Church should focus on
- Your theological position (traditional/moderate/progressive)
- Your main concerns
- How recent discussions influenced you

Start your response immediately with your stance (no introductory text):"""

            messages = [{"role": "user", "content": fallback_prompt}]
            response = self.llm_client.generate(messages)
            
            if response and len(response.strip()) > 20:
                # Clean up the response
                stance = response.strip()
                
                # Remove any JSON artifacts or formatting
                stance = re.sub(r'^["\'{}\[\]]+|["\'{}\[\]]+$', '', stance)
                stance = re.sub(r'\n+', ' ', stance)  # Replace newlines with spaces
                stance = ' '.join(stance.split())  # Normalize whitespace
                
                # Validate word count
                word_count = len(stance.split())
                if word_count > 150:
                    words = stance.split()[:125]
                    stance = " ".join(words)
                
                if word_count >= 30:  # Minimum acceptable length
                    # Update internal state
                    import datetime
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
                    
                    self.logger.info(f"{self.name} generated fallback stance (V{self.env.votingRound}.D{self.env.discussionRound})")
                    self.logger.info(f"=== FALLBACK STANCE: {self.name} ===")
                    self.logger.info(f"Word count: {len(stance.split())}")
                    self.logger.info(f"Stance Content:")
                    self.logger.info(stance)
                    self.logger.info(f"=== END FALLBACK STANCE: {self.name} ===")
                    
                    return stance
                else:
                    self.logger.warning(f"Fallback stance too short ({word_count} words)")
                    return ""
            else:
                self.logger.error("Fallback stance generation failed - empty response")
                return ""
                
        except Exception as e:
            self.logger.error(f"Fallback stance generation failed for {self.name}: {e}")
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

    def get_last_three_speeches(self) -> str:
        """Get this agent's last 3 speeches for style continuity."""
        agent_speeches = []
        
        # Go through discussion history and collect this agent's speeches
        for round_idx, round_comments in enumerate(self.env.discussionHistory):
            for comment in round_comments:
                if comment['agent_id'] == self.agent_id:
                    agent_speeches.append({
                        'round': round_idx + 1,
                        'message': comment['message']
                    })
        
        # Get last 3 speeches
        recent_speeches = agent_speeches[-3:] if len(agent_speeches) >= 3 else []
        
        if not recent_speeches:
            return ""
        
        formatted_speeches = []
        for speech in recent_speeches:
            formatted_speeches.append(f"Round {speech['round']}: {speech['message']}")
        
        return "Your last 3 speeches (for style reference):\n" + "\n\n".join(formatted_speeches) + "\n"

    def get_short_term_memory(self) -> str:
        """Compile short-term memory components for discussion prompts."""
        memory_components = []
        
        # Add last 3 speeches
        last_speeches = self.get_last_three_speeches()
        if last_speeches:
            memory_components.append(last_speeches)
        
        if not memory_components:
            return ""
        
        return "SHORT-TERM MEMORY:\n" + "\n".join(memory_components) + "\n"
    
    def get_recent_speech_snippets(self) -> str:
        """Get last 2-3 speech snippets from this agent to avoid verbatim reuse."""
        agent_speeches = []
        
        # Collect this agent's recent speeches
        for round_idx, round_comments in enumerate(self.env.discussionHistory):
            for comment in round_comments:
                if comment['agent_id'] == self.agent_id:
                    # Extract first ~30 words as snippet
                    words = comment['message'].split()
                    snippet = ' '.join(words[:30])
                    if len(words) > 30:
                        snippet += "..."
                    
                    agent_speeches.append({
                        'round': round_idx + 1,
                        'snippet': snippet
                    })
        
        # Get last 2-3 speeches
        recent_speeches = agent_speeches[-3:] if len(agent_speeches) >= 2 else agent_speeches
        
        if not recent_speeches:
            return "No previous speeches"
        
        formatted_snippets = []
        for speech in recent_speeches:
            formatted_snippets.append(f"Round {speech['round']}: \"{speech['snippet']}\"")
        
        return " | ".join(formatted_snippets)
    
    def get_last_stance(self) -> str:
        """Get the agent's last stance from their stance history."""
        if not self.stance_history:
            return "None - this is your first stance."
        
        last_entry = self.stance_history[-1]
        last_stance = last_entry.get("stance", "")
        voting_round = last_entry.get("voting_round", "?")
        discussion_round = last_entry.get("discussion_round", "?")
        
        return f"Previous stance (V{voting_round}.D{discussion_round}): {last_stance}"

    def get_persona_internal(self) -> str:
        """
        Get the agent's internal persona, generating one if it doesn't exist.
        
        Returns:
            The agent's internal persona
        """
        if not hasattr(self, 'persona_internal') or not self.persona_internal:
            return self.generate_persona_internal()
        return self.persona_internal
