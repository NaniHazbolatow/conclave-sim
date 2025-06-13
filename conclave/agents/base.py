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
        # Determine role tag for the agent for this voting round
        if self.env.testing_groups_enabled and hasattr(self.env, 'is_candidate') and self.env.is_candidate(self.agent_id):
            self.role_tag = "CANDIDATE"
        else:
            self.role_tag = "ELECTOR" 
        
        # Get the appropriate voting prompt based on agent_id and determined role
        # The PromptManager will handle selecting voting_candidate or voting_elector
        voting_prompt = self.prompt_manager.get_voting_prompt(agent_id=self.agent_id)
        
        self.logger.debug(f"Voting prompt for {self.name} (Role: {self.role_tag}): {voting_prompt}")
        
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
            messages = [{"role": "user", "content": voting_prompt}]
            result = self.tool_caller.call_tool(messages, tools, tool_choice="cast_vote")
            
            self.logger.debug(f"=== VOTE TOOL CALLING RESULT FOR {self.name} ===")
            self.logger.debug(f"Success: {result.success}")
            self.logger.debug(f"Arguments: {result.arguments}")
            self.logger.debug(f"Error: {result.error}")
            self.logger.debug(f"=== END VOTE RESULT ===")
            
            if result.success and result.arguments:
                cardinal_id = result.arguments.get("vote_cardinal_id")

                if cardinal_id is not None:
                    try:
                        cardinal_id = int(cardinal_id)
                        # Convert cardinal_id to agent_id (internal)
                        # This assumes a mapping or direct use if cardinal_id is the agent_id
                        # For now, let's assume cardinal_id from prompt IS the agent_id for simplicity
                        # If there's a separate mapping, it needs to be resolved here.
                        # Example: agent_to_vote_for_id = self.env.get_agent_id_from_cardinal_id(cardinal_id)
                        
                        agent_to_vote_for_id = None
                        for i, agent_in_env in enumerate(self.env.agents):
                            if getattr(agent_in_env, 'cardinal_id', i) == cardinal_id:
                                agent_to_vote_for_id = i
                                break
                        
                        if agent_to_vote_for_id is None:
                            available_cardinals = [getattr(agent, 'cardinal_id', i) for i, agent in enumerate(self.env.agents)]
                            raise ValueError(f"Cardinal ID {cardinal_id} not found. Available: {available_cardinals}")

                        if not self.env.is_valid_vote_candidate(agent_to_vote_for_id):
                            # Provide more context on available candidates if vote is invalid
                            # This requires a method in env to list votable candidates' cardinal IDs
                            # For now, just log the error
                            raise ValueError(f"Cardinal {cardinal_id} (Agent ID: {agent_to_vote_for_id}) is not eligible to receive votes.")

                        self.logger.debug(f"Valid Cardinal ID {cardinal_id} (Agent {agent_to_vote_for_id}) selected by {self.name}")
                        
                        self.vote_history.append({
                            "vote": agent_to_vote_for_id, # Store internal agent_id
                            "reasoning": f"Voted for Cardinal {cardinal_id} based on internal stance and voting prompt analysis."
                        })

                        self.env.cast_vote(agent_to_vote_for_id, self.agent_id)
                        voted_candidate_name = self.env.agents[agent_to_vote_for_id].name
                        self.logger.info(f"{self.name} voted for Cardinal {cardinal_id} - {voted_candidate_name}")
                        return
                        
                    except (ValueError, TypeError) as e:
                        self.logger.error(f"Invalid Cardinal ID '{cardinal_id}' provided by {self.name}: {e}")
                        # Consider not re-raising if a fallback (e.g., abstaining) is desired
                        raise ValueError(f"Invalid Cardinal ID: {cardinal_id}") 
                else:
                    self.logger.error(f"{self.name} failed to provide a Cardinal ID in vote.")
                    raise ValueError("No Cardinal ID provided in vote by agent.")
            else:
                error_msg = result.error if result.error else "Unknown tool calling failure."
                self.logger.error(f"Tool calling failed for {self.name} during voting: {error_msg}")
                raise ValueError(f"Tool calling failed: {error_msg}")

        except Exception as e:
            self.logger.error(f"Error in {self.name} voting process: {e}", exc_info=True)
            # Re-raise to ensure simulation handles critical failures
            raise

    def discuss(self) -> Optional[Dict]:
        """
        Generate a discussion contribution.
        Variables are now primarily sourced via PromptVariableGenerator.
        """
        # Determine role tag for the agent for this discussion round
        if self.env.testing_groups_enabled and hasattr(self.env, 'is_candidate') and self.env.is_candidate(self.agent_id):
            self.role_tag = "CANDIDATE"
        else:
            self.role_tag = "ELECTOR"

        # Most variables are now automatically handled by PromptManager and PromptVariableGenerator
        # We only need to pass agent_id and any specific non-standard variables if required.
        # Standard variables like agent_name, background, internal_stance, etc.,
        # are fetched by PromptVariableGenerator based on agent_id.

        # Example of passing additional, non-standard variables if needed:
        # extra_discussion_vars = {
        #     'custom_variable_for_discussion': "some_value"
        # }
        # prompt = self.prompt_manager.get_discussion_prompt(agent_id=self.agent_id, **extra_discussion_vars)

        prompt = self.prompt_manager.get_discussion_prompt(
            agent_id=self.agent_id,
            # Pass any variables not covered by PromptVariableGenerator or if overriding is needed
            # For example, if these are dynamically calculated here and not part of standard agent vars:
            # discussion_min_words=self.config.get_discussion_min_words(), 
            # discussion_max_words=self.config.get_discussion_max_words()
            # However, if these are fixed or derivable by PromptVariableGenerator, they can be removed from here.
            # For now, assuming they might be dynamic or specific to this call context:
            discussion_min_words=self.config.get_discussion_min_words(),
            discussion_max_words=self.config.get_discussion_max_words(),
            # The following are likely covered by PromptVariableGenerator if defined in prompts.yaml:
            # current_scoreboard, personal_vote_history, ballot_results_history, 
            # discussion_history, short_term_memory, recent_speech_snippets, 
            # current_discussion_participants, role_description, candidates_description
        )

        self.logger.debug(f"Discussion prompt for {self.name} (Role: {self.role_tag}):\n{prompt}")

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
            # The 'print(prompt)' can be removed or kept for debugging as needed
            # print(prompt) 
            messages = [{"role": "user", "content": prompt}]
            result = self.tool_caller.call_tool(messages, tools, tool_choice="speak_message")
            
            if result.success and result.arguments:
                message = result.arguments.get("message", "")
                if not message.strip():
                    self.logger.warning(f"{self.name} ({self.agent_id}) provided an empty message.")
                    # Optionally return a default message or handle as an error
                    message = "(Agent provided no message)"

                self.logger.info(f"{self.name} ({self.agent_id}) speaks:\n{message}")
                return {"agent_id": self.agent_id, "message": message}
            else:
                error_msg = result.error if result.error else "Unknown tool calling failure."
                self.logger.warning(f"Discussion tool calling failed for {self.name}: {error_msg}")
                # Return a message indicating failure, useful for the environment to track
                return {"agent_id": self.agent_id, "message": f"(Tool calling failed: {error_msg})"}

        except Exception as e:
            self.logger.error(f"Error in Agent {self.name} ({self.agent_id}) discussion: {e}", exc_info=True)
            # Return None or a dict with error to allow simulation to continue if desired
            return {"agent_id": self.agent_id, "message": f"(Discussion error: {e})"}

    def generate_internal_stance(self) -> str:
        """
        Generate an internal stance. Variables are now primarily sourced via PromptVariableGenerator.
        """
        import datetime # Keep for timestamping

        # Determine role tag for the agent for stance generation
        if self.env.testing_groups_enabled and hasattr(self.env, 'is_candidate') and self.env.is_candidate(self.agent_id):
            self.role_tag = "CANDIDATE"
        else:
            self.role_tag = "ELECTOR"

        # Variables like personal_vote_history, ballot_results_history, discussion_history,
        # last_stance, valid_candidates_list, candidates_list are now expected to be handled
        # by PromptVariableGenerator if they are defined in the 'stance' prompt in prompts.yaml.
        # The agent_id is passed, and PromptManager + PromptVariableGenerator will fetch them.

        prompt = self.prompt_manager.get_internal_stance_prompt(
            agent_id=self.agent_id
            # Add any non-standard variables here if needed, e.g.:
            # my_custom_stance_variable="custom_value"
        )
        
        self.logger.debug(f"Internal stance prompt for {self.name}:\n{prompt}")

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
            
        try:
            messages = [{"role": "user", "content": prompt}]
            result = self.tool_caller.call_tool(messages, tools, tool_choice="generate_stance")
            
            if result.success and result.arguments:
                stance = result.arguments.get("stance", "").strip()
                
                if not stance:
                    self.logger.warning(f"{self.name} generated an empty stance. Attempting fallback.")
                    return self._generate_stance_fallback(prompt) # Pass original prompt to fallback

                word_count = len(stance.split())
                if word_count > 150: # Max length check
                    self.logger.warning(f"Stance from {self.name} too long ({word_count} words), truncating to 125 words...")
                    stance = " ".join(stance.split()[:125])
                elif word_count < 50: # Min length warning
                    self.logger.warning(f"Stance from {self.name} too short ({word_count} words). This may affect embedding quality.")
                
                timestamp = datetime.datetime.now()
                self.internal_stance = stance
                self.last_stance_update = timestamp
                self.stance_history.append({
                    "timestamp": timestamp, "stance": stance,
                    "voting_round": self.env.votingRound, "discussion_round": self.env.discussionRound
                })
                self.logger.info(f"{self.name} generated new stance (V{self.env.votingRound}.D{self.env.discussionRound}): '{stance[:100]}...'")
                return stance
            else:
                error_msg = result.error if result.error else "Unknown tool calling failure."
                self.logger.warning(f"Tool calling failed for stance generation for {self.name}: {error_msg}. Attempting fallback.")
                return self._generate_stance_fallback(prompt) # Pass original prompt to fallback
            
        except Exception as e:
            self.logger.error(f"Error generating internal stance for {self.name}: {e}", exc_info=True)
            self.logger.info(f"Attempting fallback direct prompting for stance generation for {self.name}...")
            return self._generate_stance_fallback(prompt) # Pass original prompt to fallback
    
    def _generate_stance_fallback(self, original_prompt: str) -> str:
        """Fallback method for stance generation when tool calling fails or returns empty."""
        try:
            # Ensure datetime is imported if not already at the top level of the class/method
            import datetime

            fallback_prompt = f"""{original_prompt}

RESPOND WITH YOUR INTERNAL STANCE DIRECTLY (NO JSON OR TOOL FORMATTING):
Write exactly 75-125 words covering:
- Your preferred candidate and why
- What the Church should focus on
- Your theological position (traditional/moderate/progressive)
- Your main concerns
- How recent discussions influenced you

Start your response immediately with your stance (no introductory text):"""

            messages = [{"role": "user", "content": fallback_prompt}]
            response_content = self.llm_client.generate(messages) # Assuming generate returns the string content
            
            if response_content and len(response_content.strip()) > 20:
                stance = response_content.strip()
                stance = re.sub(r'^["\'{{\[\]}}]+|["\'{{\[\]}}]+$', '', stance) # Basic cleaning
                stance = re.sub(r'\n+', ' ', stance).strip() # Normalize newlines and whitespace
                stance = ' '.join(stance.split()) # Ensure single spaces
                
                word_count = len(stance.split())
                if word_count > 150:
                    stance = " ".join(stance.split()[:125])
                
                if word_count >= 30:  # Minimum acceptable length for a fallback
                    timestamp = datetime.datetime.now()
                    self.internal_stance = stance
                    self.last_stance_update = timestamp
                    self.stance_history.append({
                        "timestamp": timestamp, "stance": stance,
                        "voting_round": self.env.votingRound, "discussion_round": self.env.discussionRound
                    })
                    self.logger.info(f"{self.name} generated FALLBACK stance (V{self.env.votingRound}.D{self.env.discussionRound}), {word_count} words: '{stance[:100]}...'")
                    return stance
                else:
                    self.logger.warning(f"Fallback stance for {self.name} too short ({word_count} words) after cleaning.")
                    return ""
            else:
                self.logger.error(f"Fallback stance generation for {self.name} failed - empty or too short response from LLM.")
                return ""
                
        except Exception as e:
            self.logger.error(f"Fallback stance generation for {self.name} failed with exception: {e}", exc_info=True)
            return ""
    
    def get_internal_stance(self) -> str:
        """
        Get the current internal stance, generating one if it doesn't exist or needs update.
        """
        if self.should_update_stance(): # Check if stance needs update
            self.logger.info(f"Updating stance for {self.name} (V{self.env.votingRound}.D{self.env.discussionRound}). Last update: {self.last_stance_update}")
            self.generate_internal_stance()
        elif self.internal_stance is None: # Should be caught by should_update_stance, but as a safeguard
            self.logger.info(f"No stance found for {self.name}, generating initial stance.")
            self.generate_internal_stance()
            
        return self.internal_stance if self.internal_stance is not None else ""

    def should_update_stance(self) -> bool:
        """
        Determine if the agent should update their internal stance.
        Updates if: no stance, no last_stance_update, or new activity (vote/discussion round).
        """
        if self.internal_stance is None or self.last_stance_update is None:
            return True
        
        # Check if there's been new activity since last stance update
        # This requires stance_history to store the rounds at which stance was generated
        if self.stance_history:
            last_recorded_activity = (self.stance_history[-1]['voting_round'], self.stance_history[-1]['discussion_round'])
            current_activity = (self.env.votingRound, self.env.discussionRound)
            if current_activity > last_recorded_activity:
                return True # New round activity detected
        else:
            # No history, but stance exists. This implies it might be the first round after init.
            # If current rounds are > 0 and it matches initial state, it might not need update unless forced.
            # However, if it's the very first time, it should update.
            # This case is mostly covered by internal_stance is None.
            # If stance_history is empty but stance exists, it means it was set without history, perhaps an initial default.
            # Let's assume if history is empty, and rounds have started, it's safer to update.
            if self.env.votingRound > 0 or self.env.discussionRound > 0:
                 return True

        return False # Default to not updating if none of the above conditions met

    # Removed get_last_three_speeches - covered by PromptVariableGenerator's recent_speech_snippets
    # Removed get_short_term_memory - specific STM components should be individual prompt variables

    def get_last_stance(self) -> str:
        """Get the most recent internal stance text."""
        return self.internal_stance if self.internal_stance else "No previous stance recorded."

    # Persona and Profile Management (Example placeholders - adapt to your CSV loading)
    def load_persona_from_data(self, cardinal_data: Dict):
        """Load persona details from provided data (e.g., a row from CSV)."""
        self.persona_internal = cardinal_data.get('internal_persona', '')
        self.profile_public = cardinal_data.get('external_profile', '')
        # self.background is already set in __init__
        self.name = cardinal_data.get('name', self.name)
        self.cardinal_id = cardinal_data.get('cardinal_id', self.agent_id) # Ensure cardinal_id is correctly set
        self.logger.info(f"Loaded persona for {self.name} (Cardinal ID: {self.cardinal_id})")

    # Example of how an agent might be updated with data post-initialization
    # This would typically be called by the environment after loading all agent data.
    # For now, this is a placeholder method.
    def update_agent_details(self, details: Dict):
        """Update agent details, e.g., from a central data source after init."""
        if 'internal_persona' in details: self.persona_internal = details['internal_persona']
        if 'profile_public' in details: self.profile_public = details['profile_public']
        if 'role_tag' in details: self.role_tag = details['role_tag']
        # Add other relevant fields as necessary
        self.logger.debug(f"Agent {self.name} details updated.")

    def __repr__(self):
        return f"Agent(id={self.agent_id}, name='{self.name}', role='{self.role_tag}')"

    def get_recent_speech_snippets(self) -> str:
        """Placeholder: This logic should be in PromptVariableGenerator if used by prompts."""
        self.logger.warning("Agent.get_recent_speech_snippets() called directly. This should be handled by PromptVariableGenerator.")
        return "Agent-level speech snippets placeholder."

    def get_short_term_memory(self) -> str:
        """Placeholder: This logic should be in PromptVariableGenerator if used by prompts."""
        self.logger.warning("Agent.get_short_term_memory() called directly. This should be handled by PromptVariableGenerator.")
        return "Agent-level STM placeholder."

    def promptize_vote_history(self) -> str:
        """Placeholder: This logic should be in PromptVariableGenerator if used by prompts."""
        self.logger.warning("Agent.promptize_vote_history() called directly. This should be handled by PromptVariableGenerator.")
        if self.vote_history:
            return f"Your vote history: {len(self.vote_history)} votes cast."
        return "No votes cast yet by you."

    def promptize_voting_results_history(self) -> str:
        """Placeholder: This logic should be in PromptVariableGenerator if used by prompts."""
        self.logger.warning("Agent.promptize_voting_results_history() called directly. This should be handled by PromptVariableGenerator.")
        if self.env.votingHistory:
            return f"Overall voting history: {len(self.env.votingHistory)} rounds recorded."
        return "No overall voting history available."
