from typing import TYPE_CHECKING, Dict, List, Optional, Any # Added Any
if TYPE_CHECKING:
    from ..environments.conclave_env import ConclaveEnv # Moved import under TYPE_CHECKING
import json
import re
# from typing import Dict, List, Optional # Duplicate import removed
import logging
import os
import threading
import datetime # Added import for datetime
from config.scripts.adapter import ConfigAdapter # Reverted to absolute import as relative failed
from ..prompting import get_prompt_loader # Corrected import to use prompting
from ..prompting.prompt_variable_generator import PromptVariableGenerator # Added import

logger = logging.getLogger("conclave.agents") # Changed from __name__

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
    
    def get_client(self, config_adapter): # Changed parameter name for clarity
        """Get shared LLM client instance based on configuration."""
        # Access the actual Pydantic config model via config_adapter.config
        llm_config = config_adapter.config.agent.llm
        
        if llm_config.backend == 'local': # Use attribute access
            with self._lock:  # Thread-safe client creation
                if self._local_client is None:
                    from ..llm.client import HuggingFaceClient
                    # Use the adapter's method to get pre-processed kwargs if available,
                    # or construct from the Pydantic model directly.
                    # Assuming get_llm_client_kwargs is still useful or adapt.
                    client_kwargs = config_adapter.get_llm_client_kwargs() 
                    client_kwargs = {k: v for k, v in client_kwargs.items() if v is not None}
                    self._local_client = HuggingFaceClient(**client_kwargs)
                    print(f"Created shared local LLM client: {self._local_client.model_name}")
                return self._local_client
        else: # 'remote' backend
            from ..llm.client import RemoteLLMClient
            client_kwargs = config_adapter.get_llm_client_kwargs()
            client_kwargs = {k: v for k, v in client_kwargs.items() if v is not None}
            return RemoteLLMClient(**client_kwargs)

# Global shared manager instance
_shared_llm_manager = SharedLLMManager()

class Agent:
    def __init__(self, 
                 agent_id: int, 
                 conclave_env: 'ConclaveEnv', 
                 name: str, 
                 personality: str, 
                 initial_stance: str, 
                 party_loyalty: float, 
                 agent_type: str = "default"):
        self.agent_id = agent_id
        self.conclave_env = conclave_env # Reference to the environment
        # self.config_adapter = get_config() # Get the ConfigAdapter instance
        self.config_adapter = ConfigAdapter() # Instantiate ConfigAdapter
        self.name = name
        
        self.env = conclave_env
        
        self.cardinal_id = str(agent_id)  # Directly using agent_id as string
        self.internal_persona = initial_stance  # Assuming this is the intended mapping
        self.public_profile = "Not set"
        self.profile_blurb = "N/A"
        self.persona_tag = "N/A"
        
        self.vote_history = []
        self.logger = logging.getLogger(f"conclave.agents.{self.name.replace(' ', '_')}")
        
        self.role_tag = "ELECTOR"    
        
        self.internal_stance = None  
        self.stance_history = []     
        self.last_stance_update: Optional[datetime.datetime] = None # Ensure type hint
        self.current_reflection_digest: Optional[str] = "Reflection not yet generated." # Initialize explicitly
        self.last_reflection_round: int = -1
        self.last_reflection_timestamp: Optional[datetime.datetime] = None
        
        self.config = self.config_adapter.config # Access the root Pydantic config model
        
        self.prompt_loader = get_prompt_loader()
        self.prompt_variable_generator = PromptVariableGenerator(env=self.env, prompt_loader=self.prompt_loader) 
        
        try:
            # Pass the ConfigAdapter instance to the shared manager
            self.llm_client = _shared_llm_manager.get_client(self.config_adapter) 
            
            from ..llm.robust_tools import RobustToolCaller
            # Removed config_adapter from RobustToolCaller constructor
            self.tool_caller = RobustToolCaller(self.llm_client, self.logger) 
            
            self.logger.info(f"Agent {self.name} initialized with {self.config.agent.llm.backend} backend, model: {self.llm_client.model_name}") # Corrected config access
            
        except Exception as e:
            self.logger.error(f"Failed to initialize LLM client for agent {self.name}: {e}")
            raise

    def cast_vote(self, current_round: int) -> Optional[int]: # Changed return type
        """Cast a vote using the new variable-based voting prompts."""
        # Determine role and prompt based on whether the agent is a candidate
        # This check needs to access environment state about candidates.
        # Assuming self.env.candidate_ids or a similar attribute holds current candidates.
        if self.agent_id in self.env.candidate_ids: # Check if self is a candidate
            self.role_tag = "CANDIDATE"
            prompt_name = "voting_candidate"
        else:
            self.role_tag = "ELECTOR"
            prompt_name = "voting_elector"
        
        voting_prompt = self.prompt_variable_generator.generate_prompt(
            prompt_name=prompt_name,
            agent_id=self.agent_id,
            # current_round is now part of agent_variables, but can be overridden if needed
            # current_round=current_round # Pass current_round if prompt needs it explicitly beyond agent_variables
        )
        
        self.logger.debug(f"LLM Request: Voting prompt for {self.name} (Role: {self.role_tag}): {voting_prompt}")
        
        cast_vote_tool_def = self.prompt_loader.get_tool_definition("cast_vote")
        if not cast_vote_tool_def:
            self.logger.error(f"Tool definition for 'cast_vote' not found in prompts.yaml for agent {self.name}")
            # raise ValueError("Tool definition for 'cast_vote' not found.")
            return None # Return None on failure

        tools = [cast_vote_tool_def]
        
        try:
            messages = [{"role": "user", "content": voting_prompt}]
            result = self.tool_caller.call_tool(messages, tools, tool_choice="cast_vote")
            
            self.logger.debug(f"=== VOTE TOOL CALLING RESULT FOR {self.name} ====")
            self.logger.debug(f"Success: {result.success}")
            self.logger.debug(f"Arguments: {result.arguments}")
            self.logger.debug(f"Error: {result.error}")
            self.logger.debug(f"=== END VOTE RESULT ====")
            
            if result.success and result.arguments:
                cardinal_id_str = result.arguments.get("vote_cardinal_id") # vote_cardinal_id is likely a string

                if cardinal_id_str is not None:
                    try:
                        # Attempt to find the agent whose cardinal_id (string) matches the vote
                        agent_to_vote_for_id = None
                        voted_cardinal_obj = None # Store the agent object to get name later

                        for agent_in_env in self.env.agents:
                            # Ensure comparison is robust (e.g. string to string)
                            if str(getattr(agent_in_env, 'cardinal_id', '-1')) == str(cardinal_id_str):
                                agent_to_vote_for_id = agent_in_env.agent_id
                                voted_cardinal_obj = agent_in_env
                                break
                        
                        if agent_to_vote_for_id is None:
                            available_cardinals = [str(getattr(agent, 'cardinal_id', ag_idx)) for ag_idx, agent in enumerate(self.env.agents)]
                            # raise ValueError(f"Cardinal ID '{cardinal_id_str}' not found among loaded agents. Available: {available_cardinals}")
                            self.logger.error(f"Cardinal ID '{cardinal_id_str}' not found among loaded agents for {self.name}. Available: {available_cardinals}")
                            return None # Return None on failure

                        if not self.env.is_valid_vote_candidate(agent_to_vote_for_id):
                            # raise ValueError(f"Cardinal ID '{cardinal_id_str}' (Agent ID: {agent_to_vote_for_id}) is not eligible to receive votes.")
                            self.logger.error(f"Cardinal ID '{cardinal_id_str}' (Agent ID: {agent_to_vote_for_id}) is not eligible to receive votes for {self.name}.")
                            return None # Return None on failure

                        self.logger.debug(f"Valid Cardinal ID '{cardinal_id_str}' (Agent {agent_to_vote_for_id}) selected by {self.name}")
                        
                        self.vote_history.append({
                            "vote": agent_to_vote_for_id, 
                            "reasoning": f"Voted for Cardinal {cardinal_id_str} based on internal stance and voting prompt analysis.",
                            "round": current_round # Log the round
                        })

                        # self.env.cast_vote(agent_to_vote_for_id, self.agent_id) # This was causing an error, env does not have cast_vote
                        # The environment's vote tally is updated by run_voting_round based on agent.vote_history
                        voted_candidate_name = voted_cardinal_obj.name if voted_cardinal_obj else f"Agent {agent_to_vote_for_id}"
                        self.logger.info(f"{self.name} voted for Cardinal {cardinal_id_str} - {voted_candidate_name} in round {current_round}")
                        return agent_to_vote_for_id # Return the agent_id of the voted candidate
                        
                    except (ValueError, TypeError) as e:
                        self.logger.error(f"Invalid Cardinal ID '{cardinal_id_str}' provided by {self.name}: {e}")
                        # self.env.cast_vote(-1, self.agent_id) # Record abstention/failure
                        return None # Return None on failure
                else:
                    self.logger.error(f"{self.name} failed to provide a Cardinal ID in vote.")
                    # self.env.cast_vote(-1, self.agent_id) # Record abstention/failure
                    return None # Return None on failure
            else:
                error_msg = result.error if result.error else "Unknown tool calling failure."
                self.logger.error(f"Tool calling failed for {self.name} during voting: {error_msg}")
                # self.env.cast_vote(-1, self.agent_id) # Record abstention/failure
                return None # Return None on failure

        except Exception as e:
            self.logger.error(f"Error in {self.name} voting process (round {current_round}): {e}", exc_info=True)
            # self.env.cast_vote(-1, self.agent_id) # Record abstention/failure
            return None # Return None on failure

    def discuss(self, current_discussion_group_ids: Optional[List[int]] = None, current_discussion_round: Optional[int] = None) -> Optional[Dict]: # Added current_discussion_round
        """
        Generate a discussion contribution.
        Variables are now primarily sourced via PromptVariableGenerator.
        `current_discussion_group_ids` is provided by the environment for the current sub-group.
        """
        if self.env.testing_groups_enabled and hasattr(self.env, 'is_candidate') and self.env.is_candidate(self.agent_id):
            self.role_tag = "CANDIDATE"
            prompt_name_to_use = "discussion_candidate"
        else:
            self.role_tag = "ELECTOR"
            prompt_name_to_use = "discussion_elector"

        extra_vars_for_prompt = {
            'discussion_min_words': self.config.simulation.discussion_length.min_words,
            'discussion_max_words': self.config.simulation.discussion_length.max_words,
            # discussion_round is now passed to generate_prompt directly if needed by the template
        }
        
        prompt = self.prompt_variable_generator.generate_prompt(
            prompt_name=prompt_name_to_use,
            agent_id=self.agent_id,
            discussion_group_ids=current_discussion_group_ids,
            discussion_round=current_discussion_round, # Pass current_discussion_round
            **extra_vars_for_prompt
        )
        
        self.logger.debug(f"LLM Request: Discussion prompt for {self.name} (Role: {self.role_tag}):\\n{prompt}")

        speak_message_tool_def = self.prompt_loader.get_tool_definition("speak_message")
        if not speak_message_tool_def:
            self.logger.error(f"Tool definition for 'speak_message' not found in prompts.yaml for agent {self.name}")
            raise ValueError("Tool definition for 'speak_message' not found.")
        
        tools = [speak_message_tool_def]

        try:
            messages = [{"role": "user", "content": prompt}]
            self.logger.info(f"DISCUSSION TOOL CALL INITIATED for {self.name} (ID: {self.agent_id})")
            result = self.tool_caller.call_tool(messages, tools, tool_choice="speak_message")
            self.logger.info(f"DISCUSSION TOOL CALL COMPLETED for {self.name} - Success: {result.success}, Strategy: {result.strategy_used}")
            
            if result.success and result.arguments:
                # Debug logging to understand what we're getting
                self.logger.info(f"DEBUG: result.arguments type: {type(result.arguments)}, value: {result.arguments}")
                
                if isinstance(result.arguments, dict):
                    message = result.arguments.get("message", "")
                elif isinstance(result.arguments, str):
                    # Handle case where arguments is a string (fallback parsing)
                    message = result.arguments
                else:
                    self.logger.warning(f"Unexpected arguments type: {type(result.arguments)}")
                    message = str(result.arguments)
                
                if not message.strip():
                    self.logger.warning(f"{self.name} ({self.agent_id}) provided an empty message.")
                    message = "(Agent provided no message)"

                self.logger.info(f"{self.name} ({self.agent_id}) speaks:\\n{message}")
                return {"agent_id": self.agent_id, "message": message}
            else:
                error_msg = result.error if result.error else "Unknown tool calling failure."
                self.logger.warning(f"Discussion tool calling failed for {self.name}: {error_msg}")
                return {"agent_id": self.agent_id, "message": f"(Tool calling failed: {error_msg})"}

        except Exception as e:
            self.logger.error(f"Error in Agent {self.name} ({self.agent_id}) discussion: {e}", exc_info=True)
            return {"agent_id": self.agent_id, "message": f"(Discussion error: {e})"}

    def generate_internal_stance(self) -> str:
        """
        Generate an internal stance using LLM with stance generation prompt and tool.
        Variables are sourced via PromptVariableGenerator.
        """
        self.logger.info(f"Generating internal stance for {self.name} (V{self.env.votingRound}.D{self.env.discussionRound})")
        
        # Ensure reflection_digest is correctly picked up by PromptVariableGenerator
        # from self.current_reflection_digest
        stance_prompt = self.prompt_variable_generator.generate_prompt(
            prompt_name="stance",
            agent_id=self.agent_id
        )
        
        self.logger.debug(f"LLM Request: Stance prompt for {self.name}:\\n{stance_prompt}")
        
        generate_stance_tool_def = self.prompt_loader.get_tool_definition("generate_stance")
        if not generate_stance_tool_def:
            self.logger.error(f"Tool definition for 'generate_stance' not found in prompts.yaml for agent {self.name}")
            raise ValueError("Tool definition for 'generate_stance' not found.")
        tools = [generate_stance_tool_def]
        
        try:
            messages = [{"role": "user", "content": stance_prompt}]
            result = self.tool_caller.call_tool(messages, tools, tool_choice="generate_stance")
            
            self.logger.debug(f"=== STANCE TOOL CALLING RESULT FOR {self.name} ====")
            self.logger.debug(f"Success: {result.success}")
            self.logger.debug(f"Arguments: {result.arguments}")
            self.logger.debug(f"Error: {result.error}")
            self.logger.debug(f"=== END STANCE RESULT ====")
            
            if result.success and result.arguments:
                stance = result.arguments.get("stance", "").strip()
                
                if stance:
                    timestamp = datetime.datetime.now()
                    self.internal_stance = stance
                    self.last_stance_update = timestamp
                    self.stance_history.append({
                        "timestamp": timestamp, 
                        "stance": self.internal_stance, 
                        "status": "generated",
                        "voting_round": self.env.votingRound, 
                        "discussion_round": self.env.discussionRound, # Record current env discussion round
                        "reflection_used": self.current_reflection_digest if self.last_reflection_timestamp and self.last_reflection_timestamp <= timestamp else "N/A"
                    })
                    self.logger.info(f"Successfully generated stance for {self.name} (informed by reflection round {self.last_reflection_round if self.last_reflection_timestamp and self.last_reflection_timestamp <= timestamp else 'N/A'}): {stance[:100]}{'...' if len(stance) > 100 else ''}")
                    return self.internal_stance
                else:
                    self.logger.warning(f"Empty stance returned for {self.name}")
                    
            error_msg = result.error if result.error else "Unknown error in stance generation"
            self.logger.error(f"Failed to generate stance for {self.name}: {error_msg}")
            
            timestamp = datetime.datetime.now()
            self.internal_stance = f"[Stance generation failed for {self.name}]" # Keep a placeholder
            self.last_stance_update = timestamp # Still update timestamp to prevent immediate re-try loops on persistent failure
            self.stance_history.append({
                "timestamp": timestamp, 
                "stance": self.internal_stance, 
                "status": "failed",
                "voting_round": self.env.votingRound, 
                "discussion_round": self.env.discussionRound,
                "error": error_msg
            })
            return self.internal_stance
            
        except Exception as e:
            self.logger.error(f"Error in stance generation for {self.name}: {e}", exc_info=True)
            timestamp = datetime.datetime.now()
            self.internal_stance = f"[Stance generation error for {self.name}]" # Keep a placeholder
            self.last_stance_update = timestamp # Still update timestamp
            self.stance_history.append({
                "timestamp": timestamp, 
                "stance": self.internal_stance, 
                "status": "error",
                "voting_round": self.env.votingRound, 
                "discussion_round": self.env.discussionRound,
                "error": str(e)
            })
            return self.internal_stance

    def get_internal_stance(self) -> str:
        """
        Get the current internal stance, generating one if it doesn't exist or needs update.
        """
        if self.should_update_stance(): 
            self.logger.info(f"Updating stance for {self.name} (V{self.env.votingRound}.D{self.env.discussionRound}). Last Stance Update: {self.last_stance_update}, Last Reflection: {self.last_reflection_timestamp}")
            self.generate_internal_stance()
        elif self.internal_stance is None: 
            self.logger.info(f"No stance found for {self.name}, generating initial stance.")
            self.generate_internal_stance()
            
        return self.internal_stance if self.internal_stance is not None else ""

    def should_update_stance(self) -> bool:
        """
        Determine if the agent should update their internal stance.
        Updates if:
        1. No current stance or no record of when it was last updated.
        2. A new reflection has been generated since the last stance update.
        3. The game has progressed to a new voting or discussion round since the last stance update.
        """
        if self.internal_stance is None or self.last_stance_update is None:
            self.logger.debug(f"Stance update needed for {self.name}: No current stance or no last_stance_update time.")
            return True

        # Condition 1: A new reflection has occurred since the last stance update.
        if self.last_reflection_timestamp is not None and self.last_reflection_timestamp > self.last_stance_update:
            self.logger.debug(f"Stance update needed for {self.name}: New reflection available (Ref_Time: {self.last_reflection_timestamp}, Stance_Time: {self.last_stance_update}).")
            return True

        # Condition 2: General game progression (new voting/discussion round).
        if self.stance_history:
            last_stance_info = self.stance_history[-1]
            # Use -1 or a very early round if keys are missing, to ensure comparison works
            last_stance_voting_round = last_stance_info.get('voting_round', -1)
            last_stance_discussion_round = last_stance_info.get('discussion_round', -1)
            
            current_voting_round = self.env.votingRound
            current_discussion_round = self.env.discussionRound 

            if (current_voting_round, current_discussion_round) > (last_stance_voting_round, last_stance_discussion_round):
                self.logger.debug(f"Stance update needed for {self.name}: Game progressed. Current (V{current_voting_round},D{current_discussion_round}) > Last Stance (V{last_stance_voting_round},D{last_stance_discussion_round}).")
                return True
        else:
            # No stance history, but self.internal_stance might exist (e.g. loaded, or history cleared).
            # If there's any game activity, and we have a stance but no history for it,
            # it's safer to regenerate to align with current game state if not covered by reflection check.
            if self.env.votingRound > 0 or self.env.discussionRound > 0:
                # This case implies an existing stance without history, and game has started.
                # The reflection check (Condition 1) should ideally cover this if a reflection happened.
                # This is a fallback.
                self.logger.debug(f"Stance update potentially needed for {self.name}: No stance history, but game activity detected (V{self.env.votingRound},D{self.env.discussionRound}). Triggering to be safe.")
                return True
        
        self.logger.debug(f"Stance update NOT needed for {self.name} (V{self.env.votingRound}.D{self.env.discussionRound}). Last Stance Update: {self.last_stance_update}, Last Reflection: {self.last_reflection_timestamp}")
        return False

    def reflect_on_discussion(self, 
                              group_analysis_summary: Optional[str], 
                              group_transcript: str, 
                              current_discussion_round: int):
        """
        Generates a reflection digest based on the group summary, transcript, and agent's last utterance.
        Stores the digest in self.current_reflection_digest and returns it.
        Uses the 'discussion_reflection' tool.

        Args:
            group_analysis_summary: The analysis summary for the agent's discussion group.
            group_transcript: The full transcript of the agent's discussion group.
            current_discussion_round: The discussion round number for which reflection is being done.
        """
        logger.debug(f"Agent {self.name} (ID: {self.agent_id}) entering reflect_on_discussion for D_Round {current_discussion_round}.")
        logger.debug(f"Agent {self.name} received group_analysis_summary: {'Present' if group_analysis_summary else 'Absent'}")
        
        allow_reflection_config = self.config.agent.allow_reflection_without_summary # Corrected config access
        logger.debug(f"Agent {self.name} config allow_reflection_without_summary: {allow_reflection_config}")

        if not group_analysis_summary and not allow_reflection_config:
            logger.info(f"Agent {self.name} (ID: {self.agent_id}) skipping reflection as no group summary is available and allow_reflection_without_summary is false.")
            self.current_reflection_digest = "Reflection skipped: No group summary and configuration disallows."
            self.last_reflection_timestamp = datetime.datetime.now() # Still update timestamp
            self.last_reflection_round = current_discussion_round
            logger.debug(f"Agent {self.name} (ID: {self.agent_id}) exiting reflect_on_discussion (skipped).")
            return self.current_reflection_digest

        self.logger.info(f"Agent {self.name} reflecting on discussion for D_Round {current_discussion_round}.")

        # Get agent's last utterance from the provided transcript or history
        # For simplicity, we'll assume prompt_variable_generator can fetch this if needed,
        # or it can be passed if critical. Here, we rely on the prompt template to guide the agent.
        # agent_last_utterance = self.prompt_variable_generator.generate_agent_last_utterance(self.agent_id)
        # The 'discussion_reflection' prompt itself should guide the agent to consider their contributions.

        # Prepare variables for the prompt using PromptVariableGenerator for consistency
        # The prompt template 'discussion_reflection' will define what variables it needs.
        # We ensure 'group_analysis_summary' and 'group_transcript' are available.
        
        reflection_prompt = self.prompt_variable_generator.generate_prompt(
            prompt_name="discussion_reflection",
            agent_id=self.agent_id,
            discussion_round=current_discussion_round, # Pass current_discussion_round
            # Explicitly pass the necessary context for the reflection prompt
            # These will be available in the prompt template via .format()
            # The prompt template itself should reference {{group_analysis_summary}} and {{group_transcript}}
            # and other variables generated by generate_agent_variables.
            group_analysis_summary_text=group_analysis_summary if group_analysis_summary else "No analysis summary was provided for your group.",
            group_transcript_text=group_transcript if group_transcript else "No discussion transcript was provided for your group."
        )
        
        self.logger.debug(f"LLM Request: Reflection prompt for {self.name} (D_Round {current_discussion_round}):\\\\n{reflection_prompt}")

        reflection_tool_def = self.prompt_loader.get_tool_definition("discussion_reflection")
        if not reflection_tool_def:
            self.logger.error(f"Tool definition for 'discussion_reflection' not found for agent {self.name}.")
            self.current_reflection_digest = "Reflection failed: Tool definition missing."
            self.last_reflection_timestamp = datetime.datetime.now()
            self.last_reflection_round = current_discussion_round
            return self.current_reflection_digest
        
        tools = [reflection_tool_def]

        try:
            messages = [{"role": "user", "content": reflection_prompt}]
            result = self.tool_caller.call_tool(messages, tools, tool_choice="discussion_reflection")
            
            self.logger.debug(f"=== REFLECTION TOOL CALLING RESULT FOR {self.name} ====")
            self.logger.debug(f"Success: {result.success}")
            self.logger.debug(f"Arguments: {result.arguments}")
            self.logger.debug(f"Error: {result.error}")
            self.logger.debug(f"=== END REFLECTION RESULT ====")

            if result.success and result.arguments:
                reflection_text = result.arguments.get("reflection_digest", "").strip()
                if reflection_text:
                    self.current_reflection_digest = reflection_text
                    self.logger.info(f"Agent {self.name} (ID: {self.agent_id}) successfully reflected. Digest: {self.current_reflection_digest[:100]}...")
                else:
                    self.current_reflection_digest = "Reflection generated an empty digest."
                    self.logger.warning(f"Agent {self.name} (ID: {self.agent_id}) generated an empty reflection digest.")
            else:
                error_msg = result.error if result.error else "Unknown tool calling failure during reflection."
                self.current_reflection_digest = f"Reflection failed: {error_msg}"
                self.logger.error(f"Tool calling failed for {self.name} during reflection: {error_msg}")
        
        except Exception as e:
            self.logger.error(f"Error during reflection for agent {self.name} (ID: {self.agent_id}): {e}", exc_info=True)
            self.current_reflection_digest = f"Reflection error: {e}"
        
        self.last_reflection_timestamp = datetime.datetime.now()
        self.last_reflection_round = current_discussion_round # Record which discussion round this reflection pertains to
        logger.debug(f"Agent {self.name} (ID: {self.agent_id}) exiting reflect_on_discussion. Digest: {self.current_reflection_digest[:100]}...")
        return self.current_reflection_digest

    def get_last_stance(self) -> str:
        """Get the most recent internal stance text."""
        return self.internal_stance if self.internal_stance else "No previous stance recorded."

    def load_persona_from_data(self, cardinal_data: Dict):
        """Load persona details from provided data (e.g., a row from CSV)."""
        self.internal_persona = cardinal_data.get('Internal_Persona', '') # Corrected key
        self.public_profile = cardinal_data.get('Public_Profile', '')   # Corrected key
        self.name = cardinal_data.get('name', self.name)
        self.cardinal_id = cardinal_data.get('Cardinal_ID', self.agent_id) # Corrected key
        self.logger.info(f"Loaded persona for {self.name} (Cardinal ID: {self.cardinal_id})")

    def update_agent_details(self, details: Dict):
        """Update agent details, e.g., from a central data source after init."""
        if 'internal_persona' in details: self.internal_persona = details['internal_persona'] 
        if 'public_profile' in details: self.public_profile = details['public_profile'] 
        if 'role_tag' in details: self.role_tag = details['role_tag']
        self.logger.debug(f"Agent {self.name} details updated.")

    def __repr__(self):
        return f"Agent(id={self.agent_id}, name='{self.name}', role='{self.role_tag}')"
