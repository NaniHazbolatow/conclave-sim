"""
Stance management mixin for Agent class.

This mixin handles all stance-related functionality including:
- Internal stance generation and updates
- Stance history tracking
- Stance update logic and timing
"""

import logging
import datetime
from typing import Optional

logger = logging.getLogger("conclave.agents.stance")
stance_logger = logging.getLogger("conclave.stances")

class StanceMixin:
    """Mixin providing stance management behavior for agents."""
    
    def generate_internal_stance(self) -> str:
        """
        Generate an internal stance using LLM with stance generation prompt and tool.
        Variables are sourced via PromptVariableGenerator.
        """
        self.logger.info(f"Generating internal stance for {self.name} (V{self.env.votingRound}.D{self.env.discussionRound})")
        
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
                self.tool_executor.execute("generate_stance", result.arguments)
            else:
                error_msg = result.error if result.error else "Unknown error in stance generation"
                self.logger.error(f"Failed to generate stance for {self.name}: {error_msg}")
                self._update_stance_history_with_error(error_msg)

            return self.internal_stance
            
        except Exception as e:
            self.logger.error(f"Error in stance generation for {self.name}: {e}", exc_info=True)
            self._update_stance_history_with_error(str(e))
            return self.internal_stance

    def _execute_generate_stance(self, stance: str):
        """Executes the generate_stance tool call."""
        if stance and isinstance(stance, str) and stance.strip():
            timestamp = datetime.datetime.now()
            self.internal_stance = stance.strip()
            self.last_stance_update = timestamp

            stance_logger.info({
                "round": self.env.votingRound,
                "agent_name": self.name,
                "agent_id": self.agent_id,
                "stance": self.internal_stance,
            })

            self.stance_history.append({
                "timestamp": timestamp, 
                "stance": self.internal_stance, 
                "status": "generated",
                "voting_round": self.env.votingRound, 
                "discussion_round": self.env.discussionRound,
                "reflection_used": self.current_reflection_digest if self.last_reflection_timestamp and self.last_reflection_timestamp <= timestamp else "N/A"
            })
            self.logger.info(f"Successfully generated stance for {self.name}: {self.internal_stance[:100]}...")
        else:
            self.logger.warning(f"Empty stance returned for {self.name}")
            self._update_stance_history_with_error("Empty stance returned")

    def _update_stance_history_with_error(self, error_msg: str):
        """Helper to update stance history with an error."""
        timestamp = datetime.datetime.now()
        self.internal_stance = f"[Stance generation failed for {self.name}]"
        self.last_stance_update = timestamp
        self.stance_history.append({
            "timestamp": timestamp, 
            "stance": self.internal_stance, 
            "status": "error",
            "voting_round": self.env.votingRound, 
            "discussion_round": self.env.discussionRound,
            "error": error_msg
        })

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
            last_stance_voting_round = last_stance_info.get('voting_round', -1)
            last_stance_discussion_round = last_stance_info.get('discussion_round', -1)
            
            current_voting_round = self.env.votingRound
            current_discussion_round = self.env.discussionRound 

            if (current_voting_round, current_discussion_round) > (last_stance_voting_round, last_stance_discussion_round):
                self.logger.debug(f"Stance update needed for {self.name}: Game progressed. Current (V{current_voting_round},D{current_discussion_round}) > Last Stance (V{last_stance_voting_round},D{last_stance_discussion_round}).")
                return True
        else:
            # No stance history, but self.internal_stance might exist
            if self.env.votingRound > 0 or self.env.discussionRound > 0:
                self.logger.debug(f"Stance update potentially needed for {self.name}: No stance history, but game activity detected (V{self.env.votingRound},D{self.env.discussionRound}). Triggering to be safe.")
                return True
        
        self.logger.debug(f"Stance update NOT needed for {self.name} (V{self.env.votingRound}.D{self.env.discussionRound}). Last Stance Update: {self.last_stance_update}, Last Reflection: {self.last_reflection_timestamp}")
        return False

    def get_last_stance(self) -> str:
        """Get the most recent internal stance text."""
        return self.internal_stance if self.internal_stance else "No previous stance recorded."
