"""
Discussion behavior mixin for Agent class.

This mixin handles all discussion-related functionality including:
- Discussion participation with tool calling
- Message generation and formatting
- Group interaction logic
"""

import logging
from typing import Optional, Dict, List, Any

logger = logging.getLogger("conclave.agents.discussion")

class DiscussionMixin:
    """Mixin providing discussion behavior for agents."""
    
    def discuss(self, current_discussion_group_ids: Optional[List[int]] = None, current_discussion_round: Optional[int] = None) -> Optional[Dict]:
        """
        Generate a discussion contribution.
        Variables are now primarily sourced via PromptVariableGenerator.
        `current_discussion_group_ids` is provided by the environment for the current sub-group.
        """
        if self.env.predefined_groups_enabled and hasattr(self.env, 'is_candidate') and self.env.is_candidate(self.agent_id):
            self.role_tag = "CANDIDATE"
            prompt_name_to_use = "discussion_candidate"
        else:
            self.role_tag = "ELECTOR"
            prompt_name_to_use = "discussion_elector"

        extra_vars_for_prompt = {
            'discussion_min_words': self.config.simulation.discussion_length.min_words,
            'discussion_max_words': self.config.simulation.discussion_length.max_words,
        }
        
        prompt = self.prompt_variable_generator.generate_prompt(
            prompt_name=prompt_name_to_use,
            agent_id=self.agent_id,
            discussion_group_ids=current_discussion_group_ids,
            discussion_round=current_discussion_round,
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
