"""
Reflection behavior mixin for Agent class.

This mixin handles all reflection-related functionality including:
- Discussion reflection generation
- Reflection digest management
- Integration with stance updates
"""

import logging
import datetime
from typing import Optional

logger = logging.getLogger("conclave.agents.reflection")

class ReflectionMixin:
    """Mixin providing reflection behavior for agents."""
    
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
        
        allow_reflection_config = self.config.agent.allow_reflection_without_summary
        logger.debug(f"Agent {self.name} config allow_reflection_without_summary: {allow_reflection_config}")

        if not group_analysis_summary and not allow_reflection_config:
            logger.info(f"Agent {self.name} (ID: {self.agent_id}) skipping reflection as no group summary is available and allow_reflection_without_summary is false.")
            self.current_reflection_digest = "Reflection skipped: No group summary and configuration disallows."
            self.last_reflection_timestamp = datetime.datetime.now()
            self.last_reflection_round = current_discussion_round
            logger.debug(f"Agent {self.name} (ID: {self.agent_id}) exiting reflect_on_discussion (skipped).")
            return self.current_reflection_digest

        self.logger.info(f"Agent {self.name} reflecting on discussion for D_Round {current_discussion_round}.")

        reflection_prompt = self.prompt_variable_generator.generate_prompt(
            prompt_name="discussion_reflection",
            agent_id=self.agent_id,
            discussion_round=current_discussion_round,
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
        self.last_reflection_round = current_discussion_round
        logger.debug(f"Agent {self.name} (ID: {self.agent_id}) exiting reflect_on_discussion. Digest: {self.current_reflection_digest[:100]}...")
        return self.current_reflection_digest
