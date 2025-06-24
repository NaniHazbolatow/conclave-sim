"""
Voting behavior mixin for Agent class.

This mixin handles all voting-related functionality including:
- Vote casting with tool calling
- Vote validation and error handling
- Vote history tracking
"""

from typing import Optional, Dict, Any
from conclave.utils import get_logger, AgentError, LLMError, log_error_with_context

logger = get_logger("conclave.agents.voting")

class VotingMixin:
    """Mixin providing voting behavior for agents."""
    


    def cast_vote(self, current_round: int) -> Optional[int]:
        """Cast a vote using the new variable-based voting prompts (now logs and stores reasoning)."""
        try:
            if self.agent_id in self.env.candidate_ids:
                self.role_tag = "CANDIDATE"
                prompt_name = "voting_candidate"
            else:
                self.role_tag = "ELECTOR"
                prompt_name = "voting_elector"

            voting_prompt = self.prompt_variable_generator.generate_prompt(
                prompt_name=prompt_name,
                agent_id=self.agent_id,
            )

            self.logger.debug(f"LLM Request: Voting prompt for {self.name} (Role: {self.role_tag}): {voting_prompt}")

            cast_vote_tool_def = self.prompt_loader.get_tool_definition("cast_vote")
            if not cast_vote_tool_def:
                raise AgentError(
                    "Tool definition for 'cast_vote' not found in prompts.yaml",
                    agent_id=str(self.agent_id),
                    agent_name=self.name
                )

            tools = [cast_vote_tool_def]
            messages = [{"role": "user", "content": voting_prompt}]
            result = self.tool_caller.call_tool(messages, tools, tool_choice="cast_vote")

            self.logger.debug(f"=== VOTE TOOL CALLING RESULT FOR {self.name} ====")
            self.logger.debug(f"Success: {result.success}")
            self.logger.debug(f"Arguments: {result.arguments}")
            self.logger.debug(f"Error: {result.error}")
            self.logger.debug(f"=== END VOTE RESULT ====")


            def extract_vote_candidate_id_and_reasoning(tool_result):
                # Accepts int, str, or dict and returns (agent_id, reasoning)
                candidate_cardinal_id = 0
                reasoning = None
                if isinstance(tool_result, dict):
                    for key in ["vote_cardinal_id", "candidate_id", "id", "value"]:
                        if key in tool_result:
                            candidate_cardinal_id = tool_result[key]
                            break
                    reasoning = tool_result.get("reasoning")
                else:
                    candidate_cardinal_id = tool_result
                try:
                    candidate_cardinal_id = int(candidate_cardinal_id)
                except Exception:
                    candidate_cardinal_id = 0
                # Map Cardinal_ID to agent_id if possible
                agent_id = None
                if candidate_cardinal_id == 0:
                    agent_id = 0  # Abstain
                elif hasattr(self.env, "cardinal_id_to_agent_id_map") and candidate_cardinal_id in getattr(self.env, "cardinal_id_to_agent_id_map", {}):
                    agent_id = self.env.cardinal_id_to_agent_id_map[candidate_cardinal_id]
                else:
                    # If already an agent_id (e.g. fallback), use as is
                    agent_id = candidate_cardinal_id
                return agent_id, reasoning

            if result.success and result.arguments:
                candidate_id, reasoning = extract_vote_candidate_id_and_reasoning(result.arguments)
                # Store in vote history
                vote_record = {
                    "round": current_round,
                    "candidate_id": candidate_id,
                    "reasoning": reasoning,
                }
                if hasattr(self, "vote_history") and isinstance(self.vote_history, list):
                    self.vote_history.append(vote_record)
                self.logger.info(f"{self.name} voted for {candidate_id} in round {current_round}. Reasoning: {reasoning}")
                return candidate_id
            else:
                error_msg = result.error if result.error else "Unknown tool calling failure."
                raise LLMError(
                    f"Tool calling failed during voting: {error_msg}",
                    tool_name="cast_vote"
                )

        except (AgentError, LLMError):
            raise
        except Exception as e:
            log_error_with_context(self.logger, e, f"Error in {self.name} voting process (round {current_round})")
            return None


    # _execute_cast_vote removed for simplicity
