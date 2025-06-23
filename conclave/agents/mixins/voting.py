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
        """Cast a vote using the new variable-based voting prompts."""
        try:
            # Determine role and prompt based on whether the agent is a candidate
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
            
            if result.success and result.arguments:
                cardinal_id_str = result.arguments.get("vote_cardinal_id")
                
                if cardinal_id_str is not None:
                    # Handle abstention
                    if str(cardinal_id_str) == '0':
                        self.logger.info(f"{self.name} abstained from voting in round {current_round}")
                        self.vote_history.append({
                            "vote": 0,  # Representing abstention
                            "reasoning": "Agent chose to abstain.",
                            "round": current_round
                        })
                        return 0

                    try:
                        # Find the agent whose cardinal_id matches the vote
                        agent_to_vote_for_id = None
                        voted_cardinal_obj = None

                        for agent_in_env in self.env.agents:
                            if str(getattr(agent_in_env, 'cardinal_id', '-1')) == str(cardinal_id_str):
                                agent_to_vote_for_id = agent_in_env.agent_id
                                voted_cardinal_obj = agent_in_env
                                break
                        
                        if agent_to_vote_for_id is None:
                            available_cardinals = [str(getattr(agent, 'cardinal_id', ag_idx)) for ag_idx, agent in enumerate(self.env.agents)]
                            raise AgentError(
                                f"Cardinal ID '{cardinal_id_str}' not found among loaded agents. Available: {available_cardinals}",
                                agent_id=str(self.agent_id),
                                agent_name=self.name
                            )

                        if not self.env.is_valid_vote_candidate(agent_to_vote_for_id):
                            raise AgentError(
                                f"Cardinal ID '{cardinal_id_str}' (Agent ID: {agent_to_vote_for_id}) is not eligible to receive votes",
                                agent_id=str(self.agent_id),
                                agent_name=self.name
                            )

                        self.logger.debug(f"Valid Cardinal ID '{cardinal_id_str}' (Agent {agent_to_vote_for_id}) selected by {self.name}")
                        
                        self.vote_history.append({
                            "vote": agent_to_vote_for_id, 
                            "reasoning": f"Voted for Cardinal {cardinal_id_str} based on internal stance and voting prompt analysis.",
                            "round": current_round
                        })

                        voted_candidate_name = voted_cardinal_obj.name if voted_cardinal_obj else f"Agent {agent_to_vote_for_id}"
                        self.logger.info(f"{self.name} voted for Cardinal {cardinal_id_str} - {voted_candidate_name} in round {current_round}")
                        return agent_to_vote_for_id
                        
                    except (ValueError, TypeError) as e:
                        raise AgentError(
                            f"Invalid Cardinal ID '{cardinal_id_str}' provided",
                            agent_id=str(self.agent_id),
                            agent_name=self.name,
                            context={"cardinal_id": cardinal_id_str, "original_error": str(e)}
                        )
                else:
                    raise AgentError(
                        "Failed to provide a Cardinal ID in vote",
                        agent_id=str(self.agent_id),
                        agent_name=self.name
                    )
            else:
                error_msg = result.error if result.error else "Unknown tool calling failure."
                raise LLMError(
                    f"Tool calling failed during voting: {error_msg}",
                    tool_name="cast_vote"
                )

        except (AgentError, LLMError):
            # Re-raise our custom exceptions
            raise
        except Exception as e:
            log_error_with_context(self.logger, e, f"Error in {self.name} voting process (round {current_round})")
            return None
