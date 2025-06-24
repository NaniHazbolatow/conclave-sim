import logging
from typing import Dict, Any

logger = logging.getLogger(__name__)

class ToolExecutor:
    """Executes tools on behalf of an agent."""

    def __init__(self, agent):
        """Initializes the ToolExecutor with a reference to the agent."""
        self.agent = agent

    def execute(self, function_name: str, arguments: Dict[str, Any]) -> Any:
        """
        Executes a tool by name with the provided arguments.
        It looks for a method on the agent with the pattern `_execute_{function_name}`.
        """
        tool_method_name = f"_execute_{function_name}"
        tool_method = getattr(self.agent, tool_method_name, None)

        if callable(tool_method):
            logger.info(f"Agent {self.agent.name} executing tool '{function_name}' with arguments: {arguments}")
            try:
                return tool_method(**arguments)
            except Exception as e:
                logger.error(f"Error executing tool '{function_name}' for agent {self.agent.name}: {e}", exc_info=True)
                return {"status": "error", "error": f"Error executing tool '{function_name}': {e}"}
        else:
            logger.error(f"Tool '{function_name}' not found on agent {self.agent.name}. Looking for method '{tool_method_name}'.")
            return {"status": "error", "error": f"Tool '{function_name}' not found."}
