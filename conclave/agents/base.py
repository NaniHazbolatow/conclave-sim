from ..environments.conclave_env import ConclaveEnv
import json
from typing import Dict, List, Optional
import logging
import os
import threading
from ..config.manager import get_config
from ..llm.client import UnifiedLLMClient

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
        
        # Get configuration
        self.config = get_config()
        
        # Use shared LLM client to avoid multiple model instances
        try:
            self.llm_client = _shared_llm_manager.get_client(self.config)
            self.logger.info(f"Agent {self.name} initialized with {self.config.get_backend_type()} backend, model: {self.llm_client.model_name}")
            
        except Exception as e:
            self.logger.error(f"Failed to initialize LLM client for agent {self.name}: {e}")
            raise

    def cast_vote(self) -> None:
        personal_vote_history = self.promptize_vote_history()
        ballot_results_history = self.promptize_voting_results_history()
        discussion_history = self.env.get_discussion_history(self.agent_id)
        prompt = f"""You are {self.name}. Here is some information about yourself: {self.background}
You are currently participating in the conclave to decide the next pope. The candidate that secures a 2/3 supermajority of votes wins.
The candidates are:
{self.env.list_candidates_for_prompt()}

{personal_vote_history}

{ballot_results_history}

{discussion_history}

Please vote for one of the candidates using the cast_vote tool. Make sure to include both your chosen candidate and a detailed explanation of why you chose them.
        """
        if (self.agent_id == 0):
            print(prompt)
        self.logger.info(prompt)
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
            response = self._invoke_claude(prompt, tools, tool_choice="cast_vote")

            # Handle tool call response
            if hasattr(response, 'tool_calls') and response.tool_calls:
                tool_call = response.tool_calls[0]
                if tool_call.function.name == 'cast_vote':
                    tool_input = json.loads(tool_call.function.arguments)
                    vote = tool_input.get("candidate")
                    reasoning = tool_input.get("explanation", "No explanation provided.")

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

        prompt = f"""You are {self.name}. Here is some information about yourself: {self.background}
You are currently participating in the conclave to decide the next pope. The candidate that secures a 2/3 supermajority of votes wins.
The candidates are:
{self.env.list_candidates_for_prompt()}

{personal_vote_history}

{ballot_results_history}

{discussion_history}

Based on the current state of the conclave, how urgently do you feel the need to speak?
Evaluate your desire to speak on a scale from 1-100, where:
1 = You have nothing important to add at this time
100 = You have an extremely urgent point that must be heard immediately

Consider factors such as:
- How strongly do you feel about supporting or opposing specific candidates?
- Do you need to respond to something said in a previous discussion?
- Do you have important information or perspectives that haven't been shared yet?
- Are the voting trends concerning to you?

Use the evaluate_speaking_urgency tool to provide your urgency score and reasoning.
        """

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
            response = self._invoke_claude(prompt, tools, tool_choice="evaluate_speaking_urgency")

            # Handle tool call response
            if hasattr(response, 'tool_calls') and response.tool_calls:
                tool_call = response.tool_calls[0]
                if tool_call.function.name == 'evaluate_speaking_urgency':
                    tool_input = json.loads(tool_call.function.arguments)
                    urgency_score = tool_input.get("urgency_score", 50)
                    reasoning = tool_input.get("reasoning", "No reasoning provided.")
                    
                    # Ensure score is in range 1-100
                    urgency_score = max(1, min(100, int(urgency_score)))
                    
                    return {
                        "agent_id": self.agent_id,
                        "urgency_score": urgency_score,
                        "reasoning": reasoning
                    }
                else:
                    raise ValueError("Invalid tool use")
            else:
                # Fallback if no tool call
                return {
                    "agent_id": self.agent_id,
                    "urgency_score": 50,
                    "reasoning": "No urgency evaluation received from AI"
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
            urgency_context = f"""You indicated that you have an urgency level of {urgency_data['urgency_score']}/100 to speak.
Your reasoning was: {urgency_data['reasoning']}

Keep this urgency level and reasoning in mind as you formulate your response.
"""

        prompt = f"""You are {self.name}. Here is some information about yourself: {self.background}
You are currently participating in the conclave to decide the next pope. The candidate that secures a 2/3 supermajority of votes wins.
The candidates are:
{self.env.list_candidates_for_prompt()}

{personal_vote_history}

{ballot_results_history}

{discussion_history}

{urgency_context}
It's time for a discussion round. Use the speak_message tool to contribute to the discussion.
Your goal is to influence others based on your beliefs and background. You can:
1. Make your case for a particular candidate
2. Question the qualifications of other candidates
3. Respond to previous speakers
4. Share your perspectives on what the Church needs

Be authentic to your character and background. Provide a meaningful contribution of 100-300 words.
        """

        # Define speak tool
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
                                "description": "Your contribution to the discussion (100-300 words)"
                            }
                        },
                        "required": ["message"]
                    }
                }
            }
        ]

        try:
            response = self._invoke_claude(prompt, tools, tool_choice="speak_message")

            # Handle tool call response
            if hasattr(response, 'tool_calls') and response.tool_calls:
                tool_call = response.tool_calls[0]
                if tool_call.function.name == 'speak_message':
                    tool_input = json.loads(tool_call.function.arguments)
                    message = tool_input.get("message", "")

                    # Log the discussion contribution
                    self.logger.info(f"{self.name} ({self.agent_id}) speaks:\n{message}")
                    print(f"\nCardinal {self.agent_id} - {self.name} speaks:\n{message}\n")

                    # Return the discussion contribution
                    return {
                        "agent_id": self.agent_id,
                        "message": message
                    }
                else:
                    raise ValueError("Invalid tool use")
            else:
                # Fallback if no tool call
                return {
                    "agent_id": self.agent_id,
                    "message": "No discussion contribution received from AI"
                }

        except Exception as e:
            # Log the error and return None if there's a problem
            self.logger.error(f"Error in LlmAgent {self.agent_id} discussion: {e}")
            return None

    def _invoke_claude(self, prompt: str, tools: List[Dict] = [], tool_choice: str = None) -> Dict:
        """Invoke LLM through the configured client."""
        try:
            # Create messages in OpenAI format
            messages = [{"role": "user", "content": prompt}]
            
            # Check if using remote backend (supports native tool calls)
            if self.config.get_backend_type() == 'remote' and tools:
                # Use native tool calling for remote backend
                response = self._call_remote_with_tools(messages, tools, tool_choice)
                return response
            else:
                # Use basic generation and parse JSON response manually for local backends
                if tools:
                    # Add instructions for JSON format to the prompt
                    tool_instructions = self._format_tools_as_instructions(tools, tool_choice)
                    messages[0]["content"] = f"{prompt}\n\n{tool_instructions}"
                
                response_text = self.llm_client.generate(messages)
                
                if tools:
                    # Try to parse JSON response for tool calls
                    response = self._parse_tool_response(response_text, tools)
                else:
                    # Create a simple response object
                    class SimpleResponse:
                        def __init__(self, content):
                            self.content = content
                    response = SimpleResponse(response_text)
                
                return response
            
        except Exception as e:
            self.logger.error(f"Error invoking LLM: {e}")
            raise
    
    def _call_remote_with_tools(self, messages: List[Dict], tools: List[Dict], tool_choice: str = None):
        """Call remote LLM with native tool support."""
        try:
            # Prepare the API call parameters
            api_params = {
                "model": self.llm_client.model_name,
                "messages": messages,
                "tools": tools,
                **self.llm_client.generation_params
            }
            
            if tool_choice:
                # OpenAI expects object format for specific function
                # Format: {"type": "function", "function": {"name": "function_name"}}
                api_params["tool_choice"] = {"type": "function", "function": {"name": tool_choice}}
            
            # Call the OpenAI client directly
            response = self.llm_client.client.chat.completions.create(**api_params)
            
            return response.choices[0].message
            
        except Exception as e:
            self.logger.error(f"Error calling remote LLM with tools: {e}")
            raise
    
    def _format_tools_as_instructions(self, tools: List[Dict], tool_choice: str = None) -> str:
        """Format tools as instructions for models that don't support tool calling."""
        if not tools:
            return ""
            
        tool = tools[0]  # Use the first (and usually only) tool
        function = tool.get('function', {})
        name = function.get('name', 'unknown')
        description = function.get('description', '')
        parameters = function.get('parameters', {})
        
        instructions = f"\nPlease respond with a JSON object that contains:\n"
        instructions += f"- function: \"{name}\"\n"
        instructions += f"- parameters: an object with the required fields\n\n"
        
        if 'properties' in parameters:
            instructions += "Required parameters:\n"
            for param_name, param_info in parameters['properties'].items():
                param_type = param_info.get('type', 'string')
                param_desc = param_info.get('description', '')
                instructions += f"- {param_name} ({param_type}): {param_desc}\n"
        
        # Add a specific example based on the function
        if name == "cast_vote":
            instructions += f"\nExample: {{\"function\": \"cast_vote\", \"parameters\": {{\"candidate\": 1, \"explanation\": \"good leadership qualities\"}}}}\n"
        elif name == "speak_message":
            instructions += f"\nExample: {{\"function\": \"speak_message\", \"parameters\": {{\"message\": \"I believe we need strong leadership...\"}}}}\n"
        elif name == "evaluate_speaking_urgency":
            instructions += f"\nExample: {{\"function\": \"evaluate_speaking_urgency\", \"parameters\": {{\"urgency_score\": 75, \"reasoning\": \"I have important concerns to share\"}}}}\n"
        else:
            instructions += f"\nExample: {{\"function\": \"{name}\", \"parameters\": {{\"key\": \"value\"}}}}\n"
        
        instructions += "\nYour response (JSON only):"
        
        return instructions
    
    def _parse_tool_response(self, response_text: str, tools: List[Dict]) -> Dict:
        """Parse tool response from text for models without native tool support."""
        try:
            # Look for JSON in the response - handle nested braces properly
            import re
            
            # Find the first complete JSON object
            brace_count = 0
            start_pos = response_text.find('{')
            if start_pos == -1:
                raise ValueError("No JSON object found")
            
            for i, char in enumerate(response_text[start_pos:], start_pos):
                if char == '{':
                    brace_count += 1
                elif char == '}':
                    brace_count -= 1
                    if brace_count == 0:
                        json_text = response_text[start_pos:i+1]
                        break
            else:
                raise ValueError("Incomplete JSON object")
            
            tool_data = json.loads(json_text)
            
            # Create a mock tool call response
            class MockToolCall:
                def __init__(self, function_name, arguments):
                    self.function = MockFunction(function_name, arguments)
            
            class MockFunction:
                def __init__(self, name, arguments):
                    self.name = name
                    self.arguments = json.dumps(arguments)
            
            class MockResponse:
                def __init__(self, tool_call):
                    self.tool_calls = [tool_call] if tool_call else []
            
            function_name = tool_data.get('function')
            parameters = tool_data.get('parameters', {})
            
            if function_name:
                tool_call = MockToolCall(function_name, parameters)
                return MockResponse(tool_call)
            
            # If no JSON found, return empty response
            class EmptyResponse:
                def __init__(self):
                    self.tool_calls = []
                    self.content = response_text
            
            return EmptyResponse()
            
        except Exception as e:
            self.logger.warning(f"Failed to parse tool response: {e}")
            class EmptyResponse:
                def __init__(self):
                    self.tool_calls = []
                    self.content = response_text
            return EmptyResponse()

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