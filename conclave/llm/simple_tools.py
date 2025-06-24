"""
Simplified Prompt-Based Tool Calling System

This module provides a streamlined tool calling interface that uses prompt-based
JSON responses for all models, ensuring compatibility with any remote LLM.
"""

import json
import re
import logging
import time
from typing import Dict, List, Optional, Any
from dataclasses import dataclass
from conclave.prompting.prompt_models import ToolDefinition

logger = logging.getLogger("conclave.llm")

def _build_prompt_for_tools(messages: List[Dict], tools: List[Dict], tool_choice: Optional[str]) -> str:
    """Constructs a detailed prompt for the LLM to invoke a tool via JSON output."""
    tool_definitions = json.dumps(tools, indent=2)
    prompt = (
        "You are a helpful assistant with access to a set of tools. "
        "To use a tool, you must respond *only* with a valid JSON object that conforms to the tool's schema. "
        "Do not include any other text, explanations, or conversational filler in your response. "
        "Your entire response must be the JSON object for the tool call.\n\n"
        f"Here are the available tools:\n{tool_definitions}\n\n"
    )
    if tool_choice:
        prompt += f"You must use the tool named '{tool_choice}'.\n\n"
    
    # Append conversation history
    for message in messages:
        prompt += f"{message['role']}: {message['content']}\n"
    
    prompt += "assistant:"
    return prompt

@dataclass
class ToolCallResult:
    """Standardized result from tool calling."""
    success: bool
    function_name: Optional[str] = None
    arguments: Optional[Dict[str, Any]] = None
    raw_response: Optional[str] = None
    error: Optional[str] = None
    strategy_used: str = "prompt_based"

class JSONParser:
    """
    Robust JSON parser that extracts the first valid JSON object from a text response
    by correctly handling nested structures.
    """

    @staticmethod
    def clean_text(text: str) -> str:
        """Clean text of control characters and markdown fences."""
        cleaned = re.sub(r'''[\x00-\x08\x0b\x0c\x0e-\x1f\x7f-\x9f]''', '', text)
        cleaned = re.sub(r'''^```json\s*|```$''', '', cleaned, flags=re.MULTILINE).strip()
        return cleaned

    @staticmethod
    def extract_json_block(text: str) -> Optional[str]:
        """
        Extracts the first complete JSON object from a string by counting braces.
        """
        start_index = text.find('{')
        if start_index == -1:
            return None

        brace_level = 0
        in_string = False
        escape_char = False

        for i, char in enumerate(text[start_index:]):
            current_index = start_index + i
            if escape_char:
                escape_char = False
                continue

            if char == '\\':
                escape_char = True
                continue

            if char == '"' and not escape_char:
                in_string = not in_string

            if not in_string:
                if char == '{':
                    brace_level += 1
                elif char == '}':
                    brace_level -= 1
            
            if brace_level == 0:
                return text[start_index : current_index + 1]
        
        return None # Incomplete JSON if loop finishes

    @staticmethod
    def parse_tool_response(text: str, tool_choice: Optional[str] = None) -> ToolCallResult:
        """
        Parses a tool response from LLM text, handling multiple common JSON structures.
        """
        if not text or not text.strip():
            return ToolCallResult(success=False, error="Empty response")

        cleaned_text = JSONParser.clean_text(text)
        json_str = JSONParser.extract_json_block(cleaned_text)

        if not json_str:
            return ToolCallResult(success=False, error="Could not extract a complete JSON block from the response.", raw_response=text)

        try:
            parsed = json.loads(json_str)
            if not isinstance(parsed, dict):
                return ToolCallResult(success=False, error="Parsed JSON is not a dictionary.", raw_response=text)

            # Strategy 1: Standard format `{"name": "...", "arguments": {...}}`
            func_name = parsed.get('name') or parsed.get('function_name')
            arguments = parsed.get('arguments') or parsed.get('parameters')
            if func_name and isinstance(arguments, dict):
                return ToolCallResult(success=True, function_name=func_name, arguments=arguments, raw_response=text, strategy_used="prompt_based_standard")

            # Strategy 2: Nested format `{"tool_name": {"arg": "value"}}`
            if len(parsed) == 1:
                potential_func_name = list(parsed.keys())[0]
                potential_args = parsed[potential_func_name]
                if isinstance(potential_args, dict):
                    return ToolCallResult(success=True, function_name=potential_func_name, arguments=potential_args, raw_response=text, strategy_used="prompt_based_nested")

            # Strategy 3: Argument-only format `{"arg": "value"}`
            if tool_choice:
                return ToolCallResult(success=True, function_name=tool_choice, arguments=parsed, raw_response=text, strategy_used="prompt_based_args_only")

            return ToolCallResult(success=False, error="Could not determine tool call structure from the parsed JSON.", raw_response=text)

        except json.JSONDecodeError as e:
            return ToolCallResult(success=False, error=f"Failed to decode extracted JSON: {e}", raw_response=text)

class SimplifiedToolCaller:
    """Simplified tool calling interface using prompt-based approach for all models."""
    
    def __init__(self, llm_client, logger=None, max_retries=3, retry_delay=1.0):
        self.llm_client = llm_client
        self.logger = logger or logging.getLogger("conclave.llm")
        self.model_name = getattr(llm_client, 'model_name', 'unknown')
        self.max_retries = max_retries
        self.retry_delay = retry_delay
        
        self.logger.info(f"Initialized SimplifiedToolCaller for {self.model_name} with prompt-based strategy")
    
    def _call_tool_with_prompt(self, messages: List[Dict], tools: List[Dict], tool_choice: str = None) -> ToolCallResult:
        """Calls a tool using the legacy prompt-based approach with retries."""
        self.logger.info("Attempting to call tool using prompt-based strategy.")
        
        # Convert ToolDefinition objects to dicts for the prompt
        prompt_tools = [tool.to_dict() if hasattr(tool, 'to_dict') else tool for tool in tools]
        prompt = _build_prompt_for_tools(messages, prompt_tools, tool_choice)

        for i in range(self.max_retries):
            try:
                # The generate method now returns a dict, we need the 'text' part
                response_data = self.llm_client.generate(prompt)
                raw_response = response_data.get("text", "")

                if not raw_response.strip():
                    self.logger.warning(f"Attempt {i+1}/{self.max_retries}: LLM returned an empty response.")
                    time.sleep(self.retry_delay * (i + 1))
                    continue

                # Use the JSONParser to extract the tool call from the text
                parsed_result = JSONParser.parse_tool_response(raw_response, tool_choice=tool_choice)

                if parsed_result.success:
                    self.logger.info("Successfully parsed tool call using prompt-based strategy.")
                    return parsed_result
                else:
                    self.logger.warning(
                        f"Attempt {i+1}/{self.max_retries}: Failed to parse tool response. "
                        f"Error: {parsed_result.error}"
                    )
                    time.sleep(self.retry_delay * (i + 1))

            except Exception as e:
                self.logger.error(f"An exception occurred during prompt-based tool call attempt {i+1}: {e}")
                time.sleep(self.retry_delay * (i + 1))

        self.logger.error("Failed to get a valid tool call after all retries.")
        return ToolCallResult(
            success=False,
            error="Failed to get a valid tool call after multiple retries.",
            strategy_used="prompt_based"
        )

    def _call_tool_natively(self, messages: List[Dict], tools: List[Dict], tool_choice: str = None) -> ToolCallResult:
        """Calls a tool using the modern, native approach, with a prompt-based fallback."""
        self.logger.info("Attempting to call tool using native strategy.")
        
        # Ensure tools are in the correct dictionary format for the API
        api_tools = [tool.to_dict() if hasattr(tool, 'to_dict') else tool for tool in tools]
        
        response = self.llm_client.generate(
            messages,
            tools=api_tools,
            tool_choice=tool_choice
        )

        tool_calls = response.get("tool_calls")
        raw_text_response = response.get("text", "")

        if tool_calls:
            self.logger.info("Successfully received tool call via native strategy.")
            # Assuming the first tool call is the one we want
            first_call = tool_calls[0]
            function_info = first_call.get("function")
            if not function_info:
                return ToolCallResult(success=False, error="Tool call did not contain function info.")

            return ToolCallResult(
                success=True,
                function_name=function_info.get("name"),
                arguments=json.loads(function_info.get("arguments", "{}")),
                raw_response=raw_text_response,
                strategy_used="native"
            )
        
        # --- NATIVE FALLBACK LOGIC ---
        self.logger.warning(
            "Native tool call failed: Model did not return a tool_calls object. "
            "Attempting to parse the text response as a fallback."
        )
        
        # Fallback 1: Try the Llama 3.1 specific parser first
        raw_response_from_local = response.get("raw_response")
        if raw_response_from_local:
            llama_parsed_result = Llama31ToolParser.parse(raw_response_from_local)
            if llama_parsed_result and llama_parsed_result.success:
                self.logger.info("Successfully parsed tool call using Llama31ToolParser.")
                return llama_parsed_result

        # Fallback 2: If the model didn't use the tool, it might have put the JSON in the text
        if raw_text_response:
            parsed_result = JSONParser.parse_tool_response(raw_text_response, tool_choice=tool_choice)
            if parsed_result.success:
                self.logger.info("Successfully parsed tool call from text response as a native fallback.")
                # Override strategy to indicate it was a fallback
                parsed_result.strategy_used = "native_fallback_json"
                return parsed_result

        # If both native call and text parsing fail, trigger the prompt-based fallback
        self.logger.warning(
            "Native tool-calling and text-parsing fallback both failed. "
            "Switching to the prompt-based strategy as a final attempt."
        )
        return self._call_tool_with_prompt(messages, tools, tool_choice)

    def call_tool(self, messages: List[Dict], tools: List[Dict], tool_choice: str = None) -> ToolCallResult:
        """Call a tool using the best available strategy."""
        if getattr(self.llm_client, 'supports_native_tools', False):
            return self._call_tool_natively(messages, tools, tool_choice)
        else:
            self.logger.warning("LLM client does not support native tools. Falling back to prompt-based strategy.")
            return self._call_tool_with_prompt(messages, tools, tool_choice)

class Llama31ToolParser:
    """
    Parses tool calls from Llama 3.1's specific text output format.
    """
    @staticmethod
    def parse(raw_text: str) -> Optional[ToolCallResult]:
        """
        Extracts and parses a tool call from the Llama 3.1 raw text output.
        Looks for `<|begin_of_tool_code|>` and `<|end_of_tool_code|>` tags.
        """
        logger.debug("Attempting to parse tool call using Llama31ToolParser.")
        # Regex to find the content within the tool code tags
        match = re.search(r"<\|begin_of_tool_code\|>\s*(.*?)\s*<\|end_of_tool_code\|>", raw_text, re.DOTALL)
        
        if not match:
            logger.debug("Llama31ToolParser: No tool code tags found.")
            return None

        tool_code_str = match.group(1).strip()
        logger.debug(f"Llama31ToolParser: Extracted tool code: {tool_code_str}")

        # Regex to extract function name and arguments from a Python-like call
        call_match = re.match(r"(\w+)\((.*)\)", tool_code_str)
        if not call_match:
            logger.warning(f"Llama31ToolParser: Could not parse function call structure from: {tool_code_str}")
            return None

        function_name = call_match.group(1)
        args_str = call_match.group(2)

        try:
            # This is a simplified way to parse Python keyword arguments.
            # It might not handle all edge cases (e.g., nested structures in args).
            # A more robust solution might use ast.literal_eval on a dict-like string.
            arguments = dict(re.findall(r"(\w+)=['\"](.*?)['\"]", args_str))
            logger.info(f"Llama31ToolParser: Successfully parsed tool call: {function_name} with args {arguments}")
            return ToolCallResult(
                success=True,
                function_name=function_name,
                arguments=arguments,
                raw_response=raw_text,
                strategy_used="native_fallback_llama3.1_parser"
            )
        except Exception as e:
            logger.error(f"Llama31ToolParser: Failed to parse arguments from string: '{args_str}'. Error: {e}")
            return None
