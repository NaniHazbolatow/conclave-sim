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
    """Robust JSON parser with multiple extraction strategies."""
    
    @staticmethod
    def clean_text(text: str) -> str:
        """Clean text of control characters and other problematic characters."""
        # Remove control characters (except tab, newline, carriage return)
        cleaned = re.sub(r'[\x00-\x08\x0b\x0c\x0e-\x1f\x7f-\x9f]', '', text)
        
        # Fix common JSON issues in the text
        cleaned = cleaned.replace('\\n', '\n')  # Convert escaped newlines back
        cleaned = cleaned.replace('\\r', '\r')  # Convert escaped carriage returns back
        cleaned = cleaned.replace('\\t', '\t')  # Convert escaped tabs back
        
        return cleaned
    
    @staticmethod
    def extract_json_balanced_braces(text: str) -> Optional[str]:
        """Extract JSON using balanced brace counting."""
        start_pos = text.find('{')
        if start_pos == -1:
            return None
        
        brace_count = 0
        in_string = False
        escape_next = False
        
        for i, char in enumerate(text[start_pos:], start_pos):
            if escape_next:
                escape_next = False
                continue
                
            if char == '\\':
                escape_next = True
                continue
                
            if char == '"' and not escape_next:
                in_string = not in_string
                continue
                
            if not in_string:
                if char == '{':
                    brace_count += 1
                elif char == '}':
                    brace_count -= 1
                    if brace_count == 0:
                        return text[start_pos:i+1]
        
        return None
    
    @staticmethod
    def extract_json_regex(text: str) -> Optional[str]:
        """Extract JSON using regex patterns."""
        # Try to find JSON block between braces
        pattern = r'\{[^{}]*(?:\{[^{}]*\}[^{}]*)*\}'
        matches = re.findall(pattern, text, re.DOTALL)
        
        if matches:
            return matches[0]
        
        return None
    
    @staticmethod
    def parse_tool_response(text: str) -> ToolCallResult:
        """Parse tool response from LLM text with multiple fallback strategies."""
        if not text or not text.strip():
            return ToolCallResult(success=False, error="Empty response")
        
        # Clean the text first
        cleaned_text = JSONParser.clean_text(text)
        
        # Try multiple extraction methods
        extraction_methods = [
            JSONParser.extract_json_balanced_braces,
            JSONParser.extract_json_regex,
        ]
        
        for method in extraction_methods:
            try:
                json_str = method(cleaned_text)
                if json_str:
                    # Try to parse the JSON
                    parsed = json.loads(json_str)
                    
                    # Validate structure
                    if isinstance(parsed, dict):
                        function_name = parsed.get('function_name') or parsed.get('name')
                        arguments = parsed.get('arguments') or parsed.get('parameters') or {}
                        
                        if function_name:
                            return ToolCallResult(
                                success=True,
                                function_name=function_name,
                                arguments=arguments,
                                raw_response=text
                            )
            except json.JSONDecodeError:
                continue
            except Exception as e:
                logger.debug(f"JSON extraction method failed: {e}")
                continue
        
        # If all extraction methods fail, try to extract function name and arguments manually
        try:
            # Look for function name patterns
            name_patterns = [
                r'"function_name"\s*:\s*"([^"]+)"',
                r'"name"\s*:\s*"([^"]+)"',
                r'function_name:\s*"([^"]+)"',
                r'name:\s*"([^"]+)"'
            ]
            
            function_name = None
            for pattern in name_patterns:
                match = re.search(pattern, text, re.IGNORECASE)
                if match:
                    function_name = match.group(1)
                    break
            
            if function_name:
                # Try to extract arguments
                arguments = {}
                
                # Look for common argument patterns
                arg_patterns = [
                    r'"arguments"\s*:\s*\{([^}]+)\}',
                    r'"parameters"\s*:\s*\{([^}]+)\}',
                    r'arguments:\s*\{([^}]+)\}',
                    r'parameters:\s*\{([^}]+)\}'
                ]
                
                for pattern in arg_patterns:
                    match = re.search(pattern, text, re.IGNORECASE | re.DOTALL)
                    if match:
                        try:
                            # Try to parse the arguments as JSON
                            arg_content = '{' + match.group(1) + '}'
                            arguments = json.loads(arg_content)
                            break
                        except json.JSONDecodeError:
                            continue
                
                return ToolCallResult(
                    success=True,
                    function_name=function_name,
                    arguments=arguments,
                    raw_response=text
                )
        except Exception as e:
            logger.debug(f"Manual extraction failed: {e}")
        
        # Final fallback - return failure with original text
        return ToolCallResult(
            success=False,
            error=f"Could not parse tool response from: {text[:200]}...",
            raw_response=text
        )

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
        
        prompt = _build_prompt_for_tools(messages, tools, tool_choice)

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
                parsed_result = JSONParser.parse_tool_response(raw_response)

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
        response = self.llm_client.generate(
            messages,
            tools=tools,
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
        
        # If the model didn't use the tool, it might have put the JSON in the text
        if raw_text_response:
            parsed_result = JSONParser.parse_tool_response(raw_text_response)
            if parsed_result.success:
                self.logger.info("Successfully parsed tool call from text response as a native fallback.")
                # Override strategy to indicate it was a fallback
                parsed_result.strategy_used = "native_fallback"
                return parsed_result

        # If both native call and text parsing fail
        return ToolCallResult(
            success=False, 
            error="Model did not return a tool_calls object, and no valid JSON was found in the text response.",
            raw_response=raw_text_response
        )

    def call_tool(self, messages: List[Dict], tools: List[Dict], tool_choice: str = None) -> ToolCallResult:
        """Call a tool using the best available strategy."""
        if getattr(self.llm_client, 'supports_native_tools', False):
            return self._call_tool_natively(messages, tools, tool_choice)
        else:
            self.logger.warning("LLM client does not support native tools. Falling back to prompt-based strategy.")
            return self._call_tool_with_prompt(messages, tools, tool_choice)

class ToolExecutor:
    """Executes a tool call and returns the result."""
