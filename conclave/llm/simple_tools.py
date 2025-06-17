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
    
    def call_tool(self, messages: List[Dict], tools: List[Dict], tool_choice: str = None) -> ToolCallResult:
        """Call a tool using prompt-based approach with retry mechanism."""
        
        # Log the initial request
        self.logger.debug(f"Tool call request - Tool: {tool_choice}, Model: {self.model_name}")
        
        for attempt in range(self.max_retries + 1):
            if attempt > 0:
                self.logger.info(f"Retry attempt {attempt}/{self.max_retries} for {self.model_name}")
                if self.retry_delay > 0:
                    time.sleep(self.retry_delay)
            
            try:
                # Create prompt-based tool calling request
                prompt_messages = self._create_tool_prompt(messages, tools, tool_choice)
                
                # Call the LLM
                response_text = self.llm_client.generate(prompt_messages)
                self.logger.debug(f"LLM response: {response_text}")
                
                # Parse the response
                result = JSONParser.parse_tool_response(response_text)
                
                if result.success:
                    # Validate tool choice if specified
                    if tool_choice and result.function_name != tool_choice:
                        self.logger.warning(f"LLM chose function '{result.function_name}' but '{tool_choice}' was expected")
                        # Still consider it successful if parsing worked
                    
                    if attempt > 0:
                        self.logger.info(f"Tool calling succeeded on retry attempt {attempt}")
                    return result
                else:
                    self.logger.warning(f"Failed to parse tool response on attempt {attempt + 1}: {result.error}")
                    
            except Exception as e:
                self.logger.error(f"Tool calling attempt {attempt + 1} failed: {e}")
                result = ToolCallResult(success=False, error=f"Tool calling error: {str(e)}")
        
        # All attempts failed
        self.logger.error(f"All {self.max_retries + 1} tool calling attempts failed for {self.model_name}")
        return result
    
    def _create_tool_prompt(self, messages: List[Dict], tools: List[Dict], tool_choice: str = None) -> List[Dict]:
        """Create a prompt that instructs the model to respond with JSON tool calls."""
        
        # Extract the original prompt content from messages
        original_content = ""
        for msg in messages:
            if msg.get("role") == "user" and "content" in msg:
                original_content = msg["content"]
                break
        
        # Build tool descriptions
        tool_descriptions = []
        for tool in tools:
            if 'function' in tool:
                func_def = tool['function']
                tool_desc = f"- {func_def['name']}: {func_def['description']}"
                
                # Add parameter information
                if 'parameters' in func_def and 'properties' in func_def['parameters']:
                    params = func_def['parameters']['properties']
                    required = func_def['parameters'].get('required', [])
                    
                    param_list = []
                    for param_name, param_info in params.items():
                        param_desc = f"{param_name} ({param_info.get('type', 'any')})"
                        if param_name in required:
                            param_desc += " [required]"
                        if 'description' in param_info:
                            param_desc += f": {param_info['description']}"
                        param_list.append(param_desc)
                    
                    if param_list:
                        tool_desc += f"\n  Parameters: {', '.join(param_list)}"
                
                tool_descriptions.append(tool_desc)
        
        # Create the system prompt that includes the original content AND tool calling instructions
        tool_prompt = f"""{original_content}

---

Based on the above context and instructions, you must respond with a function call in JSON format.

Available functions:
{chr(10).join(tool_descriptions)}

IMPORTANT: You must respond with a JSON object in exactly this format:
{{
  "function_name": "name_of_function_to_call",
  "arguments": {{
    "parameter1": "value1",
    "parameter2": "value2"
  }}
}}

"""
        
        if tool_choice:
            tool_prompt += f"You MUST use the function '{tool_choice}'. Do not use any other function.\n\n"
        
        tool_prompt += "Respond ONLY with the JSON object, no additional text or explanation."
        
        # Create a single message with the combined prompt
        prompt_messages = [{"role": "user", "content": tool_prompt}]
        
        return prompt_messages
