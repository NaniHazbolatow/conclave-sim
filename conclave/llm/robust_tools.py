"""
Robust tool calling framework that handles multiple models and parsing edge cases.

This module provides a unified interface for tool calling that:
1. Detects model capabilities automatically
2. Falls back gracefully between native and prompt-based tool calling
3. Handles JSON parsing robustly with multiple strategies
4. Provides consistent responses regardless of the underlying model
"""

import json
import re
import logging
from typing import Dict, List, Optional, Union, Any
from dataclasses import dataclass
from enum import Enum

logger = logging.getLogger(__name__)

class ToolCallStrategy(Enum):
    """Strategy for tool calling."""
    NATIVE = "native"        # Use model's native tool calling
    PROMPT_BASED = "prompt"  # Use prompt-based JSON responses
    AUTO = "auto"           # Auto-detect best strategy

@dataclass
class ToolCallResult:
    """Standardized result from tool calling."""
    success: bool
    function_name: Optional[str] = None
    arguments: Optional[Dict[str, Any]] = None
    raw_response: Optional[str] = None
    error: Optional[str] = None
    strategy_used: Optional[ToolCallStrategy] = None

class ModelCapabilities:
    """Database of model capabilities."""
    
    # Models known to support native tool calling well
    NATIVE_TOOL_MODELS = {
        "openai/gpt-4o-mini",
        "openai/gpt-4o", 
        "openai/gpt-3.5-turbo",
        "anthropic/claude-3-haiku",
        "anthropic/claude-3-sonnet",
        "anthropic/claude-3-opus",
    }
    
    # Models that have issues with native tool calling
    PROMPT_ONLY_MODELS = {
        "meta-llama/llama-3.1-8b-instruct",
        "meta-llama/llama-3.1-70b-instruct",
        "meta-llama/llama-3.2-8b-instruct",
        "mistralai/mistral-nemo",
        "qwen/qwen2.5-8b-instruct",
    }
    
    @classmethod
    def get_preferred_strategy(cls, model_name: str) -> ToolCallStrategy:
        """Get the preferred tool calling strategy for a model."""
        if model_name in cls.NATIVE_TOOL_MODELS:
            return ToolCallStrategy.NATIVE
        elif model_name in cls.PROMPT_ONLY_MODELS:
            return ToolCallStrategy.PROMPT_BASED
        else:
            # For unknown models, try native first, fall back to prompt-based
            return ToolCallStrategy.AUTO

class JSONParser:
    """Robust JSON parser with multiple extraction strategies."""
    
    @staticmethod
    def clean_text(text: str) -> str:
        """Clean text of control characters and other problematic characters."""
        # Remove control characters (except tab, newline, carriage return)
        cleaned = re.sub(r'[\x00-\x08\x0b\x0c\x0e-\x1f\x7f-\x9f]', '', text)
        
        # Fix common JSON issues
        cleaned = cleaned.replace('\n', '\\n')  # Escape actual newlines in strings
        cleaned = cleaned.replace('\r', '\\r')  # Escape carriage returns
        cleaned = cleaned.replace('\t', '\\t')  # Escape tabs
        
        return cleaned
    
    @staticmethod
    def extract_json_methods() -> List[callable]:
        """Get list of JSON extraction methods to try in order."""
        return [
            JSONParser._extract_json_balanced_braces,
            JSONParser._extract_json_regex_simple,
            JSONParser._extract_json_regex_aggressive,
            JSONParser._extract_json_line_by_line,
        ]
    
    @staticmethod
    def _extract_json_balanced_braces(text: str) -> Optional[str]:
        """Extract JSON using balanced brace counting."""
        start_pos = text.find('{')
        if start_pos == -1:
            return None
        
        brace_count = 0
        for i, char in enumerate(text[start_pos:], start_pos):
            if char == '{':
                brace_count += 1
            elif char == '}':
                brace_count -= 1
                if brace_count == 0:
                    return text[start_pos:i+1]
        return None
    
    @staticmethod
    def _extract_json_regex_simple(text: str) -> Optional[str]:
        """Extract JSON using simple regex."""
        pattern = r'\{[^{}]*(?:\{[^{}]*\}[^{}]*)*\}'
        match = re.search(pattern, text)
        return match.group(0) if match else None
    
    @staticmethod
    def _extract_json_regex_aggressive(text: str) -> Optional[str]:
        """Extract JSON using more aggressive regex."""
        # Look for patterns like {"function": "...", "parameters": {...}}
        pattern = r'\{\s*"function"\s*:\s*"[^"]+"\s*,\s*"parameters"\s*:\s*\{[^{}]*(?:\{[^{}]*\}[^{}]*)*\}\s*\}'
        match = re.search(pattern, text)
        return match.group(0) if match else None
    
    @staticmethod
    def _extract_json_line_by_line(text: str) -> Optional[str]:
        """Extract JSON by looking for complete objects line by line."""
        lines = text.split('\n')
        for line in lines:
            line = line.strip()
            if line.startswith('{') and line.endswith('}'):
                return line
        return None
    
    @classmethod
    def parse_tool_response(cls, text: str) -> ToolCallResult:
        """Parse tool response with multiple fallback strategies."""
        if not text:
            return ToolCallResult(success=False, error="Empty response")
        
        # Clean the text first
        cleaned_text = cls.clean_text(text)
        
        # Try each extraction method
        for method in cls.extract_json_methods():
            try:
                json_text = method(cleaned_text)
                if json_text:
                    # Try to parse the extracted JSON
                    parsed = json.loads(json_text)
                    
                    # Validate the structure
                    if isinstance(parsed, dict) and 'function' in parsed:
                        return ToolCallResult(
                            success=True,
                            function_name=parsed.get('function'),
                            arguments=parsed.get('parameters', {}),
                            raw_response=text,
                            strategy_used=ToolCallStrategy.PROMPT_BASED
                        )
            except json.JSONDecodeError:
                continue
            except Exception as e:
                logger.debug(f"JSON extraction method {method.__name__} failed: {e}")
                continue
        
        # If all parsing failed, try to extract just the message content
        # Look for common patterns in responses
        message_patterns = [
            r'"message"\s*:\s*"([^"]+)"',
            r"'message'\s*:\s*'([^']+)'",
            r'message:\s*"([^"]+)"',
            r'message:\s*([^\n,}]+)',
        ]
        
        for pattern in message_patterns:
            match = re.search(pattern, text, re.IGNORECASE)
            if match:
                message = match.group(1).strip()
                if message:
                    return ToolCallResult(
                        success=True,
                        function_name="speak_message",  # Assume default function
                        arguments={"message": message},
                        raw_response=text,
                        strategy_used=ToolCallStrategy.PROMPT_BASED
                    )
        
        return ToolCallResult(
            success=False, 
            error=f"Could not parse JSON from response",
            raw_response=text
        )

class RobustToolCaller:
    """Robust tool calling interface that handles multiple models and strategies."""
    
    def __init__(self, llm_client, logger=None, config=None, max_retries=3, retry_delay=1.0):
        self.llm_client = llm_client
        self.logger = logger or logging.getLogger(__name__)
        self.model_name = getattr(llm_client, 'model_name', 'unknown')
        
        # Use config if provided, otherwise fall back to defaults
        if config:
            self.max_retries = config.get_tool_calling_max_retries()
            self.retry_delay = config.get_tool_calling_retry_delay()
            self.enable_fallback = config.get_tool_calling_enable_fallback()
        else:
            self.max_retries = max_retries
            self.retry_delay = retry_delay
            self.enable_fallback = True
        
        # Determine preferred strategy for this model
        self.preferred_strategy = ModelCapabilities.get_preferred_strategy(self.model_name)
        self.logger.info(f"Initialized RobustToolCaller for {self.model_name} with strategy: {self.preferred_strategy}, max_retries: {self.max_retries}, retry_delay: {self.retry_delay}s, fallback: {self.enable_fallback}")
    
    def call_tool(self, messages: List[Dict], tools: List[Dict], tool_choice: str = None) -> ToolCallResult:
        """Call a tool using the most appropriate strategy for the model with retry mechanism."""
        
        for attempt in range(self.max_retries + 1):
            if attempt > 0:
                self.logger.info(f"Retry attempt {attempt}/{self.max_retries} for {self.model_name}")
                
                # Add delay before retry (except for first attempt)
                if self.retry_delay > 0:
                    import time
                    time.sleep(self.retry_delay)
                
                # Add retry instruction to the prompt
                retry_message = self._add_retry_instruction(messages, attempt)
            else:
                retry_message = messages
            
            # Execute strategy
            if self.preferred_strategy == ToolCallStrategy.NATIVE:
                # Try native first
                result = self._try_native_tool_calling(retry_message, tools, tool_choice)
                if result.success:
                    return result
                
                # Fall back to prompt-based if enabled
                if self.enable_fallback:
                    self.logger.warning(f"Native tool calling failed for {self.model_name}, falling back to prompt-based")
                    result = self._try_prompt_based_tool_calling(retry_message, tools, tool_choice)
                
            elif self.preferred_strategy == ToolCallStrategy.PROMPT_BASED:
                # Use prompt-based directly
                result = self._try_prompt_based_tool_calling(retry_message, tools, tool_choice)
                
            else:  # AUTO strategy
                # Try native first, then prompt-based
                result = self._try_native_tool_calling(retry_message, tools, tool_choice)
                if result.success:
                    return result
                
                if self.enable_fallback:
                    self.logger.info(f"Auto-detection: falling back to prompt-based for {self.model_name}")
                    result = self._try_prompt_based_tool_calling(retry_message, tools, tool_choice)
            
            # If successful, return immediately
            if result.success:
                if attempt > 0:
                    self.logger.info(f"Tool calling succeeded on retry attempt {attempt}")
                return result
                
            # Log failure and continue to next attempt
            self.logger.warning(f"Attempt {attempt + 1} failed: {result.error}")
        
        # All attempts failed
        self.logger.error(f"All {self.max_retries + 1} attempts failed for {self.model_name}")
        return ToolCallResult(
            success=False,
            error=f"All {self.max_retries + 1} attempts failed. Last error: {result.error}",
            strategy_used=result.strategy_used if 'result' in locals() else ToolCallStrategy.AUTO
        )
    
    def _try_native_tool_calling(self, messages: List[Dict], tools: List[Dict], tool_choice: str = None) -> ToolCallResult:
        """Try native tool calling."""
        try:
            # Prepare API parameters
            api_params = {
                "model": self.model_name,
                "messages": messages,
                "tools": tools,
                **getattr(self.llm_client, 'generation_params', {})
            }
            
            # Add tool choice if specified and model supports it
            if tool_choice and self.model_name in ModelCapabilities.NATIVE_TOOL_MODELS:
                api_params["tool_choice"] = {"type": "function", "function": {"name": tool_choice}}
            
            # Call the API
            response = self.llm_client.client.chat.completions.create(**api_params)
            message = response.choices[0].message
            
            # Check for tool calls
            if hasattr(message, 'tool_calls') and message.tool_calls:
                tool_call = message.tool_calls[0]
                function_name = tool_call.function.name
                arguments = json.loads(tool_call.function.arguments)
                
                return ToolCallResult(
                    success=True,
                    function_name=function_name,
                    arguments=arguments,
                    raw_response=str(message),
                    strategy_used=ToolCallStrategy.NATIVE
                )
            
            # No tool calls found
            return ToolCallResult(
                success=False,
                error="No tool calls in native response",
                raw_response=str(message),
                strategy_used=ToolCallStrategy.NATIVE
            )
            
        except Exception as e:
            self.logger.warning(f"Native tool calling failed: {e}")
            return ToolCallResult(
                success=False,
                error=f"Native tool calling error: {str(e)}",
                strategy_used=ToolCallStrategy.NATIVE
            )
    
    def _try_prompt_based_tool_calling(self, messages: List[Dict], tools: List[Dict], tool_choice: str = None) -> ToolCallResult:
        """Try prompt-based tool calling."""
        try:
            if not tools:
                return ToolCallResult(success=False, error="No tools provided")
            
            # Use the first tool (typically there's only one)
            tool = tools[0]
            function = tool.get('function', {})
            name = function.get('name', 'unknown')
            description = function.get('description', '')
            parameters = function.get('parameters', {})
            
            # Create enhanced instructions
            instructions = self._format_tool_instructions(name, description, parameters)
            
            # Add instructions to the last message
            enhanced_messages = messages.copy()
            enhanced_messages[-1]["content"] += f"\n\n{instructions}"
            
            # Generate response
            response_text = self.llm_client.generate(enhanced_messages)
            
            # Parse the response
            result = JSONParser.parse_tool_response(response_text)
            result.strategy_used = ToolCallStrategy.PROMPT_BASED
            
            return result
            
        except Exception as e:
            self.logger.error(f"Prompt-based tool calling failed: {e}")
            return ToolCallResult(
                success=False,
                error=f"Prompt-based tool calling error: {str(e)}",
                strategy_used=ToolCallStrategy.PROMPT_BASED
            )
    
    def _format_tool_instructions(self, name: str, description: str, parameters: Dict) -> str:
        """Format tool instructions for prompt-based calling."""
        instructions = f"""IMPORTANT: You must respond with a valid JSON object in exactly this format:

{{"function": "{name}", "parameters": {{...}}}}

Tool: {name}
Description: {description}

"""
        
        if 'properties' in parameters:
            instructions += "Required parameters:\n"
            for param_name, param_info in parameters['properties'].items():
                param_type = param_info.get('type', 'string')
                param_desc = param_info.get('description', '')
                is_required = param_name in parameters.get('required', [])
                req_text = " (REQUIRED)" if is_required else " (optional)"
                instructions += f"- {param_name} ({param_type}){req_text}: {param_desc}\n"
        
        # Add specific examples based on function name
        if name == "cast_vote":
            instructions += f'\nExample: {{"function": "cast_vote", "parameters": {{"candidate": 1, "explanation": "Strong leadership and pastoral experience"}}}}\n'
        elif name == "speak_message":
            instructions += f'\nExample: {{"function": "speak_message", "parameters": {{"message": "I believe we need a leader who can unite the Church in these challenging times..."}}}}\n'
        elif name == "evaluate_speaking_urgency":
            instructions += f'\nExample: {{"function": "evaluate_speaking_urgency", "parameters": {{"urgency_score": 75, "reasoning": "I have important concerns about the current voting trends"}}}}\n'
        
        instructions += "\nRespond with ONLY the JSON object, no other text:"
        
        return instructions
    
    def _add_retry_instruction(self, messages: List[Dict], attempt: int) -> List[Dict]:
        """Add retry instruction to messages when tool calling fails."""
        enhanced_messages = messages.copy()
        
        # Add increasingly specific retry instructions
        if attempt == 1:
            retry_text = "\n\nIMPORTANT: Please ensure your response is in the exact JSON format required. Your previous response may not have been formatted correctly."
        elif attempt == 2:
            retry_text = "\n\nCRITICAL: You must respond with ONLY a valid JSON object. Do not include any explanatory text before or after the JSON. Format: {\"function\": \"function_name\", \"parameters\": {...}}"
        else:
            retry_text = f"\n\nFINAL ATTEMPT ({attempt}): Respond with NOTHING except the JSON object. No text, no explanations, just the JSON."
        
        # Add to the last user message
        if enhanced_messages and enhanced_messages[-1].get("role") == "user":
            enhanced_messages[-1]["content"] += retry_text
        else:
            # If no user message, add as new message
            enhanced_messages.append({"role": "user", "content": retry_text})
        
        return enhanced_messages
