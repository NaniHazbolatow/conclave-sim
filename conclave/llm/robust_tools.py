"""
Robust tool calling framework that handles multiple models and parsing edge cases.

This module provides a unified interface for tool c        if not text or not text.strip():
            return ToolCallResult(success=False, error="Empty response")
        
        # Clean the text first
        cleaned_text = cls.clean_text(text):
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
from ..prompting.prompt_models import ToolDefinition # Updated import path

logger = logging.getLogger("conclave.llm") # NEW LOGGER - for tool calling, associated with LLM operations

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
        "anthropic/claude-sonnet-3.7",
    }
    
    # Models that have issues with native tool calling
    PROMPT_ONLY_MODELS = {
        "meta-llama/llama-3.1-8b-instruct",
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
        
        logger.debug(f"=== PARSING TOOL RESPONSE ===")
        logger.debug(f"Input text length: {len(text)}")
        logger.debug(f"Input text preview: {repr(text[:100])}")
        
        # Clean the text first
        cleaned_text = cls.clean_text(text)
        logger.debug(f"Cleaned text length: {len(cleaned_text)}")
        logger.debug(f"Cleaned text preview: {repr(cleaned_text[:100])}")
        
        # Try each extraction method
        for i, method in enumerate(cls.extract_json_methods()):
            method_name = method.__name__
            
            try:
                json_text = method(cleaned_text)
                if json_text:
                    # Try to parse the extracted JSON
                    try:
                        parsed = json.loads(json_text)
                        
                        # Validate the structure
                        if isinstance(parsed, dict) and 'function' in parsed:
                            # Get parameters and apply type coercion for common issues
                            parameters = parsed.get('parameters', {})
                            
                            # Special handling for cast_vote function
                            if parsed.get('function') == 'cast_vote' and 'candidate' in parameters:
                                candidate = parameters['candidate']
                                # Convert string candidate IDs to integers
                                if isinstance(candidate, str) and candidate.isdigit():
                                    parameters['candidate'] = int(candidate)
                                elif isinstance(candidate, str):
                                    logger.warning(f"Invalid candidate string value: '{candidate}'")
                            
                            return ToolCallResult(
                                success=True,
                                function_name=parsed.get('function'),
                                arguments=parameters,
                                raw_response=text,
                                strategy_used=ToolCallStrategy.PROMPT_BASED
                            )
                    except json.JSONDecodeError:
                        continue
            except Exception:
                continue
        
        # If all parsing failed, try to extract just the message content
        # Look for common patterns in responses
        message_patterns = [
            r'"message"\s*:\s*"([^"]+)"',
            r"'message'\s*:\s*'([^']+)'",
            r'message:\s*"([^"]+)"',
            r'message:\s*([^\n,}]+)',
        ]
        
        # Special patterns for stance generation
        stance_patterns = [
            r'"stance"\s*:\s*"([^"]+)"',
            r"'stance'\s*:\s*'([^']+)'",
            r'stance:\s*"([^"]+)"',
            r'stance:\s*([^\n,}]+)',
            # Match text that looks like a stance (starts with "My" or "I")
            r'(?:My|I)\s+[^{}"]*(?:candidate|choice|prefer|support)[^{}"]*\.(?:[^{}"]*\.)*',
        ]
        
        # Special patterns for vote casting
        vote_patterns = [
            r'"candidate"\s*:\s*(\d+)',
            r"'candidate'\s*:\s*(\d+)",
            r'candidate:\s*(\d+)',
            r'vote(?:\s+for)?:?\s*(\d+)',
            r'(?:I\s+vote\s+for|voting\s+for|choose|selecting)\s+(?:cardinal\s+)?(\d+)',
            r'Cardinal\s+(\d+)',
            r'ID\s*(\d+)',
            # More flexible patterns for when models use different formats
            r'(?:candidate|choice|selected?)[\s:]*(\d+)',
            r'(\d+)(?:\s*[,:]?\s*because|\s*due\s+to|\s*for\s+his|\s*as\s+he)',
            # Look for standalone single digits (0-4) that could be votes
            r'(?:^|\s)([0-4])(?:\s|$|[,.])',
        ]
        
        explanation_patterns = [
            r'"explanation"\s*:\s*"([^"]+)"',
            r"'explanation'\s*:\s*'([^']+)'",
            r'explanation:\s*"([^"]+)"',
            r'explanation:\s*([^\n,}]+)',
            r'because:?\s*([^\n]+)',
            r'reason:?\s*([^\n]+)',
        ]
        
        for i, pattern in enumerate(message_patterns):
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
        
        # Try stance patterns
        for i, pattern in enumerate(stance_patterns):
            match = re.search(pattern, text, re.IGNORECASE | re.DOTALL)
            if match:
                stance = match.group(1).strip() if len(match.groups()) > 0 else match.group(0).strip()
                if stance and len(stance.split()) >= 20:  # Minimum viable stance
                    return ToolCallResult(
                        success=True,
                        function_name="generate_stance",
                        arguments={"stance": stance},
                        raw_response=text,
                        strategy_used=ToolCallStrategy.PROMPT_BASED
                    )
        
        # Last resort: Extract any coherent stance-like text from the response
        # This handles cases where the model provides a good response but poor JSON formatting
        if 'stance' in text.lower() or any(word in text.lower() for word in ['candidate', 'prefer', 'support', 'church', 'theological']):
            # Clean up the text and extract meaningful content
            lines = text.split('\n')
            content_lines = []
            
            for line in lines:
                line = line.strip()
                # Skip obvious JSON artifacts and metadata
                if line and not line.startswith('{') and not line.startswith('}') and not line.startswith('"'):
                    # Remove leading/trailing quotes if present
                    line = line.strip('\'"')
                    if len(line) > 10 and not line.lower().startswith(('error', 'json', 'format')):
                        content_lines.append(line)
            
            if content_lines:
                extracted_text = ' '.join(content_lines)
                word_count = len(extracted_text.split())
                
                # If it looks like a reasonable stance (30+ words), use it
                if word_count >= 30:
                    return ToolCallResult(
                        success=True,
                        function_name="generate_stance",
                        arguments={"stance": extracted_text},
                        raw_response=text,
                        strategy_used=ToolCallStrategy.PROMPT_BASED
                    )
        
        # Try vote patterns for cast_vote function
        candidate_id = None
        explanation = None
        
        # Extract candidate ID
        for pattern in vote_patterns:
            match = re.search(pattern, text, re.IGNORECASE)
            if match:
                try:
                    candidate_id = int(match.group(1))
                    break
                except (ValueError, IndexError):
                    continue
        
        # Extract explanation
        for pattern in explanation_patterns:
            match = re.search(pattern, text, re.IGNORECASE | re.DOTALL)
            if match:
                explanation = match.group(1).strip()
                # Clean up common artifacts
                explanation = explanation.rstrip('",}')
                if explanation and len(explanation) > 5:
                    break
        
        # If we found a candidate ID, try to return a vote result
        if candidate_id is not None:
            # If no explanation found, try to extract any meaningful text
            if not explanation:
                # Look for any descriptive text that might be an explanation
                lines = text.split('\n')
                explanation_candidates = []
                
                for line in lines:
                    line = line.strip().strip('\'"{}",')
                    # Skip JSON artifacts and short lines
                    if (line and len(line) > 10 and 
                        not line.lower().startswith(('function', 'parameters', 'candidate', 'explanation')) and
                        not line.startswith('{') and not line.startswith('}')):
                        explanation_candidates.append(line)
                
                if explanation_candidates:
                    explanation = explanation_candidates[0]  # Take the first meaningful line
                else:
                    explanation = "Vote cast based on discussions and candidate qualifications"
            
            return ToolCallResult(
                success=True,
                function_name="cast_vote",
                arguments={"candidate": candidate_id, "explanation": explanation},
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
        self.logger = logger or logging.getLogger("conclave.llm") # Ensure robust_tools uses conclave.llm logger
        self.model_name = getattr(llm_client, 'model_name', 'unknown')
        
        # Use config if provided, otherwise fall back to defaults
        if config: # config is an instance of ConfigLoader
            # Accessing LLMConfig and then ToolCallingConfig
            llm_config_model = config.get_llm_config() # ConfigLoader returns LLMConfig model
            tool_calling_settings = llm_config_model.tool_calling # Access the tool_calling attribute

            self.max_retries = tool_calling_settings.max_retries
            self.retry_delay = tool_calling_settings.retry_delay
            self.enable_fallback = tool_calling_settings.enable_fallback
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
        # Ensure result is defined in case all attempts fail before result is assigned
        if 'result' not in locals() or result is None: # MODIFIED: Added 'or result is None'
            # Create a default error result if no strategy was even attempted or if 'result' was not set
            return ToolCallResult(
                success=False,
                error=f"All {self.max_retries + 1} attempts failed. No specific result from strategies.",
                strategy_used=self.preferred_strategy 
            )
        return ToolCallResult(
            success=False,
            error=f"All {self.max_retries + 1} attempts failed. Last error: {result.error}",
            strategy_used=result.strategy_used if 'result' in locals() else ToolCallStrategy.AUTO
        )
    
    def _try_native_tool_calling(self, messages: List[Dict], tool_definitions_yaml: List[Dict], tool_choice: str = None) -> ToolCallResult:
        """Try native tool calling.
        The 'tool_definitions_yaml' parameter is expected to be a list of tool definitions 
        in the format loaded from prompts.yaml."""
        try:
            native_tools = []
            if tool_definitions_yaml:
                for tool_def in tool_definitions_yaml:
                    if isinstance(tool_def, dict) and tool_def.get("name") and tool_def.get("parameters"):
                        native_tools.append({
                            "type": "function",
                            "function": {
                                "name": tool_def["name"],
                                "description": tool_def.get("description", ""), # Ensure description, even if empty
                                "parameters": tool_def["parameters"] # Assumed to be in OpenAI format
                            }
                        })
                    else:
                        self.logger.warning(f"Skipping malformed tool definition for native calling: {tool_def}")
            
            if not native_tools:
                 return ToolCallResult(
                     success=False, 
                     error="No valid tools provided or all definitions were malformed for native calling.",
                     raw_response=None,
                     strategy_used=ToolCallStrategy.NATIVE
                 )

            # Prepare API parameters
            api_params = {
                "model": self.model_name,
                "messages": messages,
                "tools": native_tools, # Use the transformed list
                **getattr(self.llm_client, 'generation_params', {})
            }
            
            # Add tool choice if specified and model supports it
            if tool_choice and self.model_name in ModelCapabilities.NATIVE_TOOL_MODELS:
                if any(t["function"]["name"] == tool_choice for t in native_tools):
                    api_params["tool_choice"] = {"type": "function", "function": {"name": tool_choice}}
                else:
                    self.logger.warning(f"Tool choice '{tool_choice}' not found in the provided tools for native calling. API will use default behavior (auto).")
            
            # Call the API
            response = self.llm_client.client.chat.completions.create(**api_params)
            message = response.choices[0].message
            
            # Check for tool calls
            if hasattr(message, 'tool_calls') and message.tool_calls:
                tool_call = message.tool_calls[0] # Assuming one tool call for now
                function_name = tool_call.function.name
                
                try:
                    arguments = json.loads(tool_call.function.arguments)
                except json.JSONDecodeError as e:
                    self.logger.error(f"Failed to parse arguments from native tool call: {tool_call.function.arguments}. Error: {e}")
                    return ToolCallResult(
                        success=False,
                        error=f"Failed to parse arguments from native tool call: {e}",
                        raw_response=str(message),
                        strategy_used=ToolCallStrategy.NATIVE
                    )

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
            self.logger.warning(f"Native tool calling failed: {e}", exc_info=True)
            return ToolCallResult(
                success=False,
                error=f"Native tool calling error: {str(e)}",
                strategy_used=ToolCallStrategy.NATIVE
            )
    
    # MODIFIED: Changed from async def to def
    # MODIFIED: Added tool_choice parameter
    def _try_prompt_based_tool_calling(
        self,
        messages: List[Dict[str, str]],
        tool_definitions: List[Union[Dict, ToolDefinition]],
        tool_choice: Optional[str] = None, # ADDED tool_choice
        max_retries: int = 1 
    ) -> ToolCallResult: # MODIFIED: Return type to ToolCallResult
        """Try prompt-based tool calling."""
        try:
            if not tool_definitions:
                return ToolCallResult(success=False, error="No tools provided")

            # Convert all ToolDefinition objects in the tools list to dictionaries
            processed_tool_dicts = []
            for tool_def_item in tool_definitions:
                if isinstance(tool_def_item, ToolDefinition):
                    processed_tool_dicts.append(tool_def_item.model_dump())
                elif isinstance(tool_def_item, dict):
                    processed_tool_dicts.append(tool_def_item)
                else:
                    self.logger.warning(f"Skipping unrecognized tool definition type: {type(tool_def_item)} in _try_prompt_based_tool_calling")
                    continue # Or return an error if strict type checking is needed
            
            if not processed_tool_dicts:
                return ToolCallResult(success=False, error="No valid tool definitions after processing types.")

            self.logger.debug(f"[_try_prompt_based_tool_calling] processed_tool_dicts: {processed_tool_dicts}")
            self.logger.debug(f"[_try_prompt_based_tool_calling] tool_choice: {tool_choice}") # Now tool_choice is defined
            
            tool_to_call_name = None
            if tool_choice:
                tool_to_call_name = tool_choice
            # Ensure processed_tool_dicts is not empty before accessing its first element
            elif processed_tool_dicts and processed_tool_dicts[0].get('function') and isinstance(processed_tool_dicts[0].get('function'), dict) and processed_tool_dicts[0].get('function', {}).get('name'):
                tool_to_call_name = processed_tool_dicts[0].get('function', {}).get('name')
            elif processed_tool_dicts and processed_tool_dicts[0].get('name'): # Fallback for older flat format
                tool_to_call_name = processed_tool_dicts[0].get('name')
            else:
                 self.logger.error(f"Could not determine tool to call. Tool choice: {tool_choice}, First tool: {processed_tool_dicts[0] if processed_tool_dicts else 'N/A'}")
                 return ToolCallResult(success=False, error="Could not determine tool to call from provided definitions.")

            self.logger.debug(f"[_try_prompt_based_tool_calling] tool_to_call_name: {tool_to_call_name}")
            
            chosen_tool_def_dict = None
            for tool_dict in processed_tool_dicts: # Iterate over list of dicts
                current_tool_name = tool_dict.get('function', {}).get('name')
                self.logger.debug(f"[_try_prompt_based_tool_calling] Checking tool_dict: {current_tool_name}")
                if current_tool_name == tool_to_call_name:
                    chosen_tool_def_dict = tool_dict # This is now guaranteed to be a dict
                    self.logger.debug(f"[_try_prompt_based_tool_calling] Found matching tool_dict: {chosen_tool_def_dict}")
                    break
            
            if not chosen_tool_def_dict:
                available_tool_names = [t.get('function', {}).get('name') for t in processed_tool_dicts if t.get('function', {}).get('name')]
                self.logger.error(f"[_try_prompt_based_tool_calling] Tool '{tool_to_call_name}' not found. Available tools: {available_tool_names}")
                return ToolCallResult(success=False, error=f"Tool '{tool_to_call_name}' not found in provided tool definitions.")

            # Extract arguments for _format_tool_instructions from chosen_tool_def_dict
            name = chosen_tool_def_dict.get('function', {}).get('name')
            description = chosen_tool_def_dict.get('function', {}).get('description', "") # Ensure description exists
            parameters = chosen_tool_def_dict.get('function', {}).get('parameters')

            if not name or parameters is None: # Parameters can be an empty dict, but not None if key 'parameters' exists
                self.logger.error(f"Chosen tool definition dictionary is missing 'name' or 'parameters' under 'function': {chosen_tool_def_dict}")
                return ToolCallResult(success=False, error="Malformed chosen tool definition (missing name/parameters).")

            # The fourth argument to _format_tool_instructions is the tool_definition_dict itself.
            instructions = self._format_tool_instructions(chosen_tool_def_dict) # CORRECTED: Pass the whole dict

            # Add instructions to the last message
            enhanced_messages = messages.copy()
            if enhanced_messages and enhanced_messages[-1]["content"] is not None:
                enhanced_messages[-1]["content"] += f"\n\n{instructions}"
            else:
                # Handle cases where the last message might be None or not have content
                # This might involve appending a new user message with instructions
                # For now, we assume the last message is a user message with content.
                self.logger.warning("Last message content was None or not found, instructions may not be properly appended.")
                # Fallback: append a new message if the last one is problematic
                enhanced_messages.append({"role": "user", "content": instructions})

            # Generate response
            response_text = self.llm_client.generate(enhanced_messages)
            
            # Parse the response
            result = JSONParser.parse_tool_response(response_text)
            result.strategy_used = ToolCallStrategy.PROMPT_BASED
            
            return result
            
        except Exception as e:
            self.logger.error(f"Prompt-based tool calling failed: {e}", exc_info=True)
            return ToolCallResult(
                success=False,
                error=f"Prompt-based tool calling error: {str(e)}",
                strategy_used=ToolCallStrategy.PROMPT_BASED
            )
    
    def _format_tool_instructions(self, tool_definition_dict: Dict) -> str: # CORRECTED: Parameter name and type hint
        """Format tool instructions for prompt-based calling using tool_definition from YAML."""
        # Extract details from the OpenAI-compatible format
        function_details = tool_definition_dict.get('function', {}) 
        name = function_details.get('name')
        description = function_details.get('description')
        parameters = function_details.get('parameters', {})
        
        # Fallback for direct attributes if 'function' key is not present (older format)
        if not name: name = tool_definition_dict.get('name') # CORRECTED
        if not description: description = tool_definition_dict.get('description') # CORRECTED
        if not parameters: parameters = tool_definition_dict.get('parameters', {}) # CORRECTED

        # Example JSON and additional instructions might still be top-level in the original definition
        # or could be nested if the definition itself is purely from OpenAI.
        # For now, assume they might be top-level as per original design for prompts.yaml.
        example_json = tool_definition_dict.get('example_json') # CORRECTED
        additional_instructions = tool_definition_dict.get('additional_prompt_instructions') # CORRECTED


        instructions = f'''IMPORTANT: You must respond with a valid JSON object in exactly this format:

{{"function": "{name}", "parameters": {{...}}}}

Tool: {name}
Description: {description}

'''
        
        if 'properties' in parameters:
            instructions += "Required parameters:\n"
            for param_name, param_info in parameters['properties'].items():
                param_type = param_info.get('type', 'string')
                param_desc = param_info.get('description', '')
                is_required = param_name in parameters.get('required', [])
                req_text = " (REQUIRED)" if is_required else " (optional)"
                instructions += f"- {param_name} ({param_type}){req_text}: {param_desc}\n"
        
        if example_json:
            instructions += f'\nExample: {example_json}\n'
        
        if additional_instructions:
            instructions += f"\n{additional_instructions}\n"
        
        instructions += "\nRespond with ONLY the JSON object, no other text:"
        
        return instructions

    def _add_retry_instruction(self, messages: List[Dict], attempt: int) -> List[Dict]:
        """Add retry instruction to messages when tool calling fails."""
        enhanced_messages = messages.copy()
        
        # Add increasingly specific retry instructions
        if attempt == 1:
            retry_text = "\n\nIMPORTANT: Please ensure your response is in the exact JSON format required. Your previous response may not have been formatted correctly. Use this exact format: {\"function\": \"function_name\", \"parameters\": {\"param\": \"value\"}}"
        elif attempt == 2:
            retry_text = "\n\nCRITICAL: You must respond with ONLY a valid JSON object. Do not include any explanatory text before or after the JSON. Start with { and end with }. No markdown, no code blocks, no extra text."
        else:
            retry_text = f"\n\nFINAL ATTEMPT ({attempt}): Output ONLY the JSON. Example: {{\"function\": \"generate_stance\", \"parameters\": {{\"stance\": \"Your 75-125 word stance here\"}}}}"
        
        # Add to the last user message
        if enhanced_messages and enhanced_messages[-1].get("role") == "user":
            enhanced_messages[-1]["content"] += retry_text
        else:
            # If no user message, add as new message
            enhanced_messages.append({"role": "user", "content": retry_text})
        
        return enhanced_messages

    # MODIFIED: Changed from async def to def
    # MODIFIED: Added tool_choice parameter
    def _call_prompt_based_tool( 
        self,
        tool_definition: Union[Dict, ToolDefinition],
        tool_arguments: Dict[str, Any],
        original_message_content: str, 
        tool_choice: Optional[str] = None, # ADDED tool_choice
        max_retries: int = 1
    ) -> ToolCallResult: # MODIFIED: Return type to ToolCallResult
        """
        "Calls" a tool by sending its definition and arguments back to an LLM
        to generate a simulated tool output.
        """
        tool_def_actual_dict: Dict[str, Any]

        if hasattr(tool_definition, 'model_dump') and callable(getattr(tool_definition, 'model_dump')):
            logger.debug(f"Tool definition (type: {type(tool_definition)}) has model_dump. Converting to dict.")
            tool_def_actual_dict = tool_definition.model_dump(exclude_none=True)
        elif isinstance(tool_definition, dict):
            logger.debug(f"Tool definition is already a dict (type: {type(tool_definition)}).")
            tool_def_actual_dict = tool_definition
        else:
            logger.error(f"Tool definition is of unexpected type: {type(tool_definition)}. Value: {tool_definition!r}")
            return {"error": f"Tool definition is of unexpected type: {type(tool_definition)}", "details": str(tool_definition)}

        tool_name = tool_def_actual_dict.get("function", {}).get("name")
        tool_description = tool_def_actual_dict.get("function", {}).get("description", "")
        
        if not tool_name:
            logger.error(f"Tool name not found in processed tool definition: {tool_def_actual_dict}")
            return {"error": "Tool name not found in processed tool definition"}

        logger.info(f"Simulating call to prompt-based tool: '{tool_name}' with arguments: {tool_arguments}")

        # Construct a prompt for the LLM to simulate the tool's output
        # This prompt needs to be carefully designed.
        # It should include the original user query context if relevant.
        # For now, a simple prompt structure:
        prompt_messages = [
            {"role": "system", "content": f"""You are simulating the output of a tool.
Original user message for context (if provided and relevant): {original_message_content}

Tool Name: {tool_name}
Tool Description: {tool_description}
Tool Arguments Received: {json.dumps(tool_arguments)}

Based on the tool's purpose and the arguments it received, generate a plausible JSON output that this tool would produce.
The output MUST be a single JSON object. Do NOT add any explanatory text before or after the JSON.
If the tool is supposed to indicate success or failure, include that in the JSON.
Example of a desired output format: {{"result": "some value", "status": "success"}}
"""}
        ]

        try:
            for attempt in range(max_retries):
                logger.debug(f"Attempt {attempt + 1} for prompt-based tool call simulation.")
                
                response_text = self.llm_client.generate_response(prompt_messages) 
                logger.debug(f"Prompt-based LLM raw response: {response_text}")
                
                # Pass tool_choice to parse_tool_response if it needs it,
                # or handle the logic of comparing parsed_result.function_name with tool_choice here.
                # For now, JSONParser.parse_tool_response does not take tool_choice.
                parsed_result = JSONParser.parse_tool_response(response_text)
                
                if parsed_result.success:
                    # It's important that parsed_result itself is a ToolCallResult.
                    # We are essentially returning the result of the parsing attempt.
                    if tool_choice and parsed_result.function_name != tool_choice: 
                        self.logger.warning(f"LLM chose function '{parsed_result.function_name}' but '{tool_choice}' was expected.")
                        # Potentially override success or add to error if strict tool_choice is required.
                        # For now, we'll still consider it a success if parsing worked, but log the discrepancy.
                    
                    # Update strategy_used as this method is specifically for prompt-based simulation
                    parsed_result.strategy_used = ToolCallStrategy.PROMPT_BASED
                    return parsed_result # Return the ToolCallResult from JSONParser
                else:
                    self.logger.error(f"Failed to parse prompt-based tool response: {parsed_result.error}")
                    # Update strategy_used and return the failure ToolCallResult from JSONParser
                    parsed_result.strategy_used = ToolCallStrategy.PROMPT_BASED
                    return parsed_result

        except Exception as e:
            logger.error(f"Error during prompt-based tool call: {e}", exc_info=True)
            return ToolCallResult(success=False, error=f"Prompt-based tool calling error: {str(e)}", strategy_used=ToolCallStrategy.PROMPT_BASED)
