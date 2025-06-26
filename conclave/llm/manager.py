"""
Enhanced LLM Client Manager for Conclave Simulation

This module provides centralized LLM client management, eliminating redundancy
and simplifying tool calling by using prompt-based approach for all models.
"""

import threading
import logging
from typing import Optional, Dict, Any
from config.scripts import get_config  # Use new config adapter
from transformers import AutoModelForCausalLM, AutoTokenizer, AutoConfig
import torch
import requests
import json

logger = logging.getLogger("conclave.llm")

class LLMClientManager:
    """
    Thread-safe singleton LLM client manager.
    
    Centralizes all LLM client creation and management, ensuring consistent
    configuration and eliminating redundant client instantiations.
    """
    
    _instance: Optional['LLMClientManager'] = None
    _lock = threading.Lock()
    _clients: Dict[str, Any] = {}  # Cache for different client types
    
    def __new__(cls) -> 'LLMClientManager':
        """Create or return existing singleton instance."""
        if cls._instance is None:
            with cls._lock:
                if cls._instance is None:
                    cls._instance = super().__new__(cls)
                    cls._instance._initialized = False
        return cls._instance
    
    def __init__(self):
        """Initialize the client manager (only once)."""
        if not getattr(self, '_initialized', False):
            self._initialize()
            self._initialized = True
    
    def _initialize(self):
        """Initialize the client manager."""
        self.config_manager = get_config()  # Use new config adapter
        logger.info("LLMClientManager initialized")
    
    def get_client(self, client_type: str = "default"):
        """
        Get LLM client instance based on configuration.
        
        Args:
            client_type: Type of client ("default", "environment", etc.)
            
        Returns:
            LLM client instance
        """
        if client_type in self._clients:
            return self._clients[client_type]
        
        with self._lock:
            # Double-check pattern
            if client_type in self._clients:
                return self._clients[client_type]
            
            # Create new client - FIXED: Use correct config path
            llm_config = self.config_manager.config.models.llm  # Updated path for new config structure
            
            if llm_config.backend == 'local':
                logger.info("Creating local LLM client")
                client = self._create_local_client()
            else:  # 'remote' backend
                # Create a simple remote client inline
                client = self._create_remote_client()
            
            self._clients[client_type] = client
            logger.info(f"Created {llm_config.backend} LLM client ({client_type}): {getattr(client, 'model_name', 'unknown')}")
            return client
    
    def clear_cache(self):
        """Clear client cache (mainly for testing)."""
        with self._lock:
            self._clients.clear()
            logger.info("LLM client cache cleared")
    
    @classmethod
    def reset(cls):
        """Reset singleton instance (mainly for testing)."""
        with cls._lock:
            if cls._instance:
                cls._instance.clear_cache()
            cls._instance = None
    
    def _create_remote_client(self):
        """Create a simple remote LLM client."""
        
        class SimpleRemoteClient:
            def __init__(self, model_name, api_key=None, base_url="https://openrouter.ai/api/v1", supports_native_tools=True):
                self.model_name = model_name
                self.api_key = api_key
                self.base_url = base_url
                self.supports_native_tools = supports_native_tools
                
            def generate(self, messages, tools=None, tool_choice=None, **kwargs):
                """Generate a response using the remote API, with native tool support."""
                if isinstance(messages, str):
                    api_messages = [{"role": "user", "content": messages}]
                else:
                    api_messages = messages

                headers = {
                    "Content-Type": "application/json",
                    "Authorization": f"Bearer {self.api_key}"
                }
                
                data = {
                    "model": self.model_name,
                    "messages": api_messages,
                    "temperature": kwargs.get("temperature", 0.7),
                    "max_tokens": kwargs.get("max_tokens", 500)
                }

                # Add tool parameters to the API request payload
                if tools:
                    data["tools"] = tools
                    if tool_choice:
                        data["tool_choice"] = tool_choice
                
                request_url = f"{self.base_url}/chat/completions"
                # Enhanced logging for debugging
                logger.debug(f"--- LLM Request ---\
"
                             f"URL: {request_url}\
"
                             f"Model: {self.model_name}\
"
                             f"Supports Native Tools: {self.supports_native_tools}\
"
                             f"Payload: {json.dumps(data, indent=2)}\
"
                             f"--------------------")

                try:
                    response = requests.post(request_url, 
                                           headers=headers, json=data, timeout=30)
                    response.raise_for_status()
                    result = response.json()

                    # Enhanced logging of the response
                    logger.debug(f"--- LLM Raw Response ---\
"
                                 f"Status Code: {response.status_code}\
"
                                 f"Headers: {response.headers}\
"
                                 f"Response Body: {json.dumps(result, indent=2)}\
"
                                 f"-----------------------")

                    message = result["choices"][0]["message"]
                    
                    # Extract text and tool_calls from the response
                    response_text = message.get("content") or ""
                    tool_calls = message.get("tool_calls")

                    return {"text": response_text, "tool_calls": tool_calls}
                except requests.exceptions.HTTPError as http_err:
                    logger.error(f"HTTP error occurred: {http_err}")
                    logger.error(f"Response body: {response.text}")
                    return {"text": f"HTTP error generating response: {http_err}", "tool_calls": None}
                except Exception as e:
                    logger.error(f"An unexpected error occurred during LLM generation: {e}", exc_info=True)
                    return {"text": f"Error generating response: {e}", "tool_calls": None}
                    
            def generate_response(self, prompt, **kwargs):
                """Legacy method name for backward compatibility."""
                return self.generate(prompt, **kwargs)
        
        # Get configuration for the remote client
        llm_config = self.config_manager.config.models.llm
        remote_config = getattr(llm_config, 'remote', None)
        
        # Safely access model_name, prioritizing remote config, then main config, then default
        if remote_config and getattr(remote_config, 'model_name', None):
            model_name = remote_config.model_name
        elif getattr(llm_config, 'model_name', None):
            model_name = llm_config.model_name
        else:
            model_name = 'meta-llama/llama-3.1-8b-instruct' # Default fallback
        
        # Safely access attributes from the Pydantic model, providing defaults
        base_url = getattr(remote_config, 'base_url', 'https://openrouter.ai/api/v1') if remote_config else 'https://openrouter.ai/api/v1'
        supports_native_tools = getattr(remote_config, 'native_tool_calling', True) if remote_config else True
        
        # API key is handled by the config manager
        client_kwargs = self.config_manager.get_llm_client_kwargs()
        api_key = client_kwargs.get("api_key")
        
        return SimpleRemoteClient(
            model_name=model_name, 
            api_key=api_key, 
            base_url=base_url, 
            supports_native_tools=supports_native_tools
        )
    
    def _create_local_client(self):
        """Create a local LLM client using transformers."""

        
        class LocalLLMClient:
            def __init__(self, model_name, local_config, llm_config):
                self.model_name = model_name
                self.config = local_config
                self.llm_config = llm_config
                # Check for native tool support based on model name
                self.supports_native_tools = "llama-3.1" in self.model_name.lower()
                logger.info(f"Loading tokenizer and model for {model_name} ...")
                
                try:
                    self.tokenizer = AutoTokenizer.from_pretrained(model_name)
                    self.model = AutoModelForCausalLM.from_pretrained(
                        model_name,
                        device_map="auto",
                        torch_dtype=torch.bfloat16 # Optimal for Llama 3.1
                    )
                except Exception as e:
                    if "GatedRepoError" in str(e) or "401" in str(e):
                        error_message = (
                            "Authentication failed when trying to load the model from Hugging Face. "
                            "This is likely because you are trying to access a gated model like Llama 3.1.\n"
                            "Please do the following:\n"
                            "1. Make sure you have accepted the license for the model on the Hugging Face website.\n"
                            "2. Authenticate with Hugging Face by running `huggingface-cli login` in your terminal and entering your token.\n"
                            "   Alternatively, you can set the `HUGGING_FACE_HUB_TOKEN` environment variable."
                        )
                        logger.error(error_message)
                        # Re-raise the exception to stop the simulation, as it cannot proceed.
                        raise e
                    else:
                        # Handle other potential loading errors
                        logger.error(f"An unexpected error occurred while loading the local model: {e}", exc_info=True)
                        raise e
                
                config = AutoConfig.from_pretrained(model_name)
                logger.info("Loaded model config:")
                logger.info(config)

                # Optionally, print specific fields
                logger.info(f"Model type: {config.model_type}")
                logger.info(f"Model name: {config._name_or_path}")
                logger.info(f"Number of hidden layers: {getattr(config, 'num_hidden_layers', 'N/A')}")
                logger.info(f"Model size (hidden size): {getattr(config, 'hidden_size', 'N/A')}")
                
                # Add padding token if needed
                if self.tokenizer.pad_token is None:
                    self.tokenizer.pad_token = self.tokenizer.eos_token
                

            def generate(self, messages, tools=None, tool_choice=None, **kwargs):
                """Generate a response using the local model, with native tool support."""
                # Unify input handling to always use the chat template
                if isinstance(messages, str):
                    messages = [{"role": "user", "content": messages}]
                elif not isinstance(messages, list):
                    messages = [{"role": "user", "content": str(messages)}]

                # Prepare tool-related arguments for the template (not for model.generate!)
                template_kwargs = {}
                if self.supports_native_tools and tools:
                    template_kwargs["tools"] = tools
                    if tool_choice:
                        template_kwargs["tool_choice"] = tool_choice

                # Apply the chat template to format the conversation correctly.
                prompt = self.tokenizer.apply_chat_template(
                    messages,
                    tokenize=False,
                    add_generation_prompt=True,
                    **template_kwargs
                )

                # Tokenize and generate
                inputs = self.tokenizer(prompt, return_tensors="pt").to(self.model.device)

                # Prepare generation arguments (NO tools/tool_choice here!)
                generation_kwargs = {
                    "max_new_tokens": getattr(self.config, "max_tokens", 256) if self.config else 256,
                    "temperature": self.llm_config.temperature if hasattr(self.llm_config, "temperature") else (getattr(self.config, "temperature", 0.7) if self.config else 0.7),
                    "top_p": getattr(self.config, "top_p", 0.9) if self.config else 0.9,
                    "repetition_penalty": getattr(self.config, "repetition_penalty", 1.05) if self.config else 1.05,
                    "do_sample": getattr(self.config, "do_sample", True) if self.config else True,
                    "pad_token_id": self.tokenizer.eos_token_id
                }

                with torch.no_grad():
                    outputs = self.model.generate(
                        **inputs,
                        **generation_kwargs
                    )

                # Decode only the new tokens for the text response
                raw_response_text = self.tokenizer.decode(
                    outputs[0][len(inputs.input_ids[0]):],
                    skip_special_tokens=False
                )

                # Decode a clean version for display or direct use
                clean_response_text = self.tokenizer.decode(
                    outputs[0][len(inputs.input_ids[0]):],
                    skip_special_tokens=True
                )

                # Extract tool calls if the model output supports it (not standard for HF)
                tool_calls = getattr(outputs, "tool_calls", None)

                return {
                    "text": clean_response_text.strip(),
                    "tool_calls": tool_calls,
                    "raw_response": raw_response_text.strip()
                }
                
            def generate_response(self, prompt, **kwargs):
                """Legacy method name for backward compatibility."""
                return self.generate(prompt, **kwargs)
        
        # FIXED: Get model name directly from top-level LLM config
        llm_config = self.config_manager.config.models.llm
        
        # Based on your config.yaml, model_name is at the top level
        model_name = getattr(llm_config, 'model_name', 'meta-llama/llama-3.1-8b-instruct')
        local_config = getattr(llm_config, 'local', None) # Use None for missing config
        
        return LocalLLMClient(model_name=model_name, local_config=local_config, llm_config=llm_config)
    
    def _create_mock_client(self):
        """Create a mock client for testing."""
        class MockClient:
            def __init__(self):
                self.model_name = "mock-model"
                
            def generate(self, messages, **kwargs):
                """Generate method expected by SimplifiedToolCaller."""
                return "This is a mock response for testing purposes."
                
            def generate_response(self, prompt, **kwargs):
                """Legacy method name for backward compatibility."""
                return self.generate(prompt, **kwargs)
        
        return MockClient()


# Global accessor function
def get_llm_client_manager() -> LLMClientManager:
    """
    Get the global LLMClientManager instance.
    
    Returns:
        The singleton LLMClientManager instance.
    """
    return LLMClientManager()


def get_llm_client(client_type: str = "default"):
    """
    Get an LLM client instance.
    
    Args:
        client_type: Type of client to get
        
    Returns:
        LLM client instance
    """
    return get_llm_client_manager().get_client(client_type)
