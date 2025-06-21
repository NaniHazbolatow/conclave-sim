"""
Enhanced LLM Client Manager for Conclave Simulation

This module provides centralized LLM client management, eliminating redundancy
and simplifying tool calling by using prompt-based approach for all models.
"""

import threading
import logging
from typing import Optional, Dict, Any
from config.scripts import get_config  # Use new config adapter
from transformers import AutoModelForCausalLM, AutoTokenizer
import torch

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
        import requests
        import json
        
        class SimpleRemoteClient:
            def __init__(self, model_name, api_key=None, base_url="https://openrouter.ai/api/v1"):
                self.model_name = model_name
                self.api_key = api_key
                self.base_url = base_url
                
            def generate(self, messages, **kwargs):
                """Generate a response using the remote API (method expected by SimplifiedToolCaller)."""
                # Handle both string prompts and message lists
                if isinstance(messages, str):
                    prompt = messages
                elif isinstance(messages, list):
                    # Extract content from message format
                    if len(messages) > 0 and isinstance(messages[0], dict):
                        prompt = messages[0].get('content', str(messages))
                    else:
                        prompt = str(messages)
                else:
                    prompt = str(messages)
                
                headers = {
                    "Content-Type": "application/json",
                    "Authorization": f"Bearer {self.api_key}"
                }
                
                data = {
                    "model": self.model_name,
                    "messages": [{"role": "user", "content": prompt}],
                    "temperature": kwargs.get("temperature", 0.7),
                    "max_tokens": kwargs.get("max_tokens", 500)
                }
                
                try:
                    response = requests.post(f"{self.base_url}/chat/completions", 
                                           headers=headers, json=data, timeout=30)
                    response.raise_for_status()
                    result = response.json()
                    return result["choices"][0]["message"]["content"]
                except Exception as e:
                    return f"Error generating response: {e}"
                    
            def generate_response(self, prompt, **kwargs):
                """Legacy method name for backward compatibility."""
                return self.generate(prompt, **kwargs)
        
        # Get configuration - FIXED: Use correct method
        client_kwargs = self.config_manager.get_llm_client_kwargs()  # ConfigAdapter has this method directly
        model_name = client_kwargs.get("model_name", "openai/gpt-4o-mini")
        api_key = client_kwargs.get("api_key")
        
        return SimpleRemoteClient(model_name=model_name, api_key=api_key)
    
    def _create_local_client(self):
        """Create a local LLM client using transformers."""

        
        class LocalLLMClient:
            def __init__(self, model_name):
                self.model_name = model_name
                logger.info(f"Loading local model: {model_name}")
                
                self.tokenizer = AutoTokenizer.from_pretrained(model_name)
                self.model = AutoModelForCausalLM.from_pretrained(
                    model_name,
                    device_map="auto",
                    torch_dtype="auto"
                )
                
                # Add padding token if needed
                if self.tokenizer.pad_token is None:
                    self.tokenizer.pad_token = self.tokenizer.eos_token
                
            def generate(self, messages, **kwargs):
                """Generate a response using the local model."""
                # Handle different input formats
                if isinstance(messages, str):
                    prompt = messages
                elif isinstance(messages, list):
                    if len(messages) > 0 and isinstance(messages[0], dict):
                        prompt = messages[0].get('content', str(messages))
                    else:
                        prompt = str(messages)
                else:
                    prompt = str(messages)
                
                # Tokenize and generate
                inputs = self.tokenizer(prompt, return_tensors="pt").to(self.model.device)
                
                with torch.no_grad():
                    outputs = self.model.generate(
                        **inputs,
                        max_new_tokens=kwargs.get("max_tokens", 500),
                        temperature=kwargs.get("temperature", 0.7),
                        do_sample=True,
                        pad_token_id=self.tokenizer.eos_token_id
                    )
                
                # Decode only the new tokens
                response = self.tokenizer.decode(
                    outputs[0][len(inputs.input_ids[0]):], 
                    skip_special_tokens=True
                )
                return response.strip()
                
            def generate_response(self, prompt, **kwargs):
                """Legacy method name for backward compatibility."""
                return self.generate(prompt, **kwargs)
        
        # FIXED: Get model name directly from top-level LLM config
        llm_config = self.config_manager.config.models.llm
        
        # Based on your config.yaml, model_name is at the top level
        model_name = getattr(llm_config, 'model_name', 'meta-llama/llama-3.1-8b-instruct')
        
        return LocalLLMClient(model_name=model_name)
    
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