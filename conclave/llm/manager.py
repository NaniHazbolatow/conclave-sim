"""
Enhanced LLM Client Manager for Conclave Simulation

This module provides centralized LLM client management, eliminating redundancy
and simplifying tool calling by using prompt-based approach for all models.
"""

import threading
import logging
from typing import Optional, Dict, Any
from conclave.config import get_config_manager

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
        self.config_manager = get_config_manager()
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
            
            # Create new client
            llm_config = self.config_manager.agent_config.llm
            
            if llm_config.backend == 'local':
                # For local backend, use a simple mock client
                logger.warning("Local backend not fully supported in simplified manager. Using mock client.")
                client = self._create_mock_client()
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
        
        # Get configuration
        client_kwargs = self.config_manager.adapter.get_llm_client_kwargs()
        model_name = client_kwargs.get("model_name", "openai/gpt-4o-mini")
        api_key = client_kwargs.get("api_key")
        
        return SimpleRemoteClient(model_name=model_name, api_key=api_key)
    
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
