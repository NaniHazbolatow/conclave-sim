"""
Centralized Configuration Manager for Conclave Simulation

This module provides a thread-safe singleton ConfigManager that eliminates
redundant configuration loading and provides unified access to all configuration
settings across the entire codebase.
"""

import threading
import logging
from typing import Optional
from config.scripts.adapter import ConfigAdapter
from config.scripts.models import RefactoredConfig

logger = logging.getLogger("conclave.config")

class ConfigManager:
    """
    Thread-safe singleton configuration manager.
    
    Provides centralized access to configuration settings and eliminates
    redundant ConfigAdapter instantiations throughout the codebase.
    """
    
    _instance: Optional['ConfigManager'] = None
    _lock = threading.Lock()
    _config_adapter: Optional[ConfigAdapter] = None
    
    def __new__(cls) -> 'ConfigManager':
        """Create or return existing singleton instance."""
        if cls._instance is None:
            with cls._lock:
                if cls._instance is None:
                    cls._instance = super().__new__(cls)
                    cls._instance._initialized = False
        return cls._instance
    
    def __init__(self):
        """Initialize the configuration manager (only once)."""
        if not getattr(self, '_initialized', False):
            self._initialize()
            self._initialized = True
    
    def _initialize(self):
        """Initialize the configuration adapter."""
        try:
            self._config_adapter = ConfigAdapter()
            logger.info("ConfigManager initialized successfully")
        except Exception as e:
            logger.error(f"Failed to initialize ConfigManager: {e}")
            raise
    
    @property
    def config(self) -> RefactoredConfig:
        """Get the main configuration object."""
        if self._config_adapter is None:
            raise RuntimeError("ConfigManager not properly initialized")
        return self._config_adapter.config
    
    @property
    def adapter(self) -> ConfigAdapter:
        """Get the ConfigAdapter instance for advanced operations."""
        if self._config_adapter is None:
            raise RuntimeError("ConfigManager not properly initialized")
        return self._config_adapter
    
    # Convenience properties for commonly accessed config sections
    
    @property
    def agent_config(self):
        """Get agent configuration section."""
        return self.config.agent
    
    @property
    def simulation_config(self):
        """Get simulation configuration section."""
        return self.config.simulation
    
    @property
    def output_config(self):
        """Get output configuration section."""
        return self.config.output
    
    def get_llm_client_kwargs(self) -> dict:
        """Get LLM client configuration kwargs."""
        return self.adapter.get_llm_client_kwargs()
    
    @classmethod
    def reset(cls):
        """Reset singleton instance (mainly for testing)."""
        with cls._lock:
            cls._instance = None
            cls._config_adapter = None


# Global accessor function
def get_config_manager() -> ConfigManager:
    """
    Get the global ConfigManager instance.
    
    Returns:
        The singleton ConfigManager instance.
    """
    return ConfigManager()


# Convenience functions for quick access
def get_config() -> RefactoredConfig:
    """Get the main configuration object."""
    return get_config_manager().config


def get_agent_config():
    """Get agent configuration section."""
    return get_config_manager().agent_config


def get_simulation_config():
    """Get simulation configuration section."""
    return get_config_manager().simulation_config


def get_output_config():
    """Get output configuration section."""
    return get_config_manager().output_config
