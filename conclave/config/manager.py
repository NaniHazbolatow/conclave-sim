"""
Configuration manager for the Conclave Simulation.
Handles loading and managing configuration from YAML files.
"""

import yaml
import os
import logging
from typing import Dict, Any, Optional, List
from pathlib import Path

logger = logging.getLogger(__name__)

class ConfigManager:
    """Manages configuration loading and access for the simulation."""
    
    def __init__(self, config_path: Optional[str] = None):
        """
        Initialize the configuration manager.
        
        Args:
            config_path: Path to the configuration file. If None, looks for config.yaml
                        in the current directory.
        """
        if config_path is None:
            # Look for config.yaml in the current directory
            config_path = "config.yaml"
            
        self.config_path = Path(config_path)
        self.config = self._load_config()
        
        # Set up logging level from config
        log_level = self.config.get("simulation", {}).get("log_level", "INFO")
        logging.getLogger().setLevel(getattr(logging, log_level))
        
    def _load_config(self) -> Dict[str, Any]:
        """Load configuration from YAML file."""
        try:
            if not self.config_path.exists():
                raise FileNotFoundError(f"Configuration file not found: {self.config_path}")
                
            with open(self.config_path, 'r', encoding='utf-8') as f:
                config = yaml.safe_load(f)
                
            logger.info(f"Configuration loaded from {self.config_path}")
            return config
            
        except FileNotFoundError:
            logger.error(f"Configuration file not found: {self.config_path}")
            raise
        except yaml.YAMLError as e:
            logger.error(f"Error parsing YAML configuration: {e}")
            raise
        except Exception as e:
            logger.error(f"Error loading configuration: {e}")
            raise
    
    def get_llm_config(self) -> Dict[str, Any]:
        """Get LLM configuration section."""
        return self.config.get("llm", {})
    
    def get_backend_type(self) -> str:
        """Get the selected backend type ('local' or 'remote')."""
        return self.config.get("llm", {}).get("backend", "remote")
    
    def get_temperature(self) -> float:
        """Get the temperature setting."""
        return self.config.get("llm", {}).get("temperature", 0.7)
    
    def get_max_tokens(self) -> int:
        """Get the max tokens setting."""
        return self.config.get("llm", {}).get("max_tokens", 500)
    
    def get_local_config(self) -> Dict[str, Any]:
        """Get local (HuggingFace) configuration."""
        return self.config.get("llm", {}).get("local", {})
    
    def get_remote_config(self) -> Dict[str, Any]:
        """Get remote (OpenRouter) configuration."""
        return self.config.get("llm", {}).get("remote", {})
    
    def get_local_model_name(self) -> str:
        """Get the local model name."""
        return self.get_local_config().get("model_name", "Qwen/Qwen2.5-1.5B-Instruct")
    
    def get_remote_model_name(self) -> str:
        """Get the remote model name."""
        return self.get_remote_config().get("model_name", "openai/gpt-4o-mini")
    
    def get_simulation_config(self) -> Dict[str, Any]:
        """Get simulation configuration section."""
        return self.config.get("simulation", {})
    
    def get_embedding_config(self) -> Dict[str, Any]:
        """Get embedding configuration section."""
        return self.config.get("embeddings", {})
    
    def get_embedding_model_name(self) -> str:
        """Get the embedding model name."""
        return self.get_embedding_config().get("model_name", "intfloat/e5-large-v2")
    
    def get_embedding_device(self) -> str:
        """Get the embedding device setting."""
        return self.get_embedding_config().get("device", "auto")
    
    def get_embedding_batch_size(self) -> int:
        """Get the embedding batch size."""
        return self.get_embedding_config().get("batch_size", 32)
    
    def get_embedding_cache_enabled(self) -> bool:
        """Check if embedding caching is enabled."""
        return self.get_embedding_config().get("enable_caching", True)
    
    def get_embedding_cache_dir(self) -> str:
        """Get the embedding cache directory."""
        cache_dir = self.get_embedding_config().get("cache_dir", "~/.cache/embeddings")
        return os.path.expanduser(cache_dir)
    
    def get_embedding_similarity_threshold(self) -> float:
        """Get the similarity threshold for embedding matches."""
        return self.get_embedding_config().get("similarity_threshold", 0.7)
    
    def get_tool_calling_config(self) -> Dict[str, Any]:
        """Get tool calling configuration section."""
        return self.get_llm_config().get("tool_calling", {})
    
    def get_tool_calling_max_retries(self) -> int:
        """Get the maximum number of retry attempts for failed tool calls."""
        return self.get_tool_calling_config().get("max_retries", 3)
    
    def get_tool_calling_retry_delay(self) -> float:
        """Get the delay in seconds between retries."""
        return self.get_tool_calling_config().get("retry_delay", 1.0)
    
    def get_tool_calling_enable_fallback(self) -> bool:
        """Check if fallback to prompt-based tool calling is enabled."""
        return self.get_tool_calling_config().get("enable_fallback", True)
    
    def get_llm_max_retries(self) -> int:
        """Get maximum retry attempts for LLM calls."""
        return self.get_tool_calling_max_retries()
    
    def get_llm_backoff(self) -> float:
        """Get retry backoff seconds for LLM calls."""
        return self.get_tool_calling_config().get("retry_backoff_sec", 2.0)
    
    # Simulation configuration methods
    
    def get_max_speakers_per_round(self) -> int:
        """Get the maximum number of speakers per round."""
        if self.is_testing_groups_enabled():
            overrides = self.get_testing_group_overrides()
            if "max_speakers_per_round" in overrides:
                return overrides["max_speakers_per_round"]
        return self.get_simulation_config().get("max_speakers_per_round", 5)
    
    def get_num_discussions(self) -> int:
        """Get the number of complete discussion cycles to run."""
        if self.is_testing_groups_enabled():
            overrides = self.get_testing_group_overrides()
            if "num_discussions" in overrides:
                return overrides["num_discussions"]
        return self.get_simulation_config().get("num_discussions", 1)
    
    def get_max_election_rounds(self) -> int:
        """Get the maximum number of election rounds before stopping."""
        if self.is_testing_groups_enabled():
            overrides = self.get_testing_group_overrides()
            if "max_election_rounds" in overrides:
                return overrides["max_election_rounds"]
        return self.get_simulation_config().get("max_election_rounds", 3)
    
    def get_randomize_speaking_order(self) -> bool:
        """Get whether to randomize the order of speakers in discussion rounds."""
        return self.get_simulation_config().get("randomize_speaking_order", True)
    
    def get_discussion_min_words(self) -> int:
        """Get the minimum word count for discussion contributions."""
        return self.get_simulation_config().get("discussion_length", {}).get("min_words", 100)
    
    def get_discussion_max_words(self) -> int:
        """Get the maximum word count for discussion contributions."""
        return self.get_simulation_config().get("discussion_length", {}).get("max_words", 300)
    
    def get_visualization_config(self) -> Dict[str, Any]:
        """Get the visualization configuration."""
        return self.config.get("visualization", {})
    
    def is_local_backend(self) -> bool:
        """Check if the backend is set to local."""
        return self.get_backend_type() == "local"
    
    def is_remote_backend(self) -> bool:
        """Check if the backend is set to remote."""
        return self.get_backend_type() == "remote"
    
    def get_api_key_from_env(self) -> Optional[str]:
        """Get API key from environment variable."""
        remote_config = self.get_remote_config()
        api_key_env = remote_config.get("api_key_env", "OPENROUTER_API_KEY")
        return os.environ.get(api_key_env)
    
    def get_llm_client_kwargs(self) -> Dict[str, Any]:
        """
        Get kwargs for initializing the appropriate LLM client based on config.
        
        Returns:
            Dictionary of arguments for LLM client initialization.
        """
        if self.is_local_backend():
            local_config = self.get_local_config()
            
            # Handle cache directory path expansion
            cache_dir = local_config.get("cache_dir")
            if cache_dir and isinstance(cache_dir, str):
                cache_dir = os.path.expanduser(cache_dir)
            
            # Handle device detection
            device = local_config.get("device", "auto")
            if device == "auto":
                device = None  # Let HuggingFaceClient detect automatically
            
            return {
                "model_name": local_config.get("model_name"),
                "use_quantization": local_config.get("use_quantization", False),  # Disable by default for compatibility
                "device": device,
                "cache_dir": cache_dir,
                "temperature": self.get_temperature(),
                "max_tokens": self.get_max_tokens(),
                "do_sample": local_config.get("do_sample", True),
                "top_p": local_config.get("top_p", 0.9),
                "repetition_penalty": local_config.get("repetition_penalty", 1.1),
            }
        else:
            remote_config = self.get_remote_config()
            api_key = self.get_api_key_from_env()
            if not api_key:
                logger.warning(f"API key not found in environment variable: {remote_config.get('api_key_env', 'OPENROUTER_API_KEY')}")
            
            return {
                "model_name": remote_config.get("model_name"),
                "api_key": api_key,
                "base_url": remote_config.get("base_url"),
                "temperature": self.get_temperature(),
                "max_tokens": self.get_max_tokens(),
                "top_p": remote_config.get("top_p", 0.9),
                "frequency_penalty": remote_config.get("frequency_penalty", 0.1),
                "presence_penalty": remote_config.get("presence_penalty", 0.1),
            }
    
    def get_embedding_client_kwargs(self) -> Dict[str, Any]:
        """
        Get kwargs for initializing the embedding client based on config.
        
        Returns:
            Dictionary of arguments for embedding client initialization.
        """
        config = self.get_embedding_config()
        
        # Handle device detection
        device = config.get("device", "auto")
        if device == "auto":
            device = None  # Let the embedding client detect automatically
        
        return {
            "model_name": config.get("model_name", "intfloat/e5-large-v2"),
            "device": device,
            "batch_size": config.get("batch_size", 32),
            "enable_caching": config.get("enable_caching", True),
            "cache_dir": self.get_embedding_cache_dir(),
        }
    
    def reload_config(self):
        """Reload configuration from file."""
        self.config = self._load_config()
        logger.info("Configuration reloaded")
    
    def __str__(self) -> str:
        """String representation of current configuration."""
        backend = self.get_backend_type()
        if self.is_local_backend():
            model = self.get_local_model_name()
        else:
            model = self.get_remote_model_name()
            
        return f"ConfigManager(backend={backend}, model={model}, temp={self.get_temperature()})"
    
    def __repr__(self) -> str:
        return self.__str__()

    # Testing groups configuration methods
    def get_testing_groups_config(self) -> Dict[str, Any]:
        """Get the testing groups configuration."""
        return self.config.get("testing_groups", {})
    
    def is_testing_groups_enabled(self) -> bool:
        """Check if testing groups are enabled."""
        return self.get_testing_groups_config().get("enabled", False)
    
    def get_active_testing_group(self) -> str:
        """Get the active testing group name."""
        return self.get_testing_groups_config().get("active_group", "small")
    
    def get_testing_group_config(self, group_name: str = None) -> Dict[str, Any]:
        """Get configuration for a specific testing group."""
        if group_name is None:
            group_name = self.get_active_testing_group()
        return self.get_testing_groups_config().get(group_name, {})
    
    def get_testing_group_cardinal_ids(self, group_name: str = None) -> List[int]:
        """Get the list of cardinal IDs for a testing group."""
        group_config = self.get_testing_group_config(group_name)
        return group_config.get("cardinal_ids", [])
    
    def get_testing_group_candidate_ids(self, group_name: str = None) -> List[int]:
        """Get the list of candidate IDs for a testing group."""
        group_config = self.get_testing_group_config(group_name)
        return group_config.get("candidate_ids", [])
    
    def get_testing_group_overrides(self, group_name: str = None) -> Dict[str, Any]:
        """Get the override settings for a testing group."""
        group_config = self.get_testing_group_config(group_name)
        return group_config.get("override_settings", {})
    
    def apply_testing_group_overrides(self, group_name: str = None) -> None:
        """Apply testing group overrides to the current configuration."""
        if not self.is_testing_groups_enabled():
            return
            
        overrides = self.get_testing_group_overrides(group_name)
        if not overrides:
            return
            
        # Apply overrides to simulation config
        if "simulation" not in self.config:
            self.config["simulation"] = {}
            
        for key, value in overrides.items():
            if key == "supermajority_threshold":
                # Apply to voting config
                if "voting" not in self.config["simulation"]:
                    self.config["simulation"]["voting"] = {}
                self.config["simulation"]["voting"]["supermajority_threshold"] = value
            else:
                # Apply to simulation config
                self.config["simulation"][key] = value
    
    # Enhanced simulation methods with testing group support
    def get_num_cardinals(self) -> int:
        """Get the number of cardinals to include in simulation."""
        if self.is_testing_groups_enabled():
            group_config = self.get_testing_group_config()
            return group_config.get("total_cardinals", 5)
        return self.get_simulation_config().get("num_cardinals", 5)
    
    def get_supermajority_threshold(self) -> float:
        """Get the supermajority threshold for elections."""
        if self.is_testing_groups_enabled():
            overrides = self.get_testing_group_overrides()
            if "supermajority_threshold" in overrides:
                return overrides["supermajority_threshold"]
        return self.get_simulation_config().get("voting", {}).get("supermajority_threshold", 0.667)
    
# Global configuration instance
_config_manager = None

def get_config(config_path: Optional[str] = None) -> ConfigManager:
    """
    Get the global configuration manager instance.
    
    Args:
        config_path: Path to config file. Only used on first call.
        
    Returns:
        ConfigManager instance.
    """
    global _config_manager
    if _config_manager is None:
        _config_manager = ConfigManager(config_path)
    return _config_manager

def reload_config():
    """Reload the global configuration."""
    global _config_manager
    if _config_manager is not None:
        _config_manager.reload_config()
