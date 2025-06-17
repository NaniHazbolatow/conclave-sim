"""
Compatibility adapter for transitioning from old to new configuration system.
Provides similar interface to ConfigManager but uses refactored configuration.
"""

import os
import shutil
import logging
from pathlib import Path
from typing import Dict, Any, Optional

from .loader import load_config, get_config_loader
from .models import RefactoredConfig, TestingGroupsConfig, LLMConfig, ToolCallingConfig


# Define Filter classes for logging
class ExcludeLoggersFilter(logging.Filter):
    def __init__(self, prefixes_to_exclude):
        super().__init__()
        self.prefixes_to_exclude = prefixes_to_exclude

    def filter(self, record):
        return not any(record.name.startswith(prefix) for prefix in self.prefixes_to_exclude)

class SpecificLoggersFilter(logging.Filter):
    def __init__(self, prefixes_to_include):
        super().__init__()
        self.prefixes_to_include = prefixes_to_include

    def filter(self, record):
        return any(record.name.startswith(prefix) for prefix in self.prefixes_to_include)

class AgentConversationsFilter(logging.Filter):
    """Filters logs for agent_conversations.log.
    Includes messages from 'conclave.agents' logger
    EXCEPT if the message starts with 'LLM Request:'."""
    def filter(self, record):
        if record.name.startswith('conclave.agents'):
            return not record.getMessage().startswith('LLM Request:')
        return False

class LLMIOFilter(logging.Filter):
    """Filters logs for llm_io.log.
    Includes messages from 'conclave.llm', 'httpx',
    OR messages from 'conclave.agents' that start with 'LLM Request:'."""
    def filter(self, record):
        if record.name.startswith('conclave.llm') or record.name.startswith('httpx'):
            return True
        if record.name.startswith('conclave.agents') and record.getMessage().startswith('LLM Request:'):
            return True
        return False

class ConfigAdapter:
    """Adapter to provide old ConfigManager interface with new refactored config system."""
    _instance = None # Add singleton instance tracker

    def __new__(cls, *args, **kwargs): # Implement __new__ for singleton
        if cls._instance is None:
            cls._instance = super(ConfigAdapter, cls).__new__(cls)
        return cls._instance
    
    def __init__(self, config_dir: str = None, config_object: RefactoredConfig = None): # Add config_object parameter
        """Initialize the config adapter.
        
        Args:
            config_dir: Directory containing refactored config files
            config_object: An optional pre-loaded RefactoredConfig object.
        """
        if hasattr(self, '_initialized') and self._initialized and (config_object is None): # Avoid re-init if already done and no new object
             # If already initialized and no new config_object is given,
             # and we are not forcing a reload from a specific config_dir,
             # then we assume the existing self.config is the one to use.
             # This handles the case where ConfigAdapter() is called multiple times
             # without args after initial setup.
            if config_dir is None or Path(config_dir) == self.config_dir:
                return

        if config_object:
            self.config = config_object
            # If config_object is provided, config_dir might be irrelevant or should be consistent.
            # For simplicity, we'll assume config_dir is mainly for loading if no object is passed.
            # We can set a default or try to infer it if needed, or require it.
            # Let's assume if config_object is passed, self.config_dir can be set from a default
            # or a passed config_dir if consistent.
            if config_dir:
                self.config_dir = Path(config_dir)
            else:
                # Default config_dir if not provided with a config_object
                project_root = Path(__file__).parent.parent.parent
                self.config_dir = project_root / "config"
            self.config_path = self.config_dir # Directory instead of single file
            # Loader might need to be re-evaluated if config is passed directly
            # For now, assume loader is tied to config_dir if used for loading.
            # If config is passed directly, direct access is prioritized.
            self.loader = get_config_loader(str(self.config_dir))

        elif config_dir is None:
            # Default to the parent directory (config/) where YAML files are located
            project_root = Path(__file__).parent.parent.parent
            config_dir = project_root / "config"
            self.config_dir = Path(config_dir)
            self.config: RefactoredConfig = load_config(str(self.config_dir))
            self.config_path = self.config_dir
            self.loader = get_config_loader(str(self.config_dir))
        else:
            self.config_dir = Path(config_dir)
            self.config: RefactoredConfig = load_config(str(self.config_dir))
            self.config_path = self.config_dir
            self.loader = get_config_loader(str(self.config_dir))
        
        self._initialized = True # Mark as initialized
    
    def initialize_logging(self, dynamic_logs_dir: str): # Made dynamic_logs_dir mandatory
        """Initialize logging configuration for console and file output."""
        
        general_console_level = logging.WARNING 

        specific_loggers_level_console = logging.WARNING # Changed from INFO to WARNING
        specific_loggers_level_file = logging.DEBUG 
        
        file_log_level_system = logging.INFO

        root_logger = logging.getLogger()
        root_logger.setLevel(logging.DEBUG)

        for handler in root_logger.handlers[:]:
            root_logger.removeHandler(handler)
            handler.close()

        formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')

        ch_general = logging.StreamHandler()
        ch_general.setLevel(general_console_level)
        ch_general.setFormatter(formatter)
        ch_general.addFilter(ExcludeLoggersFilter(['conclave.agents', 'conclave.llm', 'httpx']))
        root_logger.addHandler(ch_general)

        ch_specific = logging.StreamHandler()
        ch_specific.setLevel(specific_loggers_level_console)
        ch_specific.setFormatter(formatter)
        ch_specific.addFilter(SpecificLoggersFilter(['conclave.agents', 'conclave.llm']))
        root_logger.addHandler(ch_specific)

        # File Handlers
        logs_path = Path(dynamic_logs_dir)
        logs_path.mkdir(parents=True, exist_ok=True)

        system_log_path = logs_path / "system.log"
        fh_system = logging.FileHandler(system_log_path, mode='w')
        fh_system.setLevel(file_log_level_system) 
        fh_system.setFormatter(formatter)
        # Added 'httpx' to exclusion for system.log
        fh_system.addFilter(ExcludeLoggersFilter(['conclave.agents', 'conclave.llm', 'httpx']))
        root_logger.addHandler(fh_system)

        agent_log_path = logs_path / "agent_conversations.log"
        fh_agent = logging.FileHandler(agent_log_path, mode='w')
        fh_agent.setLevel(specific_loggers_level_file) 
        fh_agent.setFormatter(formatter)
        fh_agent.addFilter(AgentConversationsFilter()) # Use the new AgentConversationsFilter
        root_logger.addHandler(fh_agent)

        llm_log_path = logs_path / "llm_io.log"
        fh_llm = logging.FileHandler(llm_log_path, mode='w')
        fh_llm.setLevel(specific_loggers_level_file) 
        fh_llm.setFormatter(formatter)
        fh_llm.addFilter(LLMIOFilter()) # Use the new LLMIOFilter
        root_logger.addHandler(fh_llm)

        discussion_log_path = logs_path / "discussions.log"
        fh_discussion = logging.FileHandler(discussion_log_path, mode='w')
        fh_discussion.setLevel(specific_loggers_level_file) # Using same level as agent/llm file logs
        fh_discussion.setFormatter(formatter)
        fh_discussion.addFilter(SpecificLoggersFilter(['conclave.discussions']))
        root_logger.addHandler(fh_discussion)
            
        agent_logger = logging.getLogger('conclave.agents')
        # agent_logger.setLevel(specific_loggers_level_file) # Level is set by handlers
        agent_logger.propagate = True 

        llm_logger = logging.getLogger('conclave.llm')
        # llm_logger.setLevel(specific_loggers_level_file) # Level is set by handlers
        llm_logger.propagate = True
        
        discussion_logger = logging.getLogger('conclave.discussions')
        discussion_logger.setLevel(specific_loggers_level_file) # Match handler level
        discussion_logger.propagate = True

        httpx_logger = logging.getLogger('httpx')
        httpx_logger.setLevel(logging.WARNING) # Set httpx to WARNING for console
        httpx_logger.propagate = True

    def get_num_cardinals(self) -> int:
        """Get number of cardinals from simulation config."""
        # Check for testing group overrides
        sim_config = self.loader.get_simulation_config_with_overrides()
        return sim_config.get("num_cardinals", self.config.simulation.num_cardinals)
    
    def get_discussion_group_size(self) -> int:
        """Get discussion group size from simulation config."""
        sim_config = self.loader.get_simulation_config_with_overrides()
        return sim_config.get("discussion_group_size", self.config.simulation.discussion_group_size)
    
    def get_max_election_rounds(self) -> int:
        """Get max election rounds from simulation config."""
        sim_config = self.loader.get_simulation_config_with_overrides()
        return sim_config.get("max_election_rounds", self.config.simulation.max_election_rounds)

    def get_simulation_config(self) -> Dict[str, Any]:
        """Get the full simulation configuration dictionary with overrides."""
        return self.loader.get_simulation_config_with_overrides()

    def get_agent_config(self) -> Dict[str, Any]:
        """Get the full agent configuration dictionary with overrides."""
        return self.loader.get_agent_config() # Use the new method

    def get_output_config(self) -> Dict[str, Any]:
        """Get the full output configuration dictionary."""
        # Output config typically doesn't have overrides from testing_groups
        return self.loader.get_output_config() # Use the new method

    def get_llm_config(self) -> LLMConfig: # Changed return type to LLMConfig
        """Get LLM configuration from agent config."""
        agent_config_data = self.loader.get_agent_config()
        return LLMConfig(**agent_config_data.get("llm", {}))

    def get_tool_calling_config(self) -> ToolCallingConfig: # Changed return type
        """Get tool calling configuration section."""
        return self.get_llm_config().tool_calling # Return the Pydantic model

    def get_testing_groups_config(self) -> Optional[Dict[str, Any]]:
        """Get the testing groups configuration dictionary."""
        if self.config.testing_groups:
            return self.config.testing_groups.dict()
        return None

    def get_discussion_min_words(self) -> int:
        """Get min words for discussion from agent config."""
        return self.get_agent_config().get("discussion_min_words", 50)

    def get_discussion_max_words(self) -> int:
        """Get max words for discussion from agent config."""
        # Use the Pydantic model for consistency
        return self.config.simulation.discussion_length.max_words

    def is_local_backend(self) -> bool:
        """Check if the configured LLM backend is local."""
        llm_config = self.get_llm_config()
        return llm_config.backend == "local" # Use attribute access

    def get_llm_client_kwargs(self) -> Dict[str, Any]:
        """Get keyword arguments for LLM client initialization."""
        llm_config = self.get_llm_config()
        if llm_config.backend == "local": # Use attribute access
            # Return as dict as HuggingFaceClient likely expects kwargs dict
            return llm_config.local.model_dump() if llm_config.local else {}
        else:
            # Return as dict as RemoteLLMClient likely expects kwargs dict
            kwargs = llm_config.remote.model_dump() if llm_config.remote else {}
            
            # Resolve API key from environment variable
            if 'api_key_env' in kwargs:
                api_key_env_name = kwargs.pop('api_key_env')  # Remove api_key_env from kwargs
                api_key = os.getenv(api_key_env_name)
                if api_key:
                    kwargs['api_key'] = api_key
                    logging.info(f"Loaded API key from environment variable: {api_key_env_name}")
                else:
                    logging.warning(f"API key environment variable '{api_key_env_name}' not found or empty")
            
            return kwargs

    def get_backend_type(self) -> str:
        """Get the LLM backend type (local or remote)."""
        return self.get_llm_config().backend

    def copy_configs_to_output_dir(self, output_dir: Path):
        """Copy configuration files to output directory.
        
        Args:
            output_dir: Directory to copy config files to
        """
        configs_used_dir = output_dir / "configs_used"
        configs_used_dir.mkdir(exist_ok=True)
        
        # Copy all YAML files from refactored config directory
        for yaml_file in self.config_dir.glob("*.yaml"):
            try:
                shutil.copy(yaml_file, configs_used_dir / yaml_file.name)
                logging.info(f"Copied {yaml_file.name} to {configs_used_dir}")
            except Exception as e:
                logging.error(f"Error copying {yaml_file.name}: {e}")


# Global instance for backwards compatibility
_config_adapter = None

def get_config(config_dir: str = None) -> ConfigAdapter:
    """Get global configuration adapter instance.
    
    Args:
        config_dir: Directory containing refactored config files
        
    Returns:
        ConfigAdapter instance providing old ConfigManager interface
    """
    global _config_adapter
    if _config_adapter is None or config_dir is not None:
        _config_adapter = ConfigAdapter(config_dir)
    return _config_adapter
