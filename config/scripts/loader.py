"""
Refactored configuration loader for the Conclave simulation system.
Loads configuration from the new consolidated config.yaml file,
with fallback to the old multi-file structure for backward compatibility.
"""

import os
import yaml
from pathlib import Path
from typing import Dict, Any, Optional
from dotenv import load_dotenv
from .models import RefactoredConfig


class RefactoredConfigLoader:
    """Configuration loader for the refactored configuration system."""
    
    def __init__(self, config_dir: str = None):
        """Initialize the configuration loader.
        
        Args:
            config_dir: Directory containing configuration files. 
                       Defaults to the project root directory.
        """
        # Load environment variables from .env file
        if config_dir is None:
            # Default to the project root directory
            config_dir = Path(__file__).parent.parent.parent
        self.config_dir = Path(config_dir)
        
        # Load .env file from project root
        env_file = self.config_dir / ".env"
        if env_file.exists():
            load_dotenv(env_file)
        
        # Define the old component configuration files (for backward compatibility)
        self.legacy_config_files = {
            "agent": "config/agent.yaml",
            "simulation": "config/simulation.yaml", 
            "output": "config/output.yaml",
            "predefined_groups": "config/groups.yaml",
            "simulation_settings": "config/simulation_settings.yaml"
        }
        
        # New consolidated config file
        self.main_config_file = "config.yaml"
    
    def load_yaml_file(self, filename: str) -> Dict[str, Any]:
        """Load a YAML configuration file.
        
        Args:
            filename: Name of the YAML file to load
            
        Returns:
            Dictionary containing the configuration data
            
        Raises:
            FileNotFoundError: If the configuration file doesn't exist
        """
        file_path = self.config_dir / filename
        
        if not file_path.exists():
            raise FileNotFoundError(f"Configuration file not found: {file_path}")
        
        with open(file_path, 'r') as f:
            return yaml.safe_load(f)
    
    def load_legacy_component_config(self, component: str) -> Dict[str, Any]:
        """Load configuration for a specific component from legacy files.
        
        Args:
            component: Name of the component (agent, simulation, output, predefined_groups, simulation_settings)
            
        Returns:
            Dictionary containing the component configuration
        """
        if component not in self.legacy_config_files:
            raise ValueError(f"Unknown component: {component}. Available: {list(self.legacy_config_files.keys())}")
        
        filename = self.legacy_config_files[component]
        return self.load_yaml_file(filename)
    
    def get_num_cardinals(self) -> int:
        """Get number of cardinals from active group or default.
        
        Returns:
            Number of cardinals to use in simulation
        """
        # First check if we have an active group configuration
        config = self.load_config()
            
        # If groups are configured, get num_cardinals from active group
        if config.groups and config.groups.active:
            try:
                predefined_groups_path = self.config_dir / "config" / "groups.yaml"
                if predefined_groups_path.exists():
                    with open(predefined_groups_path, 'r') as f:
                        groups_data = yaml.safe_load(f)
                    
                    active_group = groups_data["predefined_groups"].get(config.groups.active)
                    if active_group:
                        return active_group["total_cardinals"]
            except (FileNotFoundError, KeyError):
                pass
        
        # Fall back to default value
        return 5
        return self.load_yaml_file(filename)
    
    def load_config(self) -> RefactoredConfig:
        """Load the complete configuration from the new consolidated file or legacy files.
        
        Returns:
            RefactoredConfig object containing all configuration data
        """
        # First try to load from the new consolidated config.yaml
        main_config_path = self.config_dir / self.main_config_file
        if main_config_path.exists():
            return self._load_new_config()
        else:
            # Fall back to legacy multi-file structure
            return self._load_legacy_config()
    
    def _load_new_config(self) -> RefactoredConfig:
        """Load configuration from the new consolidated config.yaml file."""
        config_data = self.load_yaml_file(self.main_config_file)
        
        # Load predefined groups separately (still in their own file)
        try:
            predefined_groups_config = self.load_yaml_file("config/groups.yaml")
            config_data["predefined_groups"] = predefined_groups_config["predefined_groups"]
        except FileNotFoundError:
            config_data["predefined_groups"] = None
        
        return RefactoredConfig(**config_data)
    
    def _load_legacy_config(self) -> RefactoredConfig:
        """Load configuration from the legacy multi-file structure."""
        # Load core component configurations
        agent_config = self.load_legacy_component_config("agent")
        simulation_config = self.load_legacy_component_config("simulation")
        output_config = self.load_legacy_component_config("output")
        
        # Combine all configurations
        config_data = {
            "models": agent_config,  # Map 'agent' to 'models' in new structure
            "simulation": simulation_config,
            "output": output_config
        }
        
        # Load predefined groups
        try:
            predefined_groups_config = self.load_legacy_component_config("predefined_groups")
            config_data["predefined_groups"] = predefined_groups_config["predefined_groups"]
        except FileNotFoundError:
            config_data["predefined_groups"] = None

        # Load simulation settings
        try:
            simulation_settings_config = self.load_legacy_component_config("simulation_settings")
            config_data["simulation_settings"] = simulation_settings_config["simulation_settings"]
        except FileNotFoundError:
            config_data["simulation_settings"] = None
        
        # Create and validate the configuration object
        return RefactoredConfig(**config_data)
    
    def get_active_group_config(self) -> Optional[Dict[str, Any]]:
        """Get the configuration for the currently active predefined group.
        
        Returns:
            Dictionary containing the active group configuration,
            or None if not configured.
        """
        config = self.load_config()
        
        if not config.groups or not config.groups.active:
            return None
            
        try:
            # Load predefined groups
            predefined_groups_path = self.config_dir / "config" / "groups.yaml"
            if predefined_groups_path.exists():
                with open(predefined_groups_path, 'r') as f:
                    groups_data = yaml.safe_load(f)
                
                group_config = groups_data["predefined_groups"].get(config.groups.active)
                
                # If group exists but is missing detailed data, generate it
                if group_config and 'cardinal_ids' not in group_config:
                    group_config = self._generate_group_details(group_config)
                
                return group_config
            
        except (FileNotFoundError, KeyError):
            pass
            
        return None

    def _generate_group_details(self, group_config: Dict[str, Any]) -> Dict[str, Any]:
        """Generate detailed group data from basic configuration.
        
        Args:
            group_config: Basic group config with total_cardinals and num_candidates
            
        Returns:
            Complete group configuration with generated data
        """
        try:
            # Import the generation function
            import sys
            from pathlib import Path
            scripts_path = self.config_dir / "scripts"
            if str(scripts_path) not in sys.path:
                sys.path.insert(0, str(scripts_path))
            
            from generate_predefined_groups import generate_group_data
            
            # Generate the detailed data
            generated_data = generate_group_data(
                total_cardinals=group_config.get('total_cardinals'),
                num_candidates=group_config.get('num_candidates', 2),
                base_path=self.config_dir
            )
            
            # Merge with original config
            complete_config = group_config.copy()
            complete_config.update(generated_data)
            
            return complete_config
            
        except Exception as e:
            logger = logging.getLogger(__name__)
            logger.warning(f"Failed to generate group details: {e}")
            return group_config

    def get_simulation_config_with_overrides(self) -> Dict[str, Any]:
        """Get simulation configuration with any overrides applied.
        
        Returns:
            Dictionary containing the simulation configuration
        """
        config = self.load_config()
        sim_config = config.simulation.model_dump()
        
        # Add num_cardinals from active group if available
        num_cardinals = self.get_num_cardinals()
        sim_config["num_cardinals"] = num_cardinals
        
        return sim_config

    def get_agent_config(self) -> Dict[str, Any]:
        """Get agent/models configuration.
        
        Returns:
            Dictionary containing the agent configuration.
        """
        config = self.load_config()
        return config.models.model_dump()

    def get_output_config(self) -> Dict[str, Any]:
        """Get output configuration.

        Returns:
            Dictionary containing the output configuration.
        """
        config = self.load_config()
        return config.output.model_dump()


# Global configuration loader instance
_loader = None

def get_config_loader(config_dir: str = None) -> RefactoredConfigLoader:
    """Get the global configuration loader instance.
    
    Args:
        config_dir: Directory containing configuration files
        
    Returns:
        RefactoredConfigLoader instance
    """
    global _loader
    if _loader is None or config_dir is not None:
        _loader = RefactoredConfigLoader(config_dir)
    return _loader


def load_config(config_dir: str = None) -> RefactoredConfig:
    """Convenience function to load the complete configuration.
    
    Args:
        config_dir: Directory containing configuration files
        
    Returns:
        RefactoredConfig object containing all configuration data
    """
    loader = get_config_loader(config_dir)
    return loader.load_config()
