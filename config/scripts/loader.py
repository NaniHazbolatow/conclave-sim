"""
Refactored configuration loader for the Conclave simulation system.
Loads configuration from separate YAML files for different components.
"""

import os
import yaml
from pathlib import Path
from typing import Dict, Any, Optional
from .models import RefactoredConfig, TestingGroupsConfig


class RefactoredConfigLoader:
    """Configuration loader for the refactored configuration system."""
    
    def __init__(self, config_dir: str = None):
        """Initialize the configuration loader.
        
        Args:
            config_dir: Directory containing configuration files. 
                       Defaults to the 'refactored' directory next to this file.
        """
        if config_dir is None:
            # Default to the parent directory (config/) where YAML files are located
            config_dir = Path(__file__).parent.parent  
        self.config_dir = Path(config_dir)
        
        # Define the component configuration files
        self.config_files = {
            "agent": "agent.yaml",
            "simulation": "simulation.yaml", 
            "output": "output.yaml",
            "testing_groups": "testing_groups.yaml"
        }
    
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
    
    def load_component_config(self, component: str) -> Dict[str, Any]:
        """Load configuration for a specific component.
        
        Args:
            component: Name of the component (agent, simulation, output, testing_groups)
            
        Returns:
            Dictionary containing the component configuration
        """
        if component not in self.config_files:
            raise ValueError(f"Unknown component: {component}. Available: {list(self.config_files.keys())}")
        
        filename = self.config_files[component]
        return self.load_yaml_file(filename)
    
    def load_config(self, include_testing_groups: bool = True) -> RefactoredConfig:
        """Load the complete configuration from all component files.
        
        Args:
            include_testing_groups: Whether to load testing groups configuration
            
        Returns:
            RefactoredConfig object containing all configuration data
        """
        # Load core component configurations
        agent_config = self.load_component_config("agent")
        simulation_config = self.load_component_config("simulation")
        output_config = self.load_component_config("output")
        
        # Combine all configurations
        config_data = {
            "agent": agent_config,
            "simulation": simulation_config,
            "output": output_config
        }
        
        # Load testing groups if requested and file exists
        if include_testing_groups:
            try:
                testing_groups_config = self.load_component_config("testing_groups")
                config_data["testing_groups"] = testing_groups_config["testing_groups"]
            except FileNotFoundError:
                # Testing groups are optional
                config_data["testing_groups"] = None
        
        # Create and validate the configuration object
        return RefactoredConfig(**config_data)
    
    def get_active_testing_group(self) -> Optional[Dict[str, Any]]:
        """Get the configuration for the currently active testing group.
        
        Returns:
            Dictionary containing the active testing group configuration,
            or None if testing groups are disabled or not configured.
        """
        try:
            testing_groups_config = self.load_component_config("testing_groups")["testing_groups"]
            
            if not testing_groups_config.get("enabled", False):
                return None
            
            active_group_name = testing_groups_config.get("active_group", "medium")
            return testing_groups_config.get(active_group_name)
            
        except (FileNotFoundError, KeyError):
            return None
    
    def get_simulation_config_with_overrides(self) -> Dict[str, Any]:
        """Get simulation configuration with testing group overrides applied.
        
        Returns:
            Dictionary containing the simulation configuration with any
            active testing group overrides applied.
        """
        # Load base simulation config
        sim_config = self.load_component_config("simulation")["simulation"]
        
        # Apply testing group overrides if active
        active_group = self.get_active_testing_group()
        if active_group and "override_settings" in active_group:
            overrides = active_group["override_settings"]
            
            # Apply overrides to simulation config
            if "num_cardinals" in overrides:
                sim_config["num_cardinals"] = overrides["num_cardinals"]
            if "discussion_group_size" in overrides:
                sim_config["discussion_group_size"] = overrides["discussion_group_size"]
            if "max_election_rounds" in overrides:
                sim_config["max_election_rounds"] = overrides["max_election_rounds"]
            if "supermajority_threshold" in overrides:
                sim_config["voting"]["supermajority_threshold"] = overrides["supermajority_threshold"]
        
        return sim_config

    def get_agent_config(self) -> Dict[str, Any]:
        """Get agent configuration.
        
        Returns:
            Dictionary containing the agent configuration.
        """
        return self.load_component_config("agent")

    def get_output_config(self) -> Dict[str, Any]:
        """Get output configuration.

        Returns:
            Dictionary containing the output configuration.
        """
        return self.load_component_config("output")


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


def load_config(config_dir: str = None, include_testing_groups: bool = True) -> RefactoredConfig:
    """Convenience function to load the complete configuration.
    
    Args:
        config_dir: Directory containing configuration files
        include_testing_groups: Whether to load testing groups configuration
        
    Returns:
        RefactoredConfig object containing all configuration data
    """
    loader = get_config_loader(config_dir)
    return loader.load_config(include_testing_groups)
