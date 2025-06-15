"""
Smoke tests for the refactored configuration system.
Tests the new modular configuration structure and functionality.
"""

import unittest
import os
import sys
from pathlib import Path

# Add project root to path
PROJECT_ROOT_DIR = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT_DIR))

from config.refactored.scripts.loader import load_config, get_config_loader
from config.refactored.scripts.adapter import get_config
from config.refactored.models import RefactoredConfig, AgentConfig, SimulationConfig, OutputConfig


class TestRefactoredConfigLoading(unittest.TestCase):
    """Test the refactored configuration system."""

    def setUp(self):
        """Set up test fixtures."""
        try:
            self.config = load_config()
            self.config_adapter = get_config()
        except Exception as e:
            self.fail(f"Failed to load refactored config in setUp: {e}")

    def test_config_loading(self):
        """Test that configuration loads successfully."""
        self.assertIsInstance(self.config, RefactoredConfig, "Config is not RefactoredConfig type.")
        self.assertIsNotNone(self.config.agent, "Agent config is None.")
        self.assertIsNotNone(self.config.simulation, "Simulation config is None.")
        self.assertIsNotNone(self.config.output, "Output config is None.")

    def test_agent_config_access(self):
        """Test agent configuration access."""
        agent_config = self.config.agent
        self.assertIsInstance(agent_config, AgentConfig, "AgentConfig is not the correct type.")
        
        # Test LLM configuration
        self.assertIn(agent_config.llm.backend, ["local", "remote"], "Invalid LLM backend.")
        self.assertGreater(agent_config.llm.temperature, 0, "Temperature should be > 0.")
        self.assertGreater(agent_config.llm.max_tokens, 0, "Max tokens should be > 0.")
        
        # Test embeddings configuration
        self.assertIsNotNone(agent_config.embeddings.model_name, "Embeddings model name is None.")

    def test_simulation_config_access(self):
        """Test simulation configuration access."""
        sim_config = self.config.simulation
        self.assertIsInstance(sim_config, SimulationConfig, "SimulationConfig is not the correct type.")
        
        # Test core simulation parameters
        self.assertGreater(sim_config.num_cardinals, 0, "num_cardinals should be > 0.")
        self.assertGreater(sim_config.discussion_group_size, 0, "discussion_group_size should be > 0.")
        self.assertGreater(sim_config.max_election_rounds, 0, "max_election_rounds should be > 0.")
        
        # Test voting configuration
        self.assertGreaterEqual(sim_config.voting.supermajority_threshold, 0.5, 
                               "supermajority_threshold should be >= 0.5.")
        self.assertLessEqual(sim_config.voting.supermajority_threshold, 1.0, 
                            "supermajority_threshold should be <= 1.0.")

    def test_output_config_access(self):
        """Test output configuration access."""
        output_config = self.config.output
        self.assertIsInstance(output_config, OutputConfig, "OutputConfig is not the correct type.")
        
        # Test logging configuration
        self.assertIn(output_config.logging.log_level, 
                     ["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"], 
                     "Invalid log level.")
        
        # Test results configuration
        self.assertIsInstance(output_config.results.save_individual_votes, bool, 
                             "save_individual_votes should be boolean.")

    def test_testing_groups_config(self):
        """Test testing groups configuration if present."""
        if self.config.testing_groups:
            testing_groups = self.config.testing_groups
            self.assertIsInstance(testing_groups.enabled, bool, "enabled should be boolean.")
            self.assertIn(testing_groups.active_group, ["small", "medium", "large"], 
                         "Invalid active group.")
            
            # Test active group access
            loader = get_config_loader()
            active_group = loader.get_active_testing_group()
            if active_group:
                self.assertGreater(active_group["total_cardinals"], 0, 
                                 "Active group should have > 0 cardinals.")

    def test_config_adapter_compatibility(self):
        """Test that the config adapter provides backwards compatibility."""
        # Test that adapter provides old interface methods
        self.assertTrue(hasattr(self.config_adapter, 'get_num_cardinals'), 
                       "Adapter missing get_num_cardinals method.")
        self.assertTrue(hasattr(self.config_adapter, 'get_discussion_group_size'), 
                       "Adapter missing get_discussion_group_size method.")
        self.assertTrue(hasattr(self.config_adapter, 'get_max_election_rounds'), 
                       "Adapter missing get_max_election_rounds method.")
        
        # Test that methods return valid values
        self.assertGreater(self.config_adapter.get_num_cardinals(), 0, 
                          "get_num_cardinals should return > 0.")
        self.assertGreater(self.config_adapter.get_discussion_group_size(), 0, 
                          "get_discussion_group_size should return > 0.")
        self.assertGreater(self.config_adapter.get_max_election_rounds(), 0, 
                          "get_max_election_rounds should return > 0.")

    def test_simulation_config_with_overrides(self):
        """Test that testing group overrides are applied correctly."""
        loader = get_config_loader()
        sim_config_with_overrides = loader.get_simulation_config_with_overrides()
        
        self.assertIsInstance(sim_config_with_overrides, dict, 
                             "Simulation config with overrides should be dict.")
        self.assertIn("num_cardinals", sim_config_with_overrides, 
                     "Missing num_cardinals in override config.")
        self.assertGreater(sim_config_with_overrides["num_cardinals"], 0, 
                          "num_cardinals should be > 0 in override config.")


class TestConfigValidation(unittest.TestCase):
    """Test configuration validation and error handling."""

    def test_config_validation(self):
        """Test that configuration validation works."""
        config = load_config()
        
        # Test that Pydantic validation is working
        with self.assertRaises(ValueError):
            # This should fail validation
            config.simulation.num_cardinals = -1


if __name__ == "__main__":
    unittest.main()
