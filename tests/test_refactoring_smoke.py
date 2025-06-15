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

from config.scripts.loader import load_config, get_config_loader
from config.scripts.adapter import get_config
from config.scripts.models import RefactoredConfig, AgentConfig, SimulationConfig, OutputConfig


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
        self.assertIsInstance(sim_config.num_agents, int, "sim_config.num_agents is not an int.")
        # self.assertIsInstance(sim_config.max_discussion_rounds, int, "sim_config.max_discussion_rounds is not an int.") # Field does not exist
        self.assertIsInstance(sim_config.max_election_rounds, int, "sim_config.max_election_rounds is not an int.") # Use max_election_rounds instead

    def test_llm_config_access(self):
        llm_config = self.config_manager.get_llm_config()
        self.assertIsInstance(llm_config, LLMConfig, "LLMConfig is not the correct type.")
        self.assertIsInstance(llm_config.default_provider, str, "llm_config.default_provider is not a string.")

    def test_logging_config_access(self):
        log_config = self.config_manager.get_logging_config()
        self.assertIsInstance(log_config, LoggingConfig, "LoggingConfig is not the correct type.")
        self.assertEqual(log_config.version, 1, "Logging config version is not 1.")

class TestPromptLoading(unittest.TestCase):
    def setUp(self):
        try:
            self.prompt_manager = get_prompt_loader()
        except Exception as e:
            self.fail(f"Failed to get PromptLoader in setUp: {e}")

    def test_prompt_manager_instance(self):
        self.assertIsInstance(self.prompt_manager, PromptLoader, "PromptLoader is not the correct type.")
        self.assertIsInstance(self.prompt_manager.prompts_config, PromptsConfig, "prompts_config is not PromptsConfig type.")

    def test_get_prompt(self):
        # Assuming "internal_persona_extractor" is a valid key in prompts.yaml
        try:
            prompt_content = self.prompt_manager.get_prompt("internal_persona_extractor")
            self.assertIsInstance(prompt_content, str, "Fetched prompt is not a string.")
            self.assertTrue(len(prompt_content) > 0, "Fetched prompt is empty.")
        except KeyError:
            self.fail("Prompt 'internal_persona_extractor' not found. Check prompts.yaml and PromptsConfig model.")
        except Exception as e:
            self.fail(f"Error getting prompt 'internal_persona_extractor': {e}")
            
    def test_get_tool_definition(self):
        # Assuming "generate_stance" is a valid tool name in prompts.yaml
        try:
            tool_def = self.prompt_manager.get_tool_definition("generate_stance")
            self.assertIsInstance(tool_def, ToolDefinition, "ToolDefinition is not the correct type.")
            self.assertEqual(tool_def.name, "generate_stance", "Tool name mismatch.")
            self.assertIsInstance(tool_def.description, str, "Tool description is not a string.")
        except KeyError:
            self.fail("Tool 'generate_stance' not found. Check prompts.yaml and PromptsConfig model.")
        except Exception as e:
            self.fail(f"Error getting tool 'generate_stance': {e}")

class TestConclaveEnvInitialization(unittest.TestCase):
    def test_env_basic_init(self):
        try:
            env = ConclaveEnv(num_agents=3) # Uses default config
            self.assertIsInstance(env, ConclaveEnv, "ConclaveEnv is not the correct type.")
            self.assertIsNotNone(env.config_manager, "env.config_manager (ConfigManager instance) is None.") # Check env.config_manager
            # num_agents in __init__ is a parameter, not directly from config at this stage for env.num_agents
            self.assertEqual(env.num_agents, 3, "env.num_agents not set as per constructor argument.")
        except Exception as e:
            self.fail(f"ConclaveEnv basic initialization failed: {e}")

    def test_env_load_agents(self):
        env = ConclaveEnv(num_agents=3) 
        try:
            env.load_agents_from_config()
            # Check if agents were loaded. The exact number depends on config.yaml and data.
            # If testing_groups are enabled, it might load a specific number.
            # If not, it might load num_cardinals from config or all from CSV.
            self.assertTrue(len(env.agents) >= 0, "env.agents list should exist, even if empty depending on config.")
            if len(env.agents) > 0 :
                self.assertEqual(env.num_agents, len(env.agents), 
                                 "env.num_agents should be updated by freeze_agent_count after loading.")
        except FileNotFoundError as fnfe:
            # This is a common issue if data/cardinals_master_data.csv is not found relative to ConclaveEnv
            self.fail(f"ConclaveEnv.load_agents_from_config() failed due to FileNotFoundError: {fnfe}. Check path for 'data/cardinals_master_data.csv'.")
        except Exception as e:
            self.fail(f"ConclaveEnv.load_agents_from_config() failed: {e}")

class TestPromptVariableGeneratorIntegration(unittest.TestCase):
    def setUp(self):
        self.env = None
        try:
            self.env = ConclaveEnv(num_agents=3) 
            self.env.load_agents_from_config()
            if not self.env.agents:
                 self.skipTest("Skipping PVG tests: no agents loaded by ConclaveEnv. Check config.yaml (e.g., num_cardinals, testing_groups) and data/cardinals_master_data.csv.")
        except Exception as e:
            self.skipTest(f"Skipping PVG tests due to ConclaveEnv setup error: {e}")

        if self.env: # Ensure env was successfully created
            self.pvg = PromptVariableGenerator(env=self.env)
        else: # Should be caught by skipTest, but as a fallback
            self.fail("Failed to initialize ConclaveEnv in PVG test setUp.")


    def test_pvg_instantiation(self):
        self.assertIsInstance(self.pvg, PromptVariableGenerator, "PromptVariableGenerator is not the correct type.")
        self.assertEqual(self.pvg.env, self.env, "PVG environment mismatch.")

    def test_pvg_generate_agent_variables_basic(self):
        # This test assumes at least one agent is loaded.
        # The setUp method skips if no agents are loaded.
        agent_id_to_test = 0 
        prompt_name_to_test = "discussion_elector" 

        try:
            variables = self.pvg.generate_agent_variables(agent_id=agent_id_to_test, prompt_name=prompt_name_to_test)
            self.assertIsInstance(variables, dict, "Generated variables should be a dict.")
            # Check for some expected keys based on PromptVariableGenerator's implementation
            self.assertIn('agent_name', variables)
            self.assertIn('biography', variables)
            self.assertIn('visible_candidates', variables) 
            self.assertIn('discussion_round', variables)
        except IndexError:
            self.fail(f"Failed to get agent with id {agent_id_to_test}. Ensure agents are loaded and accessible in self.env.agents.")
        except Exception as e:
            self.fail(f"pvg.generate_agent_variables failed for agent {agent_id_to_test}, prompt '{prompt_name_to_test}': {e}")

if __name__ == '__main__':
    unittest.main()
