from conclave.environments.conclave_env import ConclaveEnv
from conclave.agents.base import Agent
import pandas as pd
import logging
import datetime
import os
import shutil
from pathlib import Path # Added
from config.scripts import get_config  # Updated to use refactored config system

# Create a logger for your module
logger = logging.getLogger("conclave.system") # Changed from __name__

def main():
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    base_output_dir = Path("simulations") / "outputs" / timestamp
    
    logs_dir = base_output_dir / "logs"
    viz_dir = base_output_dir / "visualizations"
    results_dir = base_output_dir / "results"
    configs_used_dir = base_output_dir / "configs_used"
    
    logs_dir.mkdir(parents=True, exist_ok=True)
    viz_dir.mkdir(parents=True, exist_ok=True)
    results_dir.mkdir(parents=True, exist_ok=True)
    configs_used_dir.mkdir(parents=True, exist_ok=True)

    # Get config manager instance
    config_manager = get_config() # Assuming config.yaml is in CWD

    # Setup logging using ConfigManager
    config_manager.initialize_logging(dynamic_logs_dir=logs_dir) # New method

    # Save the config file
    try:
        shutil.copy(config_manager.config_path, configs_used_dir / config_manager.config_path.name)
        logger.info(f"Copied {config_manager.config_path.name} to {configs_used_dir}")
    except FileNotFoundError:
        logger.error(f"{config_manager.config_path.name} not found, cannot copy.")
    except Exception as e:
        logger.error(f"Error copying {config_manager.config_path.name}: {e}")

    # Create the environment
    env = ConclaveEnv(viz_dir=str(viz_dir)) # Pass viz_dir

    # Agents are automatically loaded during environment initialization
    logger.info(f"Environment created with {len(env.agents)} agents")
    
    # Generate initial internal stances for all agents before starting
    env.generate_initial_stances()
    
    winner_id, vote_counts = env.run_voting_round()  # FIXED: Properly unpack the tuple
    winner_found = winner_id is not None  # FIXED: Determine winner status from winner_id
    
    # Save results
    results_data = {
        "timestamp": timestamp,
        "winner_cardinal_id": env.winner,
        "winner_name": env.agents[env.winner].name if env.winner is not None else "N/A",
        "total_rounds": env.current_round, # single round, so should be 1 or 0
        "winner_found": winner_found,
        # Add other relevant results here
    }
    results_file_path = results_dir / "simulation_summary.json" # Use Path object
    try:
        with open(results_file_path, 'w') as f:
            import json
            json.dump(results_data, f, indent=4)
        logger.info(f"Simulation summary saved to {results_file_path}")
    except Exception as e:
        logger.error(f"Error saving simulation summary: {e}")

if __name__ == "__main__":
    main()
