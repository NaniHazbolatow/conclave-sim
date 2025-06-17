from conclave.environments.conclave_env import ConclaveEnv
from conclave.agents.base import Agent
import pandas as pd
import logging
import datetime
import os
import shutil
from pathlib import Path
from config.scripts.adapter import ConfigAdapter # Import ConfigAdapter

# Create a logger for your module
logger = logging.getLogger("conclave.system") # Changed from __name__

def main():
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    # Use Path for directory manipulations
    base_output_dir = Path("simulations") / "outputs" / timestamp
    
    logs_dir = base_output_dir / "logs"
    viz_dir = base_output_dir / "visualizations"
    results_dir = base_output_dir / "results"
    configs_used_dir = base_output_dir / "configs_used"
    
    logs_dir.mkdir(parents=True, exist_ok=True)
    viz_dir.mkdir(parents=True, exist_ok=True)
    results_dir.mkdir(parents=True, exist_ok=True)
    configs_used_dir.mkdir(parents=True, exist_ok=True)

    # Get config adapter instance
    config_adapter = ConfigAdapter() # Use ConfigAdapter directly

    # Setup logging using ConfigAdapter
    config_adapter.initialize_logging(dynamic_logs_dir=logs_dir)

    # Save the config files
    try:
        for config_file in config_adapter.config_dir.glob('*.yaml'): # Iterate over YAML files in the config directory
            shutil.copy(config_file, configs_used_dir / config_file.name)
            logger.info(f"Copied {config_file.name} to {configs_used_dir}")
    except Exception as e:
        logger.error(f"Error copying configuration files: {e}")

    # Create the environment
    env = ConclaveEnv(viz_dir=str(viz_dir)) # Pass viz_dir to ConclaveEnv

    # Agents are initialized in ConclaveEnv.__init__, no need for env.load_agents_from_config()
    
    # Generate initial internal stances for all agents before starting
    env.generate_initial_stances()
    
    winner_found = False
    current_round = 0 # Initialize current_round
    while not winner_found and current_round < env.max_election_rounds:
        logger.info(f"--- Starting Election Round {current_round + 1} ---")
        
        # Run discussion round
        logger.info("--- Running Discussion Round ---")
        env.run_discussion_round() # Call the discussion round method
        
        # Run voting round
        logger.info("--- Running Voting Round ---")
        winner_found = env.run_voting_round()
        
        if winner_found:
            logger.info(f"Winner found in round {current_round + 1}: Cardinal {env.winner} - {env.agents[env.winner].name}")
        else:
            logger.info(f"No winner yet after round {current_round + 1}. Current Pope: {env.current_pope if hasattr(env, 'current_pope') else 'N/A'}")
            if current_round + 1 >= env.max_election_rounds:
                logger.info("Maximum election rounds reached. No winner declared by supermajority.")
                # Handle scenario where no winner is found after max rounds (e.g., select current pope or other logic)
                # For now, we'll just break and the summary will reflect no winner by supermajority.
                break 
        current_round += 1

    # Update results_data to use env.votingRound which is incremented in run_voting_round
    # or use the local current_round if that's more appropriate
    final_round_count = env.votingRound # or current_round, depending on desired logic

    if env.winner is not None:
        winner_name = env.agents[env.winner].name
        logger.info(f"Final Winner: Cardinal {env.winner} - {winner_name}")
    else:
        winner_name = "N/A (No supermajority winner)"
        logger.info("No winner by supermajority after all rounds.")
    
    # Save results
    results_data = {
        "timestamp": timestamp,
        "winner_cardinal_id": env.winner if env.winner is not None else "N/A",
        "winner_name": winner_name,
        "total_rounds_conducted": final_round_count, 
        "max_election_rounds_setting": env.max_election_rounds,
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
    
    # Generate stance progression visualization
    try:
        from conclave.visualization.cardinal_visualizer import CardinalVisualizer
        
        # Ensure all embeddings are updated
        env.update_all_embeddings_after_simulation()
        
        # Log embedding evolution summary
        evolution_summary = env.get_embedding_evolution_summary()
        logger.info(f"Embedding evolution summary: {evolution_summary.get('agents_with_embeddings', 0)}/{evolution_summary.get('total_agents', 0)} agents have embeddings")
        if evolution_summary.get('average_movements'):
            logger.info(f"Average stance movement: {evolution_summary['average_movements']['mean']:.4f} ± {evolution_summary['average_movements']['std']:.4f}")
        
        # Generate visualization
        visualizer = CardinalVisualizer(config_adapter, viz_dir=str(viz_dir))
        visualizer.generate_stance_visualization_from_env(env)
        logger.info("Stance progression visualization generated successfully")
        
    except Exception as e:
        logger.error(f"Failed to generate visualization: {e}")


if __name__ == "__main__":
    main()
