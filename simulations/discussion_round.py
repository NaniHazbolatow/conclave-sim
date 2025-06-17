import sys
import os
from pathlib import Path
import warnings

# Suppress urllib3 LibreSSL warning on macOS
warnings.filterwarnings("ignore", message="urllib3 v2 only supports OpenSSL 1.1.1+")

# Add the parent directory to the Python path so we can import conclave
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from config.scripts import get_config  # Updated to use refactored config system
from conclave.environments.conclave_env import ConclaveEnv
from conclave.agents.base import Agent
from conclave.visualization.cardinal_visualizer import CardinalVisualizer
import pandas as pd
import logging
import datetime
import shutil

# Create a logger for your module
logger = logging.getLogger("conclave.system")  # Changed from __name__


def main():
    # print("discussion_round.py: main() started")  # DEBUG PRINT
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    base_output_dir = project_root / "simulations" / "outputs" / timestamp
    # print(f"discussion_round.py: base_output_dir = {base_output_dir}")  # DEBUG PRINT

    logs_dir = base_output_dir / "logs"
    viz_dir = base_output_dir / "visualizations"
    results_dir = base_output_dir / "results"
    configs_used_dir = base_output_dir / "configs_used"

    logs_dir.mkdir(parents=True, exist_ok=True)
    viz_dir.mkdir(parents=True, exist_ok=True)
    results_dir.mkdir(parents=True, exist_ok=True)
    configs_used_dir.mkdir(parents=True, exist_ok=True)

    original_cwd = os.getcwd()
    # print(f"discussion_round.py: Original CWD = {original_cwd}")  # DEBUG PRINT
    try:
        os.chdir(project_root)  # Change CWD first
        # print(f"discussion_round.py: Changed CWD to {project_root}")  # DEBUG PRINT

        # Get config manager instance (it will find config.yaml in the new CWD)
        # print("discussion_round.py: Attempting to get config manager...")  # DEBUG PRINT
        config_manager = get_config()  # Load refactored configuration via adapter
        # print("discussion_round.py: Config manager obtained.")  # DEBUG PRINT
        config_manager.copy_configs_to_output_dir(base_output_dir)
        # print("discussion_round.py: Configs copied to output dir.")  # DEBUG PRINT

        # Setup logging using config adapter
        # print("discussion_round.py: Attempting to initialize logging...")  # DEBUG PRINT
        config_manager.initialize_logging(dynamic_logs_dir=logs_dir)
        # print("discussion_round.py: Logging initialized.")  # DEBUG PRINT

        # Load configuration
        config = config_manager  # Use the instance we already have
        num_cardinals = config.get_num_cardinals()
        discussion_group_size = config.get_discussion_group_size()  # New relevant config
        max_election_rounds = config.get_max_election_rounds()

        logger.info(
            f"Starting simulation with num_cardinals={num_cardinals}, "
            f"discussion_group_size={discussion_group_size}, "  # Updated log
            f"max_election_rounds={max_election_rounds}"
        )

        # Create the environment
        env = ConclaveEnv()  # Removed viz_dir argument
        # print("discussion_round.py: ConclaveEnv initialized.")  # DEBUG PRINT

        # Generate initial internal stances for all agents before starting
        env.generate_initial_stances()

        election_round = 0
        winner_found = False

        while election_round < max_election_rounds and not winner_found:
            election_round += 1
            logger.info(f"\n=== ELECTION ROUND {election_round} ===")
            print(f"\nStarting Election Round {election_round}")

            # Set the voting round in the environment to match the election round
            env.votingRound = election_round

            # REMOVED: env.reset_discussion_speakers_for_new_election_round()

            # Run a full discussion phase (which now handles groups internally)
            logger.info(f"\n--- Discussion Phase (Voting Round {election_round}) ---")
            # print(
            #     f"Attempting to run discussion round (Election Round {election_round})"
            # )
            env.run_discussion_round()  # No num_speakers argument needed
            # print(
            #     f"Discussion round completed (Election Round {election_round})"
            # )

            # After all discussions, run voting
            # print(
            #     f"Attempting to run voting round (Election Round {election_round})"
            # )
            winner_found = env.run_voting_round()
            # print(
            #     f"Voting round completed (Election Round {election_round}). Winner found: {winner_found}"
            # )
            print(
                f"\n📊 Election Round {election_round} completed. Winner found: {'Yes' if winner_found else 'No'}\n"
            )

        if winner_found:
            print(f"✅ CONCLAVE COMPLETE!")
            print(
                f"🎉 Winner: Cardinal {env.agents[env.winner].name} elected as Pope after {election_round} election rounds"
            )
        else:
            print(f"❌ CONCLAVE INCOMPLETE")
            print(
                f"No winner found after {max_election_rounds} election rounds. Simulation ended without papal election."
            )

        # Save results
        results_data = {
            "timestamp": timestamp,
            "winner_cardinal_id": env.winner,
            "winner_name": env.agents[env.winner].name if env.winner is not None else "N/A",
            "total_election_rounds": election_round,
            "max_election_rounds_config": max_election_rounds,
            "discussion_group_size_config": discussion_group_size,
            # Add other relevant results here
        }
        results_file_path = results_dir / "simulation_summary.json"
        try:
            with open(results_file_path, "w") as f:
                import json

                json.dump(results_data, f, indent=4)
            logger.info(f"Simulation summary saved to {results_file_path}")
        except Exception as e:
            logger.error(f"Error saving simulation summary: {e}")

        # Generate stance progression visualization using CardinalVisualizer
        # First ensure all embeddings are updated
        env.update_all_embeddings_after_simulation()
        
        # Log embedding evolution summary
        evolution_summary = env.get_embedding_evolution_summary()
        logger.info(f"Embedding evolution summary: {evolution_summary.get('agents_with_embeddings', 0)}/{evolution_summary.get('total_agents', 0)} agents have embeddings")
        if evolution_summary.get('average_movements'):
            logger.info(f"Average stance movement: {evolution_summary['average_movements']['mean']:.4f} ± {evolution_summary['average_movements']['std']:.4f}")
        
        visualizer = CardinalVisualizer(
            config_manager, viz_dir=str(viz_dir)
        )  # Pass config_manager
        visualizer.generate_stance_visualization_from_env(env)

    finally:
        # Restore the original working directory
        os.chdir(original_cwd)


if __name__ == "__main__":
    # print("discussion_round.py: Script starting (__name__ == '__main__')")  # DEBUG PRINT
    main()
    # print("discussion_round.py: Script finished.")  # DEBUG PRINT
