import sys
import os

os.environ["TOKENIZERS_PARALLELISM"] = "false"

from pathlib import Path
import warnings
import argparse

# Suppress urllib3 LibreSSL warning on macOS
warnings.filterwarnings("ignore", message="urllib3 v2 only supports OpenSSL 1.1.1+")

# Add the parent directory to the Python path so we can import conclave
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from config.scripts import get_config  # Updated to use refactored config system
from config.scripts.models import RefactoredConfig, GroupsConfig, SimulationConfig
from conclave.environments.conclave_env import ConclaveEnv
from conclave.agents.base import Agent
from conclave.visualization.cardinal_visualizer import CardinalVisualizer
from conclave.utils.logging import setup_stance_logger
import pandas as pd
import logging
import datetime
import shutil

# Create a logger for your module
logger = logging.getLogger("conclave.system")  # Changed from __name__


def parse_arguments():
    """Parse command line arguments for main simulation settings."""
    parser = argparse.ArgumentParser(
        description='Run Conclave Simulation',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Run with medium group and 10 rounds
  python run.py --group medium --max-rounds 10
  
  # Run with custom rationality and temperature
  python run.py --rationality 0.9 --temperature 0.5
  
  # Run with parallel processing disabled
  python run.py --no-parallel
  
  # Run with custom discussion group size
  python run.py --discussion-size 5
        """
    )
    
    # Core simulation parameters
    parser.add_argument('--group', type=str, choices=['small', 'medium', 'large', 'xlarge', 'full'],
                       help='Active predefined group (overrides config.yaml groups.active)')
    parser.add_argument('--max-rounds', type=int, metavar='N',
                       help='Maximum simulation rounds (Range: 1-50)')
    parser.add_argument('--parallel', action='store_true', default=None,
                       help='Enable parallel processing')
    parser.add_argument('--no-parallel', action='store_true',
                       help='Disable parallel processing')
    
    # Agent behavior parameters  
    parser.add_argument('--rationality', type=float, metavar='R',
                       help='Agent rationality (Range: 0.0=random to 1.0=fully rational)')
    parser.add_argument('--temperature', type=float, metavar='T', 
                       help='LLM temperature (Range: 0.0=deterministic to 2.0=creative)')
    parser.add_argument('--discussion-size', type=int, metavar='N',
                       help='Number of agents per discussion group')
    
    # Output control
    parser.add_argument('--output-dir', type=str, metavar='DIR',
                       help='Custom output directory (default: auto-generated timestamp)')
    parser.add_argument('--log-level', choices=['DEBUG', 'INFO', 'WARNING', 'ERROR'],
                       help='Logging level (overrides config.yaml)')
    parser.add_argument('--no-viz', action='store_true',
                       help='Disable visualization generation')
    
    return parser.parse_args()


def apply_cli_overrides(config: RefactoredConfig, args: argparse.Namespace) -> RefactoredConfig:
    """Apply command line argument overrides to configuration.
    
    Args:
        config: Base configuration loaded from files
        args: Parsed command line arguments
        
    Returns:
        Updated configuration with CLI overrides applied
    """
    # Handle parallel/no-parallel flags
    if args.no_parallel:
        config.simulation.enable_parallel = False
    elif args.parallel:
        config.simulation.enable_parallel = True
    
    # Override simulation parameters
    if args.max_rounds is not None:
        if not 1 <= args.max_rounds <= 50:
            raise ValueError("max_rounds must be between 1 and 50")
        config.simulation.max_rounds = args.max_rounds
    
    if args.rationality is not None:
        if not 0.0 <= args.rationality <= 1.0:
            raise ValueError("rationality must be between 0.0 and 1.0")
        config.simulation.rationality = args.rationality
    
    if args.temperature is not None:
        if not 0.0 <= args.temperature <= 2.0:
            raise ValueError("temperature must be between 0.0 and 2.0")
        config.simulation.temperature = args.temperature
        # Also update LLM temperature if not explicitly set differently
        config.models.llm.temperature = args.temperature
    
    if args.discussion_size is not None:
        if not 2 <= args.discussion_size <= 20:
            raise ValueError("discussion_size must be between 2 and 20")
        config.simulation.discussion_group_size = args.discussion_size
    
    # Override groups configuration
    if args.group is not None:
        if config.groups is None:
            config.groups = GroupsConfig()
        config.groups.active = args.group
    
    # Override output/logging parameters
    if args.log_level is not None:
        config.output.logging.log_level = args.log_level
    
    if args.no_viz:
        config.output.visualization.auto_generate = False
    
    return config


def main():
    # Parse command line arguments
    args = parse_arguments()
    
    # print("discussion_round.py: main() started")  # DEBUG PRINT
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    
    # Use custom output directory if specified
    if args.output_dir:
        base_output_dir = Path(args.output_dir)
    else:
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
        config_manager = get_config(str(project_root))  # Explicitly pass project root path
        
        # Apply CLI overrides to configuration
        original_config = config_manager.config
        try:
            updated_config = apply_cli_overrides(original_config, args)
            # Create new config manager with updated config
            from config.scripts.adapter import ConfigAdapter
            config_manager = ConfigAdapter(config_dir=str(project_root), config_object=updated_config)
        except ValueError as e:
            print(f"Error: {e}")
            sys.exit(1)
        # print("discussion_round.py: Config manager obtained.")  # DEBUG PRINT
        
        # Copy configs to output directory (this saves the config WITH CLI overrides)
        config_manager.copy_configs_to_output_dir(base_output_dir)
        # print("discussion_round.py: Configs copied to output dir.")  # DEBUG PRINT
        
        # Create a new config manager that loads from the output directory
        # This ensures the environment uses the exact same config as saved
        output_config_manager = config_manager.create_config_manager_from_output_dir(base_output_dir)
        output_config_manager.set_as_global_config()  # Set as global singleton

        # Setup logging using config adapter
        # print("discussion_round.py: Attempting to initialize logging...")  # DEBUG PRINT
        output_config_manager.initialize_logging(dynamic_logs_dir=logs_dir)
        # print("discussion_round.py: Logging initialized.")  # DEBUG PRINT

        # Setup dedicated stance logger
        stance_logger = setup_stance_logger(logs_dir)

        # Load configuration
        config = output_config_manager  # Use the output config manager
        num_cardinals = config.get_num_cardinals()
        discussion_group_size = config.get_discussion_group_size()  # New relevant config
        max_election_rounds = config.get_max_election_rounds()

        logger.info(
            f"Starting simulation with num_cardinals={num_cardinals}, "
            f"discussion_group_size={discussion_group_size}, "  # Updated log
            f"max_election_rounds={max_election_rounds}"
        )

        # Create the environment (now uses global config automatically)
        env = ConclaveEnv()  # No need to pass config manager anymore
        # print("discussion_round.py: ConclaveEnv initialized.")  # DEBUG PRINT

        # Generate initial internal stances for all agents before starting
        env.generate_initial_stances()

        election_round = 0
        winner_found = False
        vote_counts = {}

        while election_round < max_election_rounds and not winner_found:
            election_round += 1
            logger.info(f"\n=== ELECTION ROUND {election_round} ===")
            print(f"\nStarting Election Round {election_round}")

            # Set the voting round in the environment to match the election round
            env.votingRound = election_round

            # REMOVED: env.reset_discussion_speakers_for_new_election_round()

            # Run a full discussion phase (which now handles groups internally,
            # including analysis and reflection at the end of env.run_discussion_round())
            logger.info(f"\n--- Discussion, Analysis & Reflection Phase (Voting Round {election_round}) ---")
            # print(
            #     f"Attempting to run discussion round (Election Round {election_round})"
            # )
            env.run_discussion_round()  # This internally calls analyze_discussion_round and agents_reflect_on_discussions
            # print(
            #     f"Discussion round completed (Election Round {election_round})"
            # )

            # Explicitly update stances after discussion and reflection
            logger.info(f"\n--- Stance Update Phase (Voting Round {election_round}) ---")
            env.update_internal_stances()

            # Run a voting round
            logger.info(f"\n--- Voting Phase (Voting Round {election_round}) ---")
            # print(f"Attempting to run voting round (Election Round {election_round})")
            winner_id, vote_counts = env.run_voting_round()  # FIXED: Properly unpack the tuple
            winner_found = winner_id is not None  # FIXED: Determine winner status from winner_id
            # print(f"Voting round completed (Election Round {election_round})")

            print(
                f"\n📊 Election Round {election_round} completed. Winner found: {'Yes' if winner_found else 'No'}\n"
            )

        if winner_found:
            print(f"✅ CONCLAVE COMPLETE!")
            if env.winner is not None:  # ADDED CHECK for env.winner
                print(
                    f"🎉 Winner: Cardinal {env.agents[env.winner].name} elected as Pope after {election_round} election rounds"
                )
            else:
                print("Error: Winner was reported, but the winner's ID was not set in the environment.")
                logger.error("winner_found is True, but env.winner is None. This indicates an issue in run_voting_round setting self.winner.")
        else:
            print(f"❌ CONCLAVE INCOMPLETE")
            print(
                f"No winner found after {max_election_rounds} election rounds. Simulation ended without papal election."
            )

        # Save comprehensive simulation results
        try:
            env.save_simulation_results(base_output_dir, timestamp)
            logger.info("All simulation results saved successfully")
        except Exception as e:
            logger.error(f"Error saving comprehensive simulation results: {e}")
            # Fallback to basic results saving
            try:
                results_data = {
                    "timestamp": timestamp,
                    "winner_cardinal_id": env.winner,
                    "winner_name": env.agents[env.winner].name if env.winner is not None else "N/A",
                    "total_election_rounds": election_round,
                    "max_election_rounds_config": max_election_rounds,
                    "discussion_group_size_config": discussion_group_size,
                }
                results_file_path = results_dir / "simulation_summary.json"
                with open(results_file_path, "w") as f:
                    import json
                    json.dump(results_data, f, indent=4)
                logger.info(f"Basic simulation summary saved to {results_file_path}")
            except Exception as fallback_error:
                logger.error(f"Error saving even basic simulation summary: {fallback_error}")

        # Generate enhanced visualizations
        if not args.no_viz:
            try:
                logger.info("Generating final visualizations...")
                visualizer = CardinalVisualizer(str(viz_dir))
                # Pass the final individual votes to the visualizer
                final_individual_votes = env.individual_votes_history[-1] if env.individual_votes_history else {}
                visualizer.generate_all_visualizations(env, vote_counts, final_individual_votes)
                logger.info("All visualizations generated successfully.")
            except Exception as e:
                logger.error(f"Failed to generate visualizations: {e}", exc_info=True)
        else:
            logger.info("Visualization generation skipped due to --no-viz flag.")

    finally:
        # Restore the original working directory
        os.chdir(original_cwd)


if __name__ == "__main__":
    main()
    #print("discussion_round.py: Script starting (__name__ == '__main__')