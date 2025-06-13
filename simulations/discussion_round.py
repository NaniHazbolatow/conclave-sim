import sys
import os
from pathlib import Path
import warnings

# Suppress urllib3 LibreSSL warning on macOS
warnings.filterwarnings("ignore", message="urllib3 v2 only supports OpenSSL 1.1.1+")

# Add the parent directory to the Python path so we can import conclave
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from conclave.environments.conclave_env import ConclaveEnv
from conclave.agents.base import Agent
from conclave.config.manager import get_config
from conclave.visualization.cardinal_visualizer import CardinalVisualizer
import pandas as pd
import logging
import datetime

timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")

# Ensure the logs directory exists (relative to project root)
logs_dir = project_root / 'logs'
logs_dir.mkdir(exist_ok=True)

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler(logs_dir / f"discussion_round_{timestamp}.log"),
        # logging.StreamHandler()
    ]
)

# Create a logger for your module
logger = logging.getLogger(__name__)

def main():
    # Change to the project root directory so config.yaml can be found
    original_cwd = os.getcwd()
    os.chdir(project_root)
    
    try:
        # Load configuration
        config = get_config()
        num_cardinals = config.get_num_cardinals() # Still relevant for overall setup
        # REMOVED: max_speakers_per_round = config.get_max_speakers_per_round()
        # REMOVED: num_discussions = config.get_num_discussions()
        discussion_group_size = config.get_discussion_group_size() # New relevant config
        max_election_rounds = config.get_max_election_rounds()
        
        logger.info(f"Starting simulation with num_cardinals={num_cardinals}, "
                   f"discussion_group_size={discussion_group_size}, " # Updated log
                   f"max_election_rounds={max_election_rounds}")
        
        # Create the environment
        env = ConclaveEnv()

        # Load agents based on testing groups configuration
        env.load_agents_from_config()

        # Generate initial internal stances for all agents before starting
        env.generate_initial_stances()

        election_round = 0
        winner_found = False
        
        while election_round < max_election_rounds and not winner_found:
            election_round += 1
            logger.info(f"\n=== ELECTION ROUND {election_round} ===")
            
            # REMOVED: env.reset_discussion_speakers_for_new_election_round()
            
            # Run a full discussion phase (which now handles groups internally)
            # The number of discussion *cycles* (previously num_discussions) is now implicitly 1 full pass of all groups.
            # If you need multiple full discussion phases before a vote, this loop structure would need adjustment.
            # For now, assuming one full discussion phase (all groups speak once) per election round.
            logger.info(f"\n--- Discussion Phase (Voting Round {election_round}) ---")
            env.run_discussion_round() # No num_speakers argument needed
            # Stance updates are now handled within run_discussion_round after all groups have spoken.
            # REMOVED: env.update_internal_stances() 
            
            # After all discussions, run voting
            winner_found = env.run_voting_round()
            print(f"\n📊 Election Round {election_round} completed. Winner found: {'Yes' if winner_found else 'No'}\n")

        if winner_found:
            print(f"✅ CONCLAVE COMPLETE!")
            print(f"🎉 Winner: Cardinal {env.agents[env.winner].name} elected as Pope after {election_round} election rounds")
        else:
            print(f"❌ CONCLAVE INCOMPLETE")
            print(f"No winner found after {max_election_rounds} election rounds. Simulation ended without papal election.")
        
        # Generate stance progression visualization using CardinalVisualizer
        visualizer = CardinalVisualizer(config)
        visualizer.generate_stance_visualization_from_env(env)
    
    finally:
        # Restore the original working directory
        os.chdir(original_cwd)


if __name__ == "__main__":
    main()
