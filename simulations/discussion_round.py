from conclave.environments.conclave_env import ConclaveEnv
from conclave.agents.base import Agent
from conclave.config.manager import get_config
import pandas as pd
import logging
import datetime
import os

timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")

# Ensure the logs directory exists
os.makedirs('logs', exist_ok=True)

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler(f"logs/discussion_round_{timestamp}.log"),
        # logging.StreamHandler()
    ]
)

# Create a logger for your module
logger = logging.getLogger(__name__)

def main():
    # Load configuration
    config = get_config()
    
    # Create the environment
    env = ConclaveEnv()

    # Read cardinals from CSV file
    cardinals_df = pd.read_csv('data/cardinal_electors_2025.csv')

    # Create Agent instances and add them to env.agents (using config for number of cardinals)
    num_cardinals = config.get_num_cardinals()
    for idx, row in cardinals_df.head(num_cardinals).iterrows():
        agent = Agent(
            agent_id=idx,
            name=row['Name'],
            background=row['Background'],
            env=env
        )
        env.agents.append(agent)

    # Set the number of agents in the environment
    env.num_agents = len(env.agents)
    logger.info(f"\n{env.list_candidates_for_prompt(randomize=False)}")

    # Get simulation parameters from config
    max_speakers_per_round = config.get_max_speakers_per_round()
    num_discussions = config.get_num_discussions()
    max_election_rounds = config.get_max_election_rounds()
    
    logger.info(f"Starting simulation with {num_cardinals} cardinals")
    logger.info(f"Configuration: {max_speakers_per_round} speakers per round, {num_discussions} discussions, max {max_election_rounds} election rounds")

    election_round = 0
    winner_found = False
    
    while election_round < max_election_rounds and not winner_found:
        election_round += 1
        logger.info(f"\n=== ELECTION ROUND {election_round} ===")
        
        # Run the configured number of discussions
        for discussion_num in range(1, num_discussions + 1):
            logger.info(f"\n--- Discussion Cycle {discussion_num} ---")
            # Run a discussion round with configured number of speakers
            # Set random=True to select cardinals randomly instead of by urgency
            env.run_discussion_round(num_speakers=max_speakers_per_round, random_selection=True)
        
        # After all discussions, run voting
        winner_found = env.run_voting_round()
        print(f"Election round {election_round} completed. Winner found: {winner_found}")

    if winner_found:
        print(f"Winner found after {election_round} election rounds: Cardinal {env.winner} - {env.agents[env.winner].name}")
    else:
        print(f"No winner found after {max_election_rounds} election rounds. Simulation ended without papal election.")


if __name__ == "__main__":
    main()
