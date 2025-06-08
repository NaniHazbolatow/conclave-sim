from conclave.environments.conclave_env import ConclaveEnv
from conclave.agents.base import Agent
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
    # Create the environment
    env = ConclaveEnv()

    # Read cardinals from CSV file
    cardinals_df = pd.read_csv('data/cardinal_electors_2025.csv')

    # Create Agent instances and add them to env.agents
    counter = 0
    for idx, row in cardinals_df.head(5).iterrows():
        # counter += 1
        # if counter > 50:
        #     break
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

    winner_found = False
    while not winner_found:
        # Run a discussion round with 5 speakers
        # Set random=True to select cardinals randomly instead of by urgency
        env.run_discussion_round(num_speakers=5, random_selection=True)
        winner_found = env.run_voting_round()
        print(f"winner_found: {winner_found}")

    print(f"Winner found: Cardinal {env.winner} - {env.agents[env.winner].name}")


if __name__ == "__main__":
    main()
