from environments.conclave_env import ConclaveEnv
from agents.base import Agent
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
        logging.FileHandler(f"logs/single_round_run_{timestamp}.log"),
        # logging.StreamHandler()
    ]
)

# Create a logger for your module
logger = logging.getLogger(__name__)

def main():
    # Create the environment
    env = ConclaveEnv()

    # Read cardinals from CSV file
    cardinals_df = pd.read_csv('cardinal_electors_2025.csv')

    # Create Agent instances and add them to env.agents
    counter = 0
    for idx, row in cardinals_df.iterrows():
        counter += 1
        if counter > 20:
            break
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
    env.run_voting_round()


if __name__ == "__main__":
    main()
