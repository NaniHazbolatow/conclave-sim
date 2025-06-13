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
        logging.FileHandler(f"logs/single_round_run_{timestamp}.log"),
        # logging.StreamHandler()
    ]
)

# Create a logger for your module
logger = logging.getLogger(__name__)

def main():
    # Create the environment
    env = ConclaveEnv()

    # Load agents based on testing groups configuration
    env.load_agents_from_config()
    
    # Generate initial internal stances for all agents before starting
    env.generate_initial_stances()
    
    env.run_voting_round()


if __name__ == "__main__":
    main()
