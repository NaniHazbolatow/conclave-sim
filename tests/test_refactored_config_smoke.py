import sys
import os

# Add project root to Python path to allow imports from config module
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
sys.path.insert(0, project_root)

from config.scripts.loader import load_config
from config.scripts.adapter import get_config

def test_smoke_refactored_config():
    print("Starting refactored configuration smoke test...")
    config = load_config()
    print("Refactored configuration loaded successfully!")
    print(f"LLM Backend: {config.agent.llm.backend}")
    print(f"Number of Cardinals: {config.simulation.num_cardinals}")
    print(f"Active Testing Group: {config.testing_groups.active_group if config.testing_groups else 'None'}")

    # Test adapter
    adapter = get_config()
    print("\nConfigAdapter created successfully!")
    print(f"Adapter - Num Cardinals: {adapter.get_num_cardinals()}")

    print("\nSmoke test completed successfully!")

if __name__ == "__main__":
    test_smoke_refactored_config()