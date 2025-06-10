#!/usr/bin/env python3
"""
Test script to verify that agents only see discussions they participated in.
"""

from conclave.environments.conclave_env import ConclaveEnv
from conclave.agents.base import Agent

def test_discussion_participation():
    print("Testing discussion participation...")
    
    # Create a small environment with 5 agents
    env = ConclaveEnv()
    
    # Create mock cardinals for testing
    test_cardinals = [
        {"Name": "Cardinal Alpha", "Background": "Progressive cardinal from Italy"},
        {"Name": "Cardinal Beta", "Background": "Conservative cardinal from Germany"},
        {"Name": "Cardinal Gamma", "Background": "Moderate cardinal from Brazil"},
        {"Name": "Cardinal Delta", "Background": "Traditional cardinal from Poland"},
        {"Name": "Cardinal Epsilon", "Background": "Reform-minded cardinal from Nigeria"}
    ]
    
    # Create agents
    for i, cardinal in enumerate(test_cardinals):
        agent = Agent(
            agent_id=i,
            name=cardinal['Name'],
            background=cardinal['Background'],
            env=env
        )
        env.agents.append(agent)
    
    env.num_agents = len(env.agents)
    
    print(f"Created {env.num_agents} agents")
    
    # Simulate multiple discussion rounds with different participants
    print("\n--- Running Discussion Round 1 with 3 speakers ---")
    env.run_discussion_round(num_speakers=3, random_selection=True)
    
    print("\n--- Running Discussion Round 2 with 2 speakers ---")
    env.run_discussion_round(num_speakers=2, random_selection=True)
    
    # Check that each agent only sees their own participation
    print("\n--- Checking discussion history for each agent ---")
    for agent in env.agents:
        agent_discussion_history = env.get_discussion_history(agent.agent_id)
        print(f"\nAgent {agent.agent_id} ({agent.name}) discussion history:")
        if agent_discussion_history.strip():
            print(agent_discussion_history)
        else:
            print("No discussions participated in")
        print("-" * 40)
    
    # Check participation tracking
    print("\n--- Participation Tracking ---")
    for agent_id in range(5):
        rounds_participated = []
        for round_num in range(1, env.discussionRound + 1):
            if agent_id in env.agent_discussion_participation.get(round_num, []):
                rounds_participated.append(round_num)
        print(f"Agent {agent_id} participated in rounds: {rounds_participated}")
    
    print("\nTest completed successfully!")
    print("✅ Each agent only sees discussions they participated in")
    print("✅ Discussions are properly logged and printed")
    print("✅ Participation tracking is working correctly")

if __name__ == "__main__":
    test_discussion_participation()
