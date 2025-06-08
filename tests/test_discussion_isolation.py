#!/usr/bin/env python3
"""
Test script to verify that agents only see discussions they participated in.
"""

from conclave.environments.conclave_env import ConclaveEnv
from conclave.agents.base import Agent
import pandas as pd
import logging

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def test_discussion_isolation():
    """Test that agents only see discussions they participated in."""
    
    # Create a small environment with just 3 agents for testing
    env = ConclaveEnv(num_agents=3)
    
    # Create 3 test agents
    test_agents = [
        {"name": "Cardinal A", "background": "Conservative traditionalist focused on doctrine"},
        {"name": "Cardinal B", "background": "Progressive reformer focused on social justice"},
        {"name": "Cardinal C", "background": "Moderate diplomat focused on unity"}
    ]
    
    for i, agent_data in enumerate(test_agents):
        agent = Agent(
            agent_id=i,
            name=agent_data["name"],
            background=agent_data["background"],
            env=env
        )
        env.agents.append(agent)
    
    env.num_agents = len(env.agents)
    
    print("=== Testing Discussion Isolation ===")
    print(f"Created {env.num_agents} agents: {[agent.name for agent in env.agents]}")
    
    # Simulate manual discussion to test isolation
    # Round 1: Only agents 0 and 1 participate
    print("\n--- Round 1: Agents 0 and 1 participate ---")
    env.discussionRound = 1
    round1_comments = [
        {"agent_id": 0, "message": "I believe we need strong traditional leadership."},
        {"agent_id": 1, "message": "The Church must embrace modern social justice."}
    ]
    env.discussionHistory.append(round1_comments)
    
    # Track participation manually for this test
    for comment in round1_comments:
        agent_id = comment['agent_id']
        if agent_id not in env.agent_discussion_participation:
            env.agent_discussion_participation[agent_id] = []
        env.agent_discussion_participation[agent_id].append(0)  # Round index 0
    
    print("Participation tracking:", env.agent_discussion_participation)
    
    # Test what each agent sees
    for agent_id in range(3):
        history = env.get_discussion_history(agent_id)
        print(f"\nAgent {agent_id} ({env.agents[agent_id].name}) sees:")
        if history:
            print(history)
        else:
            print("No discussion history (agent didn't participate)")
    
    # Round 2: Only agents 1 and 2 participate
    print("\n--- Round 2: Agents 1 and 2 participate ---")
    env.discussionRound = 2
    round2_comments = [
        {"agent_id": 1, "message": "We must focus on helping the poor and marginalized."},
        {"agent_id": 2, "message": "Both tradition and progress have their place."}
    ]
    env.discussionHistory.append(round2_comments)
    
    # Track participation
    for comment in round2_comments:
        agent_id = comment['agent_id']
        if agent_id not in env.agent_discussion_participation:
            env.agent_discussion_participation[agent_id] = []
        env.agent_discussion_participation[agent_id].append(1)  # Round index 1
    
    print("Participation tracking:", env.agent_discussion_participation)
    
    # Test what each agent sees after round 2
    for agent_id in range(3):
        history = env.get_discussion_history(agent_id)
        print(f"\nAgent {agent_id} ({env.agents[agent_id].name}) sees:")
        if history:
            print(history)
        else:
            print("No discussion history (agent didn't participate)")
    
    # Test that get_discussion_history() without agent_id still returns all discussions
    print("\n--- Full discussion history (no agent_id filter) ---")
    full_history = env.get_discussion_history()
    print(full_history)
    
    print("\n=== Test Complete ===")
    print("Expected behavior:")
    print("- Agent 0 should only see Round 1")
    print("- Agent 1 should see both Round 1 and Round 2") 
    print("- Agent 2 should only see Round 2")
    print("- Full history should show all rounds")

if __name__ == "__main__":
    test_discussion_isolation()
