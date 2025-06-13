#!/usr/bin/env python3
"""
Test Discussion Prompt Generator

This script generates and prints a sample discussion prompt to test the prompt system.
"""

import sys
import os

# Add the project root to Python path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from conclave.environments.conclave_env import ConclaveEnv
from conclave.config.prompts import get_prompt_manager
from conclave.config.prompt_variables import PromptVariableGenerator

def test_discussion_prompt():
    """Generate and print a test discussion prompt."""
    
    print("🧪 Testing Discussion Prompt Generation")
    print("=" * 60)
    
    # Create environment and load agents
    print("Loading environment and agents...")
    env = ConclaveEnv()
    env.load_agents_from_config()
    
    # Generate initial stances for more realistic prompts
    print("Generating initial stances...")
    env.generate_initial_stances()
    
    # Simulate some discussion history for more realistic context
    print("Simulating discussion context...")
    env.discussionRound = 1
    env.votingRound = 1
    
    # Add some mock discussion history
    mock_discussion = [
        {
            'agent_id': 0,
            'message': 'Distinguished brothers, we must consider the global challenges facing our Church today, particularly the need for unity between tradition and pastoral care.'
        },
        {
            'agent_id': 1, 
            'message': 'Your Eminence Arinze raises important points. However, I believe we must also prioritize the voice of the peripheries and ensure our next Pope can bridge cultural divides.'
        }
    ]
    env.discussionHistory.append(mock_discussion)
    
    # Get prompt manager
    prompt_manager = get_prompt_manager()
    
    # Test both candidate and elector discussion prompts
    test_agents = [
        (0, "ELECTOR"),  # First agent as elector
        (1, "CANDIDATE") if env.testing_groups_enabled and 1 in env.candidate_ids else (1, "ELECTOR")  # Second agent
    ]
    
    for agent_id, role in test_agents:
        agent = env.agents[agent_id]
        agent.role_tag = role  # Set role for testing
        
        print(f"\n{'='*60}")
        print(f"🎭 {role} DISCUSSION PROMPT")
        print(f"Agent: {agent.name} (Cardinal {getattr(agent, 'cardinal_id', agent_id)})")
        print(f"{'='*60}")
        
        try:
            # Generate discussion prompt
            discussion_prompt = prompt_manager.get_discussion_prompt(agent_id=agent_id)
            
            print(discussion_prompt)
            
        except Exception as e:
            print(f"❌ Error generating discussion prompt: {e}")
            import traceback
            traceback.print_exc()
    
    # Also test variable generation separately
    print(f"\n{'='*60}")
    print("🔧 TESTING VARIABLE GENERATION")
    print(f"{'='*60}")
    
    var_gen = PromptVariableGenerator(env)
    
    # Test agent variables for first agent
    agent_id = 0
    try:
        variables = var_gen.generate_agent_variables(agent_id)
        
        print(f"\nAgent Variables for {env.agents[agent_id].name}:")
        print("-" * 40)
        
        for key, value in variables.items():
            if isinstance(value, str) and len(value) > 100:
                # Truncate long values for readability
                truncated = value[:100] + "..." if len(value) > 100 else value
                print(f"{key}: {truncated}")
            else:
                print(f"{key}: {value}")
                
    except Exception as e:
        print(f"❌ Error generating variables: {e}")
        import traceback
        traceback.print_exc()
    
    # Test group variables if we have enough agents
    if len(env.agents) >= 5:
        print(f"\n{'='*40}")
        print("👥 GROUP VARIABLES TEST")
        print(f"{'='*40}")
        
        try:
            group_participants = list(range(min(5, len(env.agents))))
            group_vars = var_gen.generate_group_variables(group_participants)
            
            for key, value in group_vars.items():
                print(f"\n{key}:")
                print(value)
                
        except Exception as e:
            print(f"❌ Error generating group variables: {e}")
    
    print(f"\n{'='*60}")
    print("✅ Test completed!")
    print(f"{'='*60}")

if __name__ == "__main__":
    test_discussion_prompt()
