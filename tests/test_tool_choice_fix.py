#!/usr/bin/env python3
"""
Test script to verify the tool_choice parameter fix works with OpenRouter/DeepInfra.
"""

import os
import sys
import asyncio

# Add the project root to the Python path
sys.path.insert(0, os.path.abspath(os.path.dirname(__file__)))

from conclave.agents.base import Agent
from conclave.environments.conclave_env import ConclaveEnv
from conclave.config.manager import ConfigManager

def test_tool_choice_fix():
    """Test that the tool_choice parameter format fix works."""
    print("=" * 60)
    print("TESTING TOOL_CHOICE PARAMETER FIX")
    print("=" * 60)
    
    try:
        # Initialize config
        config_manager = ConfigManager()
        config = config_manager.get_config()
        
        print(f"Backend: {config['llm']['backend']}")
        print(f"Model: {config['llm']['model_name']}")
        print(f"Base URL: {config['llm']['remote']['base_url']}")
        print()
        
        # Create a minimal environment and agent
        env = ConclaveEnv()
        agent = Agent(
            agent_id=1,
            name="Test Cardinal",
            background="Test cardinal for debugging voting.",
            env=env
        )
        
        # Check what the LLM client looks like
        print(f"LLM Client type: {type(agent.llm_client)}")
        print(f"LLM Client base_url: {getattr(agent.llm_client, 'base_url', 'Not found')}")
        print()
        
        # Test tool calling directly
        print("Testing tool calling with voting...")
        try:
            # This should trigger the fixed tool_choice parameter handling
            agent.cast_vote()
            print("✅ Vote casting completed without tool_choice errors!")
            
            # Check if vote was recorded
            if agent.vote_history:
                last_vote = agent.vote_history[-1]
                print(f"✅ Vote recorded: {last_vote}")
            else:
                print("⚠️  No vote recorded - may need further debugging")
                
        except Exception as e:
            print(f"❌ Vote casting failed: {e}")
            return False
            
    except Exception as e:
        print(f"❌ Test setup failed: {e}")
        return False
    
    return True

if __name__ == "__main__":
    success = test_tool_choice_fix()
    if success:
        print("\n🎉 Tool choice fix test completed!")
    else:
        print("\n💥 Tool choice fix test failed!")
