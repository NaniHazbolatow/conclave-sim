#!/usr/bin/env python3
"""
Test script to verify the three fixes:
1. Persona information in voting prompts
2. Coalition proposals restricted to candidates
3. Momentum tags only shown when applicable (already fixed)
"""

import sys
import os

# Add the current directory to Python path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from conclave.environments.conclave_env import ConclaveEnv
from conclave.config.manager import ConfigManager
from conclave.config.prompts import get_prompt_manager
import logging

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def test_persona_in_voting_prompts():
    """Test that persona_internal is properly populated in voting prompts."""
    print("=" * 60)
    print("TEST 1: Persona Information in Voting Prompts")
    print("=" * 60)
    
    # Create environment with testing groups
    config = ConfigManager()
    env = ConclaveEnv()
    env.load_agents_from_config()
    
    # Get the first agent and generate internal personas
    agent = env.agents[0]
    agent.generate_internal_stance()  # This should populate persona_internal
    
    # Get voting prompt
    prompt_manager = get_prompt_manager()
    voting_prompt = prompt_manager.get_voting_prompt(agent_id=0)
    
    print(f"Agent: {agent.name}")
    print(f"Persona Internal: {getattr(agent, 'persona_internal', 'NOT SET')}")
    print("\nVoting Prompt Preview:")
    print("-" * 40)
    print(voting_prompt[:500] + "..." if len(voting_prompt) > 500 else voting_prompt)
    
    # Check if persona is in the prompt
    has_persona = "{persona_internal}" not in voting_prompt and getattr(agent, 'persona_internal', '')
    print(f"\n✅ Persona properly populated: {has_persona}")
    
    return has_persona

def test_candidate_restriction():
    """Test that coalition proposals are restricted to actual candidates."""
    print("\n" + "=" * 60)
    print("TEST 2: Coalition Proposals Restricted to Candidates")
    print("=" * 60)
    
    # Test with testing groups enabled
    config = ConfigManager()
    env = ConclaveEnv()
    env.load_agents_from_config()
    
    # Get visible candidates
    from conclave.config.prompt_variables import PromptVariableGenerator
    var_gen = PromptVariableGenerator(env)
    
    visible_candidates = var_gen.get_visible_candidates_ids()
    all_agents = list(range(len(env.agents)))
    actual_candidates = env.get_candidates_list()
    
    print(f"Testing Groups Enabled: {env.testing_groups_enabled}")
    print(f"Total Agents: {len(env.agents)}")
    print(f"All Agent IDs: {all_agents}")
    print(f"Actual Candidates: {actual_candidates}")
    print(f"Visible Candidates (for voting): {visible_candidates}")
    
    # Check if visible candidates is properly restricted
    is_restricted = set(visible_candidates).issubset(set(actual_candidates))
    print(f"\n✅ Coalition restricted to candidates: {is_restricted}")
    
    # Test candidate info generation
    visible_candidates_info = var_gen.generate_visible_candidates()
    print(f"\nVisible Candidates Info:")
    print("-" * 40)
    print(visible_candidates_info)
    
    return is_restricted

def test_momentum_tags():
    """Test that momentum tags only appear when there's previous round data."""
    print("\n" + "=" * 60)
    print("TEST 3: Momentum Tags Only With Previous Data")
    print("=" * 60)
    
    # Create environment
    config = ConfigManager()
    env = ConclaveEnv()
    env.load_agents_from_config()
    
    # Test with no voting history (round 1)
    from conclave.config.prompt_variables import PromptVariableGenerator
    var_gen = PromptVariableGenerator(env)
    
    scoreboard_round1 = var_gen.generate_compact_scoreboard()
    print(f"Round 1 Scoreboard (no previous data): {scoreboard_round1}")
    
    # Check if momentum tags are absent
    has_momentum_tags = any(tag in scoreboard_round1 for tag in ["gaining", "losing", "stable", "dropped"])
    print(f"✅ No momentum tags in first round: {not has_momentum_tags}")
    
    # Simulate some voting history
    env.votingHistory = [
        {0: 2, 1: 1, 2: 1},  # Previous round
        {0: 3, 1: 1, 2: 0}   # Current round
    ]
    
    scoreboard_round2 = var_gen.generate_compact_scoreboard()
    print(f"Round 2 Scoreboard (with previous data): {scoreboard_round2}")
    
    # Check if momentum tags are present
    has_momentum_tags_round2 = any(tag in scoreboard_round2 for tag in ["gaining", "losing", "stable", "dropped"])
    print(f"✅ Momentum tags present in subsequent rounds: {has_momentum_tags_round2}")
    
    return not has_momentum_tags and has_momentum_tags_round2

def main():
    """Run all tests."""
    print("Testing Three Fixes for Conclave Simulation")
    print("=" * 60)
    
    try:
        test1_passed = test_persona_in_voting_prompts()
        test2_passed = test_candidate_restriction()
        test3_passed = test_momentum_tags()
        
        print("\n" + "=" * 60)
        print("SUMMARY")
        print("=" * 60)
        print(f"✅ Test 1 (Persona in Voting): {'PASSED' if test1_passed else 'FAILED'}")
        print(f"✅ Test 2 (Candidate Restriction): {'PASSED' if test2_passed else 'FAILED'}")
        print(f"✅ Test 3 (Momentum Tags): {'PASSED' if test3_passed else 'FAILED'}")
        
        all_passed = test1_passed and test2_passed and test3_passed
        print(f"\n🎉 All tests {'PASSED' if all_passed else 'FAILED'}")
        
        return all_passed
        
    except Exception as e:
        print(f"❌ Error during testing: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
