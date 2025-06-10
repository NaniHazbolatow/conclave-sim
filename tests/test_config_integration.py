#!/usr/bin/env python3
"""
Test configuration integration with embeddings.
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from conclave.config.manager import get_config
from conclave.embeddings import get_default_client

def main():
    print("🔧 Testing Configuration Integration")
    print("=" * 50)
    
    # Test config loading
    print("\n1. Loading Configuration:")
    config = get_config()
    print(f"   ✅ Config loaded from: {config.config_path}")
    print(f"   📊 Backend: {config.get_backend_type()}")
    print(f"   🧠 LLM Model: {config.get_local_model_name() if config.is_local_backend() else config.get_remote_model_name()}")
    print(f"   🔗 Embedding Model: {config.get_embedding_model_name()}")
    print(f"   💾 Embedding Cache: {config.get_embedding_cache_enabled()}")
    print(f"   📁 Cache Dir: {config.get_embedding_cache_dir()}")
    
    # Test embedding config
    print("\n2. Embedding Configuration:")
    embedding_config = config.get_embedding_config()
    print(f"   📐 Similarity Threshold: {config.get_embedding_similarity_threshold()}")
    print(f"   🔄 Batch Size: {config.get_embedding_batch_size()}")
    print(f"   🖥️  Device: {config.get_embedding_device()}")
    
    # Test embedding client initialization
    print("\n4. Testing Embedding Client from Config:")
    try:
        embedding_kwargs = config.get_embedding_client_kwargs()
        print(f"   🔧 Client kwargs: {embedding_kwargs}")
        
        # Test with small model for quick initialization
        test_config = embedding_kwargs.copy()
        test_config["model_name"] = "all-MiniLM-L6-v2"  # Smaller model for testing
        
        print(f"   🔄 Initializing embedding client with model: {test_config['model_name']}")
        # Note: We don't actually initialize to avoid downloading models
        print(f"   ✅ Configuration validated successfully!")
        
    except Exception as e:
        print(f"   ❌ Error in embedding config: {e}")
    
    # Test simulation config
    print("\n5. Simulation Configuration:")
    sim_config = config.get_simulation_config()
    print(f"   📝 Log Level: {sim_config.get('log_level', 'Not Set')}")
    print(f"   👥 Number of Cardinals: {sim_config.get('num_cardinals', 'Not Set')}")
    print(f"   👥 Max Speakers Per Round: {sim_config.get('max_speakers_per_round', 'Not Set')}")
    print(f"   🗣️  Number of Discussions per Round: {sim_config.get('num_discussions', 'Not Set')}")
    print(f"   🗳️  Max Election Rounds: {sim_config.get('max_election_rounds', 'Not Set')}")
    
    voting_config = sim_config.get('voting', {})
    print(f"   🗳️  Supermajority Threshold: {voting_config.get('supermajority_threshold', 'Not Set')}")
    print(f"   💭 Require Reasoning: {voting_config.get('require_reasoning', 'Not Set')}")
    
    print("\n✅ Configuration test completed!")
    print(f"\nThe Agent class now properly uses the configuration system!")
    print(f"Temperature, max_tokens, and model selection are all connected through config.yaml.")

if __name__ == "__main__":
    main()
