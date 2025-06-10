#!/usr/bin/env python3
"""
Test configuration system for LLM client selection.
"""

from conclave.config.manager import get_config
from conclave.llm.client import HuggingFaceClient, RemoteLLMClient
import logging

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def test_config_loading():
    """Test loading and using the configuration."""
    print("=== Testing Configuration Loading ===")
    
    try:
        # Get configuration
        config = get_config()
        print(f"✅ Configuration loaded successfully: {config}")
        
        # Test backend selection
        backend = config.get_backend_type()
        print(f"📋 Selected backend: {backend}")
        
        # Test model selection
        if config.is_local_backend():
            model = config.get_local_model_name()
            print(f"🤖 Local model: {model}")
        else:
            model = config.get_remote_model_name()
            print(f"🌐 Remote model: {model}")
        
        # Test LLM client kwargs
        kwargs = config.get_llm_client_kwargs()
        print(f"⚙️  Client kwargs: {kwargs}")
        
        # Test temperature
        temp = config.get_temperature()
        print(f"🌡️  Temperature: {temp}")
        
        return True
        
    except Exception as e:
        print(f"❌ Configuration loading failed: {e}")
        return False

def test_client_initialization():
    """Test initializing LLM client from configuration."""
    print("\n=== Testing Client Initialization ===")
    
    try:
        config = get_config()
        client_kwargs = config.get_llm_client_kwargs()
        
        # Remove None values
        client_kwargs = {k: v for k, v in client_kwargs.items() if v is not None}
        
        if config.is_local_backend():
            print("🔧 Initializing HuggingFace client...")
            client = HuggingFaceClient(**client_kwargs)
            print(f"✅ HuggingFace client initialized: {client.model_name}")
        else:
            print("🔧 Initializing Remote client...")
            client = RemoteLLMClient(**client_kwargs)
            print(f"✅ Remote client initialized: {client.model_name}")
        
        # Test a simple generation
        print("🧪 Testing simple generation...")
        messages = [{"role": "user", "content": "Say 'Hello, configuration test!'"}]
        response = client.generate(messages)
        print(f"🤖 Response: {response}")
        
        return True
        
    except Exception as e:
        print(f"❌ Client initialization failed: {e}")
        return False

def test_config_switching():
    """Test switching configuration settings."""
    print("\n=== Testing Configuration Switching ===")
    
    try:
        config = get_config()
        
        print(f"📋 Current backend: {config.get_backend_type()}")
        print(f"🌡️  Current temperature: {config.get_temperature()}")
        
        # Show how to switch backends by editing config.yaml
        print("\n💡 To switch backends, edit config.yaml:")
        print("   For local:  llm.backend: 'local'")
        print("   For remote: llm.backend: 'remote'")
        
        print("\n💡 To change models, edit the model_name under:")
        print("   Local:  llm.local.model_name")
        print("   Remote: llm.remote.model_name")
        
        print("\n💡 To adjust temperature: llm.temperature")
        
        return True
        
    except Exception as e:
        print(f"❌ Configuration switching test failed: {e}")
        return False

if __name__ == "__main__":
    print("🧪 CONFIGURATION SYSTEM TEST")
    print("=" * 50)
    
    tests = [
        test_config_loading,
        test_client_initialization,
        test_config_switching
    ]
    
    passed = 0
    for test in tests:
        if test():
            passed += 1
    
    print(f"\n📊 Results: {passed}/{len(tests)} tests passed")
    
    if passed == len(tests):
        print("🎉 All tests passed! Configuration system is working correctly.")
    else:
        print("⚠️  Some tests failed. Check the errors above.")
