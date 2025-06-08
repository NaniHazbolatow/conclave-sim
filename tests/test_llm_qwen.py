#!/usr/bin/env python3
"""
Test script to diagnose LLM behavior with Qwen model configuration.
This will help understand what happens when we use the Qwen model ID with OpenRouter.
"""

import os
import sys
import asyncio
from typing import List

# Add the project root to the Python path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from conclave.llm.client import get_llm_client
from conclave.config.manager import ConfigManager

async def test_basic_llm_call():
    """Test a basic LLM call to see if we get any response."""
    print("=" * 60)
    print("TESTING BASIC LLM CALL WITH QWEN MODEL")
    print("=" * 60)
    
    # Initialize config
    config_manager = ConfigManager()
    config = config_manager.get_config()
    
    print(f"Current backend: {config['llm']['backend']}")
    print(f"Current model: {config['llm']['model_name']}")
    print(f"API Key present: {'OPENROUTER_API_KEY' in os.environ}")
    print()
    
    try:
        # Get LLM client
        llm_client = get_llm_client(config)
        print(f"LLM Client type: {type(llm_client)}")
        
        # Test simple text generation
        print("\n--- Testing simple text generation ---")
        messages = [
            {"role": "user", "content": "Hello! Can you respond with just the word 'SUCCESS' to confirm you're working?"}
        ]
        
        response = await llm_client.generate_response(messages)
        print(f"Response: {response}")
        
        if response:
            print("✅ Basic text generation WORKS")
        else:
            print("❌ Basic text generation FAILED - no response")
            
    except Exception as e:
        print(f"❌ Basic text generation FAILED with error: {e}")
        print(f"Error type: {type(e)}")
        return False
    
    return True

async def test_tool_calling():
    """Test LLM's ability to use tools (function calling)."""
    print("\n" + "=" * 60)
    print("TESTING TOOL CALLING CAPABILITY")
    print("=" * 60)
    
    config_manager = ConfigManager()
    config = config_manager.get_config()
    
    try:
        llm_client = get_llm_client(config)
        
        # Define a simple tool
        tools = [
            {
                "type": "function",
                "function": {
                    "name": "cast_vote",
                    "description": "Cast a vote for a candidate",
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "candidate": {
                                "type": "string",
                                "description": "The name of the candidate to vote for"
                            },
                            "reasoning": {
                                "type": "string", 
                                "description": "Explanation for the vote choice"
                            }
                        },
                        "required": ["candidate", "reasoning"]
                    }
                }
            }
        ]
        
        messages = [
            {
                "role": "user", 
                "content": "You must vote for one of these candidates: Alice, Bob, Charlie. Please use the cast_vote tool to vote for Alice with the reasoning 'Test vote'."
            }
        ]
        
        print("Testing tool calling...")
        response = await llm_client.generate_response(messages, tools=tools)
        print(f"Tool calling response: {response}")
        
        if response and "cast_vote" in str(response):
            print("✅ Tool calling might be working")
        else:
            print("❌ Tool calling appears to NOT be working")
            
    except Exception as e:
        print(f"❌ Tool calling test FAILED with error: {e}")
        print(f"Error type: {type(e)}")
        return False
    
    return True

async def test_openrouter_models():
    """Test what models are actually available on OpenRouter."""
    print("\n" + "=" * 60)
    print("TESTING OPENROUTER MODEL AVAILABILITY")
    print("=" * 60)
    
    try:
        import aiohttp
        
        api_key = os.getenv('OPENROUTER_API_KEY')
        if not api_key:
            print("❌ No OpenRouter API key found")
            return False
            
        headers = {
            'Authorization': f'Bearer {api_key}',
            'Content-Type': 'application/json'
        }
        
        async with aiohttp.ClientSession() as session:
            async with session.get('https://openrouter.ai/api/v1/models', headers=headers) as response:
                if response.status == 200:
                    data = await response.json()
                    models = data.get('data', [])
                    
                    print(f"Found {len(models)} available models")
                    
                    # Look for Qwen models
                    qwen_models = [m for m in models if 'qwen' in m.get('id', '').lower()]
                    print(f"\nFound {len(qwen_models)} Qwen models:")
                    for model in qwen_models[:10]:  # Show first 10
                        print(f"  - {model.get('id')}")
                    
                    # Check if our specific model exists
                    target_model = "Qwen/Qwen2.5-3B-Instruct"
                    model_exists = any(m.get('id') == target_model for m in models)
                    
                    if model_exists:
                        print(f"\n✅ Model '{target_model}' IS available on OpenRouter")
                    else:
                        print(f"\n❌ Model '{target_model}' is NOT available on OpenRouter")
                        print("Similar models found:")
                        similar = [m.get('id') for m in models if 'qwen' in m.get('id', '').lower() and '2.5' in m.get('id', '')]
                        for sim in similar[:5]:
                            print(f"  - {sim}")
                    
                else:
                    print(f"❌ Failed to fetch models. Status: {response.status}")
                    
    except Exception as e:
        print(f"❌ Model availability test FAILED: {e}")
        return False
    
    return True

async def main():
    """Run all diagnostic tests."""
    print("🔍 DIAGNOSING LLM ISSUES WITH QWEN MODEL")
    print("This will help understand why votes aren't being cast.")
    print()
    
    # Test 1: Basic connectivity and response
    basic_success = await test_basic_llm_call()
    
    # Test 2: Tool calling capability
    tool_success = await test_tool_calling()
    
    # Test 3: Check available models
    model_success = await test_openrouter_models()
    
    print("\n" + "=" * 60)
    print("DIAGNOSTIC SUMMARY")
    print("=" * 60)
    print(f"Basic LLM call: {'✅ PASS' if basic_success else '❌ FAIL'}")
    print(f"Tool calling: {'✅ PASS' if tool_success else '❌ FAIL'}")
    print(f"Model availability: {'✅ PASS' if model_success else '❌ FAIL'}")
    
    if not any([basic_success, tool_success, model_success]):
        print("\n🚨 CRITICAL: Multiple failures detected!")
        print("Recommendation: Check API key and model configuration")
    elif basic_success and not tool_success:
        print("\n⚠️  WARNING: Basic calls work but tool calling fails!")
        print("This explains why votes aren't being cast.")
    else:
        print("\n✅ Most tests passed - issue might be elsewhere")

if __name__ == "__main__":
    asyncio.run(main())
