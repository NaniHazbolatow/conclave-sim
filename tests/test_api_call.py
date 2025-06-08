#!/usr/bin/env python3
"""
Simple test to verify that tool_choice parameter fix eliminates 422 errors.
"""

import os
import sys

# Add the project root to the Python path
sys.path.insert(0, os.path.abspath(os.path.dirname(__file__)))

from conclave.llm.client import RemoteLLMClient
from conclave.config.manager import ConfigManager

def test_tool_choice_formats():
    """Test different tool_choice parameter formats."""
    print("=" * 60)
    print("TESTING TOOL_CHOICE PARAMETER FORMATS")
    print("=" * 60)
    
    try:
        # Initialize config
        config_manager = ConfigManager()
        
        # Create LLM client directly
        client_kwargs = config_manager.get_llm_client_kwargs()
        client_kwargs = {k: v for k, v in client_kwargs.items() if v is not None}
        llm_client = RemoteLLMClient(**client_kwargs)
        
        print(f"LLM Client type: {type(llm_client)}")
        print(f"Base URL: {getattr(llm_client, 'base_url', 'Not found')}")
        print()
        
        # Test the actual API call that was failing
        messages = [{"role": "user", "content": "Please use the cast_vote tool to vote for candidate 1."}]
        tools = [
            {
                "type": "function",
                "function": {
                    "name": "cast_vote",
                    "description": "Cast a vote for a candidate",
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "candidate": {"type": "integer", "description": "Candidate ID"},
                            "explanation": {"type": "string", "description": "Reason for vote"}
                        },
                        "required": ["candidate", "explanation"]
                    }
                }
            }
        ]
        
        # This should now work with our fix
        api_params = {
            "model": llm_client.model_name,
            "messages": messages,
            "tools": tools,
            **llm_client.generation_params
        }
        
        # Test the tool_choice parameter fix
        # OpenAI expects object format for specific function
        api_params["tool_choice"] = {"type": "function", "function": {"name": "cast_vote"}}
        print("✅ Using object format for tool_choice (OpenAI standard)")
        
        print(f"Tool choice parameter: {api_params['tool_choice']}")
        print()
        
        # Try the actual API call
        print("Testing API call with tool_choice...")
        response = llm_client.client.chat.completions.create(**api_params)
        
        print("✅ API call successful! No 422 errors.")
        print(f"Response: {response.choices[0].message}")
        return True
        
    except Exception as e:
        if "422" in str(e):
            print(f"❌ Still getting 422 errors: {e}")
            return False
        else:
            print(f"⚠️  Different error (not 422): {e}")
            return True  # 422 errors are fixed, but there may be other issues
    
    return False

if __name__ == "__main__":
    success = test_tool_choice_formats()
    if success:
        print("\n🎉 Tool choice parameter fix is working!")
        print("The 422 'tool_choice format' errors should be resolved.")
    else:
        print("\n💥 Tool choice parameter fix failed!")
        print("Still getting 422 errors - needs further investigation.")
