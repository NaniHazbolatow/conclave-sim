#!/usr/bin/env python3
"""
Enhanced test script for local LLM prompting with HuggingFace, Ollama, and Remote support.

This script comprehensively tests all LLM backends for the conclave simulation,
including model downloading, swapping, and performance comparison.
"""

import asyncio
import json
import logging
import os
import sys
import time
from typing import Dict, List, Optional, Tuple

# Add the project root to the path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from conclave.llm.client import (
    HuggingFaceClient, 
    LocalLLMClient, 
    RemoteLLMClient, 
    UnifiedLLMClient,
    create_llm_client,
    HF_AVAILABLE
)

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def test_huggingface_availability():
    """Test if HuggingFace is available and can load models."""
    print("\n" + "="*60)
    print("TESTING HUGGINGFACE AVAILABILITY")
    print("="*60)
    
    if not HF_AVAILABLE:
        print("❌ HuggingFace transformers not available")
        print("   Install with: pip install torch transformers accelerate")
        return False
    
    print("✅ HuggingFace transformers available")
    
    try:
        # Try to create a client with a small model
        print("🔄 Testing with small model (DialoGPT-small)...")
        client = HuggingFaceClient(
            model_name="microsoft/DialoGPT-small",
            use_quantization=False  # Disable for testing
        )
        
        if client.is_available():
            print("✅ HuggingFace client created successfully")
            model_info = client.get_model_info()
            print(f"📊 Model info: {json.dumps(model_info, indent=2)}")
            return True
        else:
            print("❌ HuggingFace client creation failed")
            return False
            
    except Exception as e:
        print(f"❌ HuggingFace test failed: {e}")
        return False


def test_model_swapping():
    """Test swapping between different HuggingFace models."""
    print("\n" + "="*60)
    print("TESTING MODEL SWAPPING")
    print("="*60)
    
    if not HF_AVAILABLE:
        print("❌ Skipping model swapping test - HuggingFace not available")
        return False
    
    try:
        client = HuggingFaceClient(model_name="distilgpt2", use_quantization=False)
        print(f"✅ Initial model loaded: {client.model_name}")
        
        # Test swapping to different models
        test_models = ["gpt2", "microsoft/DialoGPT-small"]
        successful_swaps = 0
        
        for model in test_models:
            print(f"\n🔄 Swapping to model: {model}")
            start_time = time.time()
            success = client.load_model(model)
            end_time = time.time()
            
            if success:
                print(f"✅ Successfully swapped to {model} in {end_time - start_time:.2f}s")
                successful_swaps += 1
            else:
                print(f"❌ Failed to swap to {model}")
        
        print(f"\n📊 Successfully swapped {successful_swaps}/{len(test_models)} models")
        return successful_swaps > 0
        
    except Exception as e:
        print(f"❌ Model swapping test failed: {e}")
        return False


def test_basic_prompting_all_backends():
    """Test basic prompting across all available backends."""
    print("\n" + "="*60)
    print("TESTING BASIC PROMPTING - ALL BACKENDS")
    print("="*60)
    
    backends_to_test = []
    
    # Test HuggingFace
    if HF_AVAILABLE:
        try:
            hf_client = create_llm_client(backend="huggingface", model_name="distilgpt2", use_quantization=False)
            backends_to_test.append(("HuggingFace", hf_client))
        except Exception as e:
            print(f"⚠️ Could not initialize HuggingFace: {e}")
    
    # Test Ollama
    try:
        ollama_client = create_llm_client(backend="ollama")
        if ollama_client.is_available():
            backends_to_test.append(("Ollama", ollama_client))
    except Exception as e:
        print(f"⚠️ Could not initialize Ollama: {e}")
    
    # Test Remote
    if os.getenv("OPENROUTER_API_KEY"):
        try:
            remote_client = create_llm_client(backend="remote")
            backends_to_test.append(("Remote", remote_client))
        except Exception as e:
            print(f"⚠️ Could not initialize Remote: {e}")
    
    if not backends_to_test:
        print("❌ No backends available for testing")
        return False
    
    # Test each backend
    prompt = [{"role": "user", "content": "Hello! Please respond with exactly 'Backend working' if you understand."}]
    results = []
    
    for backend_name, client in backends_to_test:
        print(f"\n🧪 Testing {backend_name}...")
        start_time = time.time()
        
        try:
            response = client.prompt(prompt, max_new_tokens=50, temperature=0.3)
            end_time = time.time()
            
            print(f"🤖 Response: {response[:100]}...")
            print(f"⏱️  Time: {end_time - start_time:.2f}s")
            
            # Check if response is reasonable
            success = len(response) > 5 and not response.startswith("Error:")
            results.append((backend_name, success, end_time - start_time))
            
            if success:
                print(f"✅ {backend_name} prompting successful")
            else:
                print(f"❌ {backend_name} prompting failed")
                
        except Exception as e:
            print(f"❌ {backend_name} error: {e}")
            results.append((backend_name, False, 0))
    
    # Summary
    print(f"\n📊 Backend Test Results:")
    for backend, success, response_time in results:
        status = "✅" if success else "❌"
        print(f"   {status} {backend}: {'PASS' if success else 'FAIL'} ({response_time:.2f}s)")
    
    return any(success for _, success, _ in results)


def test_cardinal_simulation_enhanced():
    """Test enhanced cardinal simulation with different scenarios."""
    print("\n" + "="*60)
    print("TESTING ENHANCED CARDINAL SIMULATION")
    print("="*60)
    
    try:
        client = create_llm_client(backend="auto")
        print(f"✅ Using backend: {client.get_model_info().get('active_backend', 'unknown')}")
        
        scenarios = [
            {
                "name": "Conservative Cardinal",
                "system": "You are Cardinal Francesco, a traditional conservative cardinal from Italy, aged 70. You value church doctrine and tradition above all.",
                "prompt": "The conclave is considering Cardinal Santos, a progressive from Brazil who supports more inclusive policies. How do you feel about his candidacy?"
            },
            {
                "name": "Progressive Cardinal", 
                "system": "You are Cardinal Maria, a progressive cardinal from Germany, aged 55. You support church modernization and social justice.",
                "prompt": "Cardinal O'Malley from Boston is gaining support among traditional cardinals. What are your thoughts on his potential papacy?"
            },
            {
                "name": "Moderate Cardinal",
                "system": "You are Cardinal Chen, a moderate cardinal from Hong Kong, aged 62. You seek balance between tradition and progress.",
                "prompt": "The conclave is deadlocked between progressive and conservative factions. How might you help find consensus?"
            }
        ]
        
        successful_scenarios = 0
        
        for scenario in scenarios:
            print(f"\n--- {scenario['name']} ---")
            messages = [
                {"role": "system", "content": scenario["system"]},
                {"role": "user", "content": scenario["prompt"]}
            ]
            
            start_time = time.time()
            response = client.prompt(messages, temperature=0.8, max_new_tokens=200)
            end_time = time.time()
            
            print(f"🤖 Response: {response[:200]}...")
            print(f"⏱️  Time: {end_time - start_time:.2f}s")
            
            # Check if response seems reasonable for a cardinal
            cardinal_keywords = ['church', 'pope', 'cardinal', 'faith', 'catholic', 'conclave', 'doctrine']
            if any(keyword in response.lower() for keyword in cardinal_keywords) and len(response) > 50:
                print(f"✅ {scenario['name']} simulation successful")
                successful_scenarios += 1
            else:
                print(f"❌ {scenario['name']} simulation inadequate")
        
        print(f"\n📊 Successful scenarios: {successful_scenarios}/{len(scenarios)}")
        return successful_scenarios >= len(scenarios) // 2
        
    except Exception as e:
        print(f"❌ Cardinal simulation failed: {e}")
        return False


def test_structured_voting_output():
    """Test structured JSON output for voting decisions."""
    print("\n" + "="*60)
    print("TESTING STRUCTURED VOTING OUTPUT")
    print("="*60)
    
    try:
        client = create_llm_client(backend="auto")
        
        voting_prompt = [
            {
                "role": "system", 
                "content": """You are a cardinal voting in a papal conclave. You must respond with valid JSON only.
                
                Format:
                {
                    "vote": "cardinal_name",
                    "reasoning": "brief explanation",
                    "confidence": 8,
                    "ballot_round": 1
                }"""
            },
            {
                "role": "user",
                "content": """Ballot Round 3 Results:
                
1. Cardinal Rodriguez (Mexico) - 42 votes - Progressive, social justice advocate
2. Cardinal Benedetti (Italy) - 38 votes - Conservative, traditional theology  
3. Cardinal Kim (South Korea) - 25 votes - Moderate, interfaith dialogue

Who do you vote for in Round 4? Respond with JSON only."""
            }
        ]
        
        print("🗳️ Testing structured voting output...")
        response = client.prompt(voting_prompt, temperature=0.3, max_new_tokens=150)
        
        print(f"🤖 Raw response: {response}")
        
        # Try to extract and parse JSON
        json_response = extract_json_from_response(response)
        
        if json_response:
            print(f"✅ Successfully parsed JSON:")
            print(json.dumps(json_response, indent=2))
            
            # Validate required fields
            required_fields = ['vote', 'reasoning', 'confidence']
            if all(field in json_response for field in required_fields):
                print("✅ All required fields present")
                return True
            else:
                print(f"❌ Missing required fields. Got: {list(json_response.keys())}")
                return False
        else:
            print("❌ Could not parse JSON from response")
            return False
            
    except Exception as e:
        print(f"❌ Structured voting test failed: {e}")
        return False


def extract_json_from_response(response: str) -> Optional[Dict]:
    """Extract JSON from model response, handling various formats."""
    try:
        # Try direct parsing first
        return json.loads(response.strip())
    except:
        pass
    
    # Try to find JSON in markdown code blocks
    import re
    json_patterns = [
        r'```json\s*(\{.*?\})\s*```',
        r'```\s*(\{.*?\})\s*```',
        r'(\{[^}]*"vote"[^}]*\})',  # Look for JSON with "vote" field
    ]
    
    for pattern in json_patterns:
        matches = re.findall(pattern, response, re.DOTALL)
        if matches:
            try:
                return json.loads(matches[0])
            except:
                continue
    
    return None


def test_unified_client_auto_selection():
    """Test the unified client's automatic backend selection."""
    print("\n" + "="*60)
    print("TESTING UNIFIED CLIENT AUTO-SELECTION")
    print("="*60)
    
    try:
        # Test auto-selection
        client = create_llm_client(backend="auto")
        model_info = client.get_model_info()
        
        print(f"✅ Auto-selected backend: {model_info.get('active_backend', 'unknown')}")
        print(f"📊 Model info: {json.dumps(model_info, indent=2)}")
        
        # Test basic functionality
        response = client.prompt([
            {"role": "user", "content": "What backend are you using? Respond briefly."}
        ], max_new_tokens=50)
        
        print(f"🤖 Response: {response}")
        
        if len(response) > 10 and not response.startswith("Error:"):
            print("✅ Unified client working correctly")
            return True
        else:
            print("❌ Unified client response inadequate")
            return False
            
    except Exception as e:
        print(f"❌ Unified client test failed: {e}")
        return False


def test_performance_comparison():
    """Compare performance across available backends."""
    print("\n" + "="*60)
    print("TESTING PERFORMANCE COMPARISON")
    print("="*60)
    
    test_prompt = [{"role": "user", "content": "Explain the role of a papal conclave in one sentence."}]
    backends = []
    
    # Collect available backends
    if HF_AVAILABLE:
        try:
            hf_client = create_llm_client(backend="huggingface", model_name="distilgpt2", use_quantization=False)
            backends.append(("HuggingFace", hf_client))
        except:
            pass
    
    try:
        ollama_client = create_llm_client(backend="ollama")
        if ollama_client.is_available():
            backends.append(("Ollama", ollama_client))
    except:
        pass
    
    if os.getenv("OPENROUTER_API_KEY"):
        try:
            remote_client = create_llm_client(backend="remote")
            backends.append(("Remote", remote_client))
        except:
            pass
    
    if not backends:
        print("❌ No backends available for performance testing")
        return False
    
    print(f"🏁 Testing {len(backends)} backends...")
    results = []
    
    for backend_name, client in backends:
        print(f"\n📊 Testing {backend_name} performance...")
        
        times = []
        for i in range(3):  # Run 3 times for average
            start_time = time.time()
            response = client.prompt(test_prompt, max_new_tokens=100, temperature=0.3)
            end_time = time.time()
            
            if not response.startswith("Error:"):
                times.append(end_time - start_time)
                print(f"   Run {i+1}: {end_time - start_time:.2f}s")
            else:
                print(f"   Run {i+1}: FAILED")
        
        if times:
            avg_time = sum(times) / len(times)
            results.append((backend_name, avg_time, len(times)))
            print(f"✅ {backend_name} average: {avg_time:.2f}s ({len(times)} successful runs)")
        else:
            print(f"❌ {backend_name} failed all runs")
    
    # Summary
    if results:
        print(f"\n🏆 Performance Summary:")
        sorted_results = sorted(results, key=lambda x: x[1])
        for i, (backend, avg_time, runs) in enumerate(sorted_results, 1):
            print(f"   {i}. {backend}: {avg_time:.2f}s avg ({runs}/3 runs)")
        return True
    else:
        print("❌ No successful performance tests")
        return False


def test_memory_usage():
    """Test memory usage with HuggingFace models."""
    print("\n" + "="*60)
    print("TESTING MEMORY USAGE")
    print("="*60)
    
    if not HF_AVAILABLE:
        print("❌ Skipping memory test - HuggingFace not available")
        return False
    
    try:
        import psutil
        import torch
        
        process = psutil.Process()
        
        print("📊 Memory usage before model loading:")
        print(f"   RAM: {process.memory_info().rss / 1024**2:.1f} MB")
        if torch.cuda.is_available():
            print(f"   GPU: {torch.cuda.memory_allocated() / 1024**2:.1f} MB")
        
        # Load a small model
        client = HuggingFaceClient(model_name="distilgpt2", use_quantization=False)
        
        print("\n📊 Memory usage after model loading:")
        print(f"   RAM: {process.memory_info().rss / 1024**2:.1f} MB")
        if torch.cuda.is_available():
            print(f"   GPU: {torch.cuda.memory_allocated() / 1024**2:.1f} MB")
        
        # Test a prompt
        response = client.prompt([{"role": "user", "content": "Hello!"}], max_new_tokens=50)
        
        print("\n📊 Memory usage after inference:")
        print(f"   RAM: {process.memory_info().rss / 1024**2:.1f} MB")
        if torch.cuda.is_available():
            print(f"   GPU: {torch.cuda.memory_allocated() / 1024**2:.1f} MB")
        
        print("✅ Memory usage test completed")
        return True
        
    except ImportError:
        print("⚠️ psutil not available for memory monitoring")
        return True
    except Exception as e:
        print(f"❌ Memory test failed: {e}")
        return False


def run_all_tests():
    """Run all tests and provide a comprehensive summary."""
    print("🧪 CONCLAVE SIMULATION - ENHANCED LLM TESTING")
    print("=" * 70)
    print("Testing HuggingFace, Ollama, and Remote LLM backends")
    print("=" * 70)
    
    tests = [
        ("HuggingFace Availability", test_huggingface_availability),
        ("Model Swapping", test_model_swapping),
        ("Basic Prompting (All Backends)", test_basic_prompting_all_backends),
        ("Enhanced Cardinal Simulation", test_cardinal_simulation_enhanced),
        ("Structured Voting Output", test_structured_voting_output),
        ("Unified Client Auto-Selection", test_unified_client_auto_selection),
        ("Performance Comparison", test_performance_comparison),
        ("Memory Usage", test_memory_usage),
    ]
    
    results = []
    
    for test_name, test_func in tests:
        print(f"\n🔄 Running: {test_name}")
        try:
            result = test_func()
            results.append((test_name, result))
        except Exception as e:
            logger.error(f"Test '{test_name}' failed with exception: {e}")
            results.append((test_name, False))
    
    # Comprehensive Summary
    print("\n" + "="*70)
    print("COMPREHENSIVE TEST SUMMARY")
    print("="*70)
    
    passed = sum(1 for _, result in results if result)
    total = len(results)
    
    print("📋 Individual Test Results:")
    for test_name, result in results:
        status = "✅ PASS" if result else "❌ FAIL"
        print(f"   {status} {test_name}")
    
    print(f"\n📊 Overall Score: {passed}/{total} tests passed ({passed/total*100:.1f}%)")
    
    # Recommendations
    print("\n💡 Recommendations:")
    if passed == total:
        print("   🎉 Excellent! All systems are working perfectly.")
        print("   🚀 Your setup is ready for the conclave simulation.")
    elif passed >= total * 0.75:
        print("   ✅ Good! Most systems are working well.")
        print("   🔧 Address any failed tests for optimal performance.")
    elif passed >= total * 0.5:
        print("   ⚠️  Partial functionality detected.")
        print("   🛠️  Consider installing missing dependencies or fixing configuration.")
    else:
        print("   ❌ Multiple systems need attention.")
        print("   📖 Check the LOCAL_LLM_SETUP.md guide for help.")
    
    # Installation tips
    print("\n🛠️  Quick Setup Tips:")
    print("   • HuggingFace: pip install torch transformers accelerate")
    print("   • Ollama: curl -fsSL https://ollama.ai/install.sh | sh")
    print("   • Remote: export OPENROUTER_API_KEY='your-key'")
    
    return passed >= total * 0.75


if __name__ == "__main__":
    success = run_all_tests()
    sys.exit(0 if success else 1)