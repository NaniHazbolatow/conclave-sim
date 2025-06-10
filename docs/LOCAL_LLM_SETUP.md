# Local LLM Setup Guide

This guide will help you set up local LLM inference for the ConclaveSim project using HuggingFace Transformers and/or Ollama.

## What You Need to Download

### Option 1: HuggingFace Transformers (Recommended)
Direct Python integration with thousands of models from Hugging Face.

**Installation:**
```bash
# Core requirements
pip install torch transformers accelerate

# For 4-bit quantization (saves memory):
pip install bitsandbytes

# Alternative: Install with uv
uv add torch transformers accelerate bitsandbytes
```

**Popular Models to Try:**
```bash
# These will be downloaded automatically when you run the code
# Small models (2-4GB RAM):
- microsoft/DialoGPT-small
- gpt2
- distilgpt2

# Medium models (8-16GB RAM):
- microsoft/DialoGPT-medium
- gpt2-medium

# Large models (16GB+ RAM):
- meta-llama/Llama-3.2-1B-Instruct
- meta-llama/Llama-3.2-3B-Instruct
- microsoft/DialoGPT-large

# Very large models (32GB+ RAM):
- meta-llama/Llama-3.1-8B-Instruct
- meta-llama/Llama-3.1-70B-Instruct
- mistralai/Mistral-7B-Instruct-v0.3
```

### Option 2: Ollama (Alternative)
Pre-optimized local LLM server.

**Installation:**
```bash
# Option 1: Download from website
# Go to https://ollama.ai and download the installer for macOS

# Option 2: Using Homebrew
brew install ollama
```

**Download Models:**
```bash
# Start Ollama service (if not auto-started)
ollama serve

# In a new terminal, download models:
ollama pull llama3.1:8b        # 8GB RAM
ollama pull llama3.1:13b       # 16GB RAM  
ollama pull mistral:7b         # 8GB RAM
ollama pull codellama:13b      # 16GB RAM
```

## Hardware Requirements

### HuggingFace Models
| Model Size | RAM Required | VRAM (GPU) | Performance | Quality |
|------------|-------------|------------|-------------|---------|
| Small (117M-355M) | 2-4GB | 1-2GB | Very Fast | Basic |
| Medium (345M-774M) | 4-8GB | 2-4GB | Fast | Good |
| Large (1.5B-3B) | 8-16GB | 4-8GB | Medium | Better |
| XL (7B-8B) | 16-32GB | 8-16GB | Slow | Excellent |

### Device Support
- **CPU**: Works on any modern CPU (slower)
- **Apple Silicon (M1/M2/M3)**: Excellent performance with MPS
- **NVIDIA GPU**: Best performance with CUDA
- **AMD GPU**: Limited support

## Testing Your Setup

1. **Install dependencies:**
   ```bash
   uv sync
   # For HuggingFace support:
   uv add torch transformers accelerate bitsandbytes
   ```

2. **Run the comprehensive test:**
   ```bash
   uv run python testlocalprompting.py
   ```

The test script will:
- Check HuggingFace availability  
- Test model loading and swapping
- Test basic prompting
- Test structured responses (tool calling simulation)
- Test Ollama (if available)
- Test unified client with auto-selection

## Expected Output

### HuggingFace Success:
```
🤗 Testing HuggingFace Transformers Availability
✅ HuggingFace transformers are available

🔄 Testing model loading: microsoft/DialoGPT-small
✅ Model loaded successfully in 15.3s
📊 Model info:
   model_name: microsoft/DialoGPT-small
   device: mps
   parameters: 117000000
   memory_footprint: 0.47 GB (approx)

==================================================
2. Testing basic prompting with HuggingFace...
🤖 Sending prompt...
✅ Response received in 2.45s
📝 Response: As Cardinal Giuseppe Versaldi, I believe the next Pope should possess...
```

### Auto-Selection:
```
🚀 Testing Local LLM Integration Suite

==================================================
4. Testing Unified Client (Auto-selection)...
✅ Using HuggingFace: microsoft/DialoGPT-medium
✅ Unified client initialized
📊 Active backend: huggingface
📊 Model: microsoft/DialoGPT-medium
✅ Unified client working correctly
```

## Integration with Your Codebase

The updated `llm_client.py` now provides:

1. **HuggingFaceClient**: Direct Hugging Face integration with quantization
2. **LocalLLMClient**: Ollama integration (unchanged)
3. **RemoteLLMClient**: OpenRouter integration (unchanged)
4. **UnifiedLLMClient**: Automatically chooses the best available backend

### Quick Integration Example:

```python
from conclave.llm.client import UnifiedLLMClient

class Agent:
    def __init__(self, agent_id: int, name: str, background: str, env: ConclaveEnv):
        # ... existing code ...
        
        # Replace OpenAI client with unified client
        self.llm_client = UnifiedLLMClient(
            backend="auto",  # or "huggingface", "ollama", "remote"
            model_name="microsoft/DialoGPT-medium"  # HF model name
        )
    
    def cast_vote(self) -> None:
        # ... existing prompt setup ...
        
        # Replace OpenAI API call:
        response = self.llm_client.chat_completion(
            messages=[{"role": "user", "content": prompt}],
            tools=tools,
            tool_choice="cast_vote",
            temperature=0.5,
            max_tokens=1000
        )
        
        # Rest of the code remains the same!
```

## Model Recommendations

### For Development/Testing:
- **HuggingFace**: `microsoft/DialoGPT-small` or `gpt2`
- **Ollama**: `llama3.1:8b`

### For Production:
- **HuggingFace**: `microsoft/DialoGPT-medium` or `meta-llama/Llama-3.2-3B-Instruct`
- **Ollama**: `llama3.1:13b`

### For Best Quality:
- **HuggingFace**: `meta-llama/Llama-3.1-8B-Instruct`
- **Ollama**: `llama3.1:70b`

## Advantages of HuggingFace vs Ollama

### HuggingFace Advantages:
✅ **Model Variety**: Access to 100,000+ models  
✅ **Control**: Full control over inference parameters  
✅ **Quantization**: 4-bit/8-bit models for efficiency  
✅ **Custom Models**: Use fine-tuned or custom models  
✅ **Offline**: No external dependencies  
✅ **Research**: Latest models available immediately  

### Ollama Advantages:
✅ **Simplicity**: Easy installation and management  
✅ **Optimization**: Pre-optimized for inference  
✅ **Stability**: Reliable and well-tested  
✅ **Memory**: Better memory management  

## Troubleshooting

### "ModuleNotFoundError: No module named 'torch'"
```bash
pip install torch transformers accelerate
# Or for Apple Silicon:
pip install torch --index-url https://download.pytorch.org/whl/cpu
```

### "CUDA out of memory" / "MPS out of memory"
- Use a smaller model
- Enable quantization: `use_quantization=True`
- Set `device="cpu"` to use CPU instead

### Model downloads are slow
- Models are cached in `~/.cache/huggingface/`
- Large models (7B+) can be 15GB+ downloads
- Consider using smaller models for testing

### JSON parsing errors with structured responses
- Local models need careful prompting for structured output
- Try lower temperature (0.1-0.3)
- Use larger models for better instruction following
- Check the test script for working examples

## Performance Comparison

| Aspect | HuggingFace | Ollama | Remote (OpenRouter) |
|--------|-------------|--------|-------------------|
| **Setup Time** | 5-30 min | 5 min | 1 min |
| **First Response** | 30-120s | 5-15s | 1-3s |
| **Subsequent** | 2-10s | 2-8s | 1-3s |
| **Cost** | Free | Free | $0.001-0.01/request |
| **Privacy** | Complete | Complete | Shared with provider |
| **Model Choice** | 100,000+ | ~50 | ~100 |
| **Customization** | Full control | Limited | None |

## Next Steps

1. **Choose your backend** based on your priorities
2. **Run the test script** to verify everything works
3. **Update base.py** to use `UnifiedLLMClient`
4. **Test with a subset** of cardinals first
5. **Monitor performance** and adjust model/parameters as needed

## Memory Usage Tips

- **Enable quantization** for large models: `use_quantization=True`
- **Use smaller models** for development: `microsoft/DialoGPT-small`
- **Monitor RAM usage** with Activity Monitor
- **Close other applications** when running large models
- **Consider CPU-only** if you have limited VRAM: `device="cpu"`

## Expected Output

When working correctly, you should see:
```
🚀 Testing Local LLM Integration with Ollama

1. Checking Ollama status...
✅ Ollama is running. Available models: ['llama3.1:8b']

==================================================
2. Testing basic prompting...
🤖 Sending prompt to llama3.1:8b...
✅ Response received in 3.45s
📝 Response: As Cardinal Giuseppe Versaldi, I believe the next Pope should possess...

==================================================
3. Testing structured response (simulating vote casting)...
🛠️ Testing structured response with llama3.1:8b...
✅ Structured response received in 4.12s
✅ Successfully parsed JSON response
📊 Structured Response:
{
  "reasoning": "Given my moderate stance and focus on education...",
  "action": "cast_vote",
  "parameters": {
    "candidate": "2"
  }
}
```

## Integration with Your Codebase

The `llm_client.py` module provides:

1. **UnifiedLLMClient**: Automatically chooses between local and remote LLMs
2. **LocalLLMClient**: Direct interface to Ollama
3. **RemoteLLMClient**: Interface to OpenRouter (your current setup)

To integrate into your `base.py`, you would replace the OpenAI client with:

```python
from llm_client import UnifiedLLMClient

class Agent:
    def __init__(self, agent_id: int, name: str, background: str, env: ConclaveEnv):
        # ... existing code ...
        
        # Replace OpenAI client with unified client
        self.llm_client = UnifiedLLMClient(
            prefer_local=True,  # Set to False to use remote only
            local_model="llama3.1:8b"  # Your chosen model
        )
```

## Troubleshooting

### "Cannot connect to Ollama"
- Make sure Ollama is installed: `ollama --version`
- Start the service: `ollama serve`
- Check if it's running: `curl http://localhost:11434/api/tags`

### "Model not found"
- List available models: `ollama list`
- Download the model: `ollama pull llama3.1:8b`

### Slow responses
- Try a smaller model (8b instead of 13b)
- Ensure sufficient RAM is available
- Close other memory-intensive applications

### JSON parsing errors
- Local models sometimes need fine-tuning of prompts
- Try adjusting temperature (lower = more structured)
- Consider using a larger model for better instruction following

## Performance Comparison

| Aspect | Local LLM | Remote LLM |
|--------|-----------|------------|
| **Speed** | 2-10s per request | 1-3s per request |
| **Cost** | Free after setup | Pay per token |
| **Privacy** | Complete privacy | Data sent to provider |
| **Quality** | Depends on model size | Generally high |
| **Reliability** | Depends on hardware | Depends on internet |
| **Setup** | Requires local setup | Requires API key |

## Next Steps

1. Run the test script to verify everything works
2. Choose your preferred model based on hardware
3. Modify `base.py` to use the unified client
4. Test with a small subset of cardinals first
5. Adjust temperature and other parameters as needed

## Models Recommendations

- **Development/Testing**: `llama3.1:8b` or `mistral:7b`
- **Production**: `llama3.1:13b` (if you have 16GB+ RAM)
- **Best Quality**: `llama3.1:70b` (requires powerful hardware)
