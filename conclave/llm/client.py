"""
LLM Client with HuggingFace, Ollama, and Remote support for the Conclave Simulation.

This module provides a unified interface for interacting with different LLM backends:
- HuggingFaceClient: Direct integration with HuggingFace transformers
- LocalLLMClient: Ollama integration for local models
- RemoteLLMClient: OpenRouter API for remote models
- UnifiedLLMClient: Automatically selects the best available backend

Key features for HuggingFace integration:
- Model downloading and caching
- Quantization support (4-bit, 8-bit)
- Easy model swapping
- Memory optimization
- Device detection (CPU/CUDA/MPS)
"""

import json
import logging
import os
import platform
import threading
import time
from abc import ABC, abstractmethod
from typing import Any, Dict, List, Optional, Union

# Core dependencies
import requests
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

# HuggingFace dependencies (optional)
try:
    import torch
    from transformers import (
        AutoModelForCausalLM,
        AutoTokenizer,
        BitsAndBytesConfig,
        pipeline
    )
    HF_AVAILABLE = True
except ImportError:
    HF_AVAILABLE = False
    torch = None

logger = logging.getLogger(__name__)


class LLMClientBase(ABC):
    """Abstract base class for LLM clients."""
    
    @abstractmethod
    def prompt(self, messages: List[Dict[str, str]], **kwargs) -> str:
        """Send a prompt to the LLM and return the response."""
        pass
    
    @abstractmethod
    def is_available(self) -> bool:
        """Check if this client is available and functional."""
        pass
    
    @abstractmethod
    def get_model_info(self) -> Dict[str, Any]:
        """Get information about the current model."""
        pass


class HuggingFaceClient(LLMClientBase):
    """HuggingFace transformers client with model swapping and quantization support."""
    
    # Recommended models by size category
    RECOMMENDED_MODELS = {
        "small": [
            "microsoft/phi-2",
            "distilgpt2", 
            "microsoft/DialoGPT-small",
        ],
        "medium": [
            "microsoft/phi-3-mini-4k-instruct",
            "Qwen/Qwen2.5-1.5B-Instruct",
            "gpt2",
            "microsoft/DialoGPT-medium",
        ],
        "large": [
            "gpt2-large",
            "microsoft/phi-3-small-8k-instruct",
            "Qwen/Qwen2.5-3B-Instruct",
            "meta-llama/Llama-3.2-3B-Instruct",
        ],
        "xlarge": [
            "microsoft/phi-3-medium-4k-instruct",
            "Qwen/Qwen2.5-7B-Instruct",
            "meta-llama/Llama-3.2-1B-Instruct",
            "mistralai/Mistral-7B-Instruct-v0.3",
        ]
    }
    
    def __init__(self, 
                 model_name: Optional[str] = None, 
                 use_quantization: bool = True,
                 device: Optional[str] = None,
                 cache_dir: Optional[str] = None,
                 **kwargs):
        """
        Initialize HuggingFace client.
        
        Args:
            model_name: Name of the model to load. If None, will use a default small model.
            use_quantization: Whether to use 4-bit quantization to save memory.
            device: Device to run on ('cpu', 'cuda', 'mps'). Auto-detected if None.
            cache_dir: Directory to cache models. Uses HF default if None.
            **kwargs: Additional generation parameters (temperature, max_tokens, etc.)
        """
        if not HF_AVAILABLE:
            raise RuntimeError("HuggingFace transformers not available. Install with: pip install torch transformers accelerate")
        
        self.use_quantization = use_quantization
        self.cache_dir = cache_dir or os.path.expanduser("~/.cache/huggingface")
        self.device = device or self._detect_device()
        
        # Store generation parameters
        self.generation_params = {
            'temperature': kwargs.get('temperature', 0.8),
            'max_tokens': kwargs.get('max_tokens', 150),
            'do_sample': kwargs.get('do_sample', True),
            'top_p': kwargs.get('top_p', 0.9),
            'repetition_penalty': kwargs.get('repetition_penalty', 1.1),
        }
        
        # Initialize with default model if none specified
        self.model_name = model_name or "gpt2"  # Use gpt2 as a reliable default
        self.model = None
        self.tokenizer = None
        self.pipeline = None
        
        # Thread lock for safe concurrent access
        self._generation_lock = threading.Lock()
        
        # Load the initial model
        self.load_model(self.model_name)
    
    def _detect_device(self) -> str:
        """Detect the best available device."""
        if torch.cuda.is_available():
            return "cuda"
        elif hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
            return "mps"  # Apple Silicon
        else:
            return "cpu"
    
    def _get_quantization_config(self) -> Optional[BitsAndBytesConfig]:
        """Get quantization configuration for memory optimization."""
        if not self.use_quantization or self.device == "cpu":
            return None
        
        try:
            return BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_compute_dtype=torch.float16,
                bnb_4bit_use_double_quant=True,
                bnb_4bit_quant_type="nf4"
            )
        except Exception as e:
            logger.warning(f"Could not create quantization config: {e}")
            return None
    
    def load_model(self, model_name: str) -> bool:
        """
        Load a new model, replacing the current one.
        
        Args:
            model_name: Name of the HuggingFace model to load.
            
        Returns:
            True if model loaded successfully, False otherwise.
        """
        try:
            logger.info(f"Loading HuggingFace model: {model_name}")
            start_time = time.time()
            
            # Clear existing model to free memory
            if self.model is not None:
                del self.model
                del self.tokenizer
                del self.pipeline
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
            
            # Load tokenizer
            self.tokenizer = AutoTokenizer.from_pretrained(
                model_name,
                cache_dir=self.cache_dir,
                trust_remote_code=True
            )
            
            # Add padding token if not present
            if self.tokenizer.pad_token is None:
                self.tokenizer.pad_token = self.tokenizer.eos_token
            
            # Get quantization config
            quantization_config = self._get_quantization_config()
            
            # Load model
            model_kwargs = {
                "cache_dir": self.cache_dir,
                "trust_remote_code": True,
            }
            
            if quantization_config:
                model_kwargs["quantization_config"] = quantization_config
                model_kwargs["device_map"] = "auto"
            else:
                model_kwargs["torch_dtype"] = torch.float16 if self.device != "cpu" else torch.float32
            
            self.model = AutoModelForCausalLM.from_pretrained(
                model_name,
                **model_kwargs
            )
            
            if not quantization_config:
                self.model = self.model.to(self.device)
            
            # Create generation pipeline
            self.pipeline = pipeline(
                "text-generation",
                model=self.model,
                tokenizer=self.tokenizer,
                device=self.device if not quantization_config else None,
                torch_dtype=torch.float16 if self.device != "cpu" else torch.float32,
            )
            
            self.model_name = model_name
            load_time = time.time() - start_time
            logger.info(f"Model {model_name} loaded successfully in {load_time:.2f}s")
            return True
            
        except Exception as e:
            logger.error(f"Failed to load model {model_name}: {e}")
            return False
    
    def prompt(self, messages: List[Dict[str, str]], **kwargs) -> str:
        """
        Send a prompt to the HuggingFace model.
        
        Args:
            messages: List of message dictionaries with 'role' and 'content' keys.
            **kwargs: Additional generation parameters.
            
        Returns:
            Generated response text.
        """
        if self.pipeline is None:
            raise RuntimeError("No model loaded")
        
        # Use thread lock to ensure safe concurrent access
        with self._generation_lock:
            # Convert messages to prompt text
            prompt = self._messages_to_prompt(messages)
            
            # Set default generation parameters from stored config
            generation_kwargs = {
                "max_new_tokens": kwargs.get("max_new_tokens", kwargs.get("max_tokens", self.generation_params.get("max_tokens", 150))),
                "temperature": kwargs.get("temperature", self.generation_params.get("temperature", 0.8)),
                "do_sample": kwargs.get("do_sample", self.generation_params.get("do_sample", True)),
                "top_p": kwargs.get("top_p", self.generation_params.get("top_p", 0.9)),
                "repetition_penalty": kwargs.get("repetition_penalty", self.generation_params.get("repetition_penalty", 1.1)),
                "pad_token_id": self.tokenizer.eos_token_id,
                "return_full_text": False,
                "use_cache": False,  # Disable cache to avoid DynamicCache issues
                "past_key_values": None,  # Explicitly set to None
            }
            generation_kwargs.update(kwargs)
            
            try:
                # Generate response
                result = self.pipeline(prompt, **generation_kwargs)
                
                if isinstance(result, list) and len(result) > 0:
                    response = result[0].get("generated_text", "").strip()
                else:
                    response = str(result).strip()
                
                return response
                
            except Exception as e:
                logger.error(f"Error generating response: {e}")
                return f"Error: {str(e)}"
    
    def generate(self, messages: List[Dict[str, str]], **kwargs) -> str:
        """Generate text using the HuggingFace model (alias for prompt method)."""
        return self.prompt(messages, **kwargs)
    
    def _messages_to_prompt(self, messages: List[Dict[str, str]]) -> str:
        """Convert OpenAI-style messages to a prompt string."""
        prompt_parts = []
        
        for message in messages:
            role = message.get("role", "user")
            content = message.get("content", "")
            
            if role == "system":
                prompt_parts.append(f"System: {content}\n")
            elif role == "user":
                prompt_parts.append(f"Human: {content}\n")
            elif role == "assistant":
                prompt_parts.append(f"Assistant: {content}\n")
        
        # Add assistant prompt
        prompt_parts.append("Assistant: ")
        return "".join(prompt_parts)
    
    def is_available(self) -> bool:
        """Check if HuggingFace client is available."""
        return HF_AVAILABLE and self.pipeline is not None
    
    def get_model_info(self) -> Dict[str, Any]:
        """Get information about the current model."""
        if not self.is_available():
            return {"status": "unavailable"}
        
        info = {
            "backend": "huggingface",
            "model_name": self.model_name,
            "device": self.device,
            "quantization": self.use_quantization,
            "status": "ready"
        }
        
        # Add memory info if available
        if torch.cuda.is_available():
            info["gpu_memory"] = {
                "allocated": f"{torch.cuda.memory_allocated() / 1024**3:.2f}GB",
                "cached": f"{torch.cuda.memory_reserved() / 1024**3:.2f}GB"
            }
        
        return info
    
    def get_available_models(self, category: str = "all") -> List[str]:
        """
        Get list of recommended models by category.
        
        Args:
            category: Model size category ('small', 'medium', 'large', 'xlarge', 'all')
            
        Returns:
            List of model names.
        """
        if category == "all":
            models = []
            for cat_models in self.RECOMMENDED_MODELS.values():
                models.extend(cat_models)
            return models
        
        return self.RECOMMENDED_MODELS.get(category, [])


class LocalLLMClient(LLMClientBase):
    """Client for local Ollama LLM server."""
    
    def __init__(self, base_url: str = "http://localhost:11434", model_name: str = "llama3.2"):
        self.base_url = base_url.rstrip('/')
        self.model_name = model_name
    
    def prompt(self, messages: List[Dict[str, str]], **kwargs) -> str:
        """Send a prompt to Ollama."""
        try:
            response = requests.post(
                f"{self.base_url}/api/chat",
                json={
                    "model": self.model_name,
                    "messages": messages,
                    "stream": False,
                    **kwargs
                },
                timeout=60
            )
            response.raise_for_status()
            return response.json()["message"]["content"]
        except Exception as e:
            logger.error(f"Ollama request failed: {e}")
            return f"Error: {str(e)}"
    
    def is_available(self) -> bool:
        """Check if Ollama server is running."""
        try:
            response = requests.get(f"{self.base_url}/api/tags", timeout=5)
            return response.status_code == 200
        except:
            return False
    
    def get_model_info(self) -> Dict[str, Any]:
        """Get information about the Ollama setup."""
        if not self.is_available():
            return {"status": "unavailable", "backend": "ollama"}
        
        try:
            response = requests.get(f"{self.base_url}/api/tags")
            models = response.json().get("models", [])
            return {
                "backend": "ollama",
                "model_name": self.model_name,
                "available_models": [m["name"] for m in models],
                "status": "ready"
            }
        except:
            return {"status": "error", "backend": "ollama"}


class RemoteLLMClient(LLMClientBase):
    """Client for remote LLM APIs (OpenRouter using OpenAI client)."""
    
    def __init__(self, 
                 api_key: Optional[str] = None, 
                 model_name: str = "openai/gpt-4o-mini",
                 base_url: str = "https://openrouter.ai/api/v1",
                 **kwargs):
        self.api_key = api_key or os.getenv("OPENROUTER_API_KEY")
        self.model_name = model_name
        self.base_url = base_url
        
        # Store additional generation parameters
        self.generation_params = {
            'temperature': kwargs.get('temperature', 0.7),
            'max_tokens': kwargs.get('max_tokens', 500),
            'top_p': kwargs.get('top_p', 0.9),
            'frequency_penalty': kwargs.get('frequency_penalty', 0.1),
            'presence_penalty': kwargs.get('presence_penalty', 0.1),
        }
        
        # Import OpenAI client only when needed
        try:
            from openai import OpenAI
            self.client = OpenAI(
                base_url=self.base_url,
                api_key=self.api_key
            )
            self._available = bool(self.api_key)
        except ImportError:
            logger.warning("OpenAI client not available for remote LLM")
            self._available = False
    
    def prompt(self, messages: List[Dict[str, str]], tools: List[Dict] = None, tool_choice: str = None, **kwargs) -> str:
        """Send a prompt to OpenRouter API using OpenAI client."""
        if not self._available or not self.api_key:
            return "Error: No API key provided or OpenAI client unavailable"
        
        try:
            # Merge default params with any provided kwargs
            params = self.generation_params.copy()
            params.update(kwargs)
            
            # Extract relevant parameters, handle legacy names
            max_tokens = params.get('max_new_tokens', params.get('max_tokens', 500))
            temperature = params.get('temperature', 0.7)
            top_p = params.get('top_p', 0.9)
            frequency_penalty = params.get('frequency_penalty', 0.1)
            presence_penalty = params.get('presence_penalty', 0.1)
            
            # Prepare the API call parameters
            api_params = {
                "model": self.model_name,
                "messages": messages,
                "max_tokens": max_tokens,
                "temperature": temperature,
                "top_p": top_p,
                "frequency_penalty": frequency_penalty,
                "presence_penalty": presence_penalty
            }
            
            # Add tools if provided
            if tools:
                api_params["tools"] = tools
                if tool_choice:
                    api_params["tool_choice"] = {"type": "function", "function": {"name": tool_choice}}
            
            response = self.client.chat.completions.create(**api_params)
            
            if not response or not response.choices:
                return "Error: Empty response from API"
                
            return response.choices[0].message.content
        except Exception as e:
            logger.error(f"Remote LLM request failed: {e}")
            return f"Error: {str(e)}"
    
    def generate(self, messages: List[Dict[str, str]], **kwargs) -> str:
        """Generate text using the remote LLM (alias for prompt method)."""
        return self.prompt(messages, **kwargs)
    
    def is_available(self) -> bool:
        """Check if remote API is available."""
        return self._available
    
    def get_model_info(self) -> Dict[str, Any]:
        """Get information about the remote setup."""
        return {
            "backend": "remote",
            "model_name": self.model_name,
            "api_available": self.is_available(),
            "status": "ready" if self.is_available() else "no_api_key"
        }


class UnifiedLLMClient:
    """
    Unified client that automatically selects the best available LLM backend.
    
    Fallback order: HuggingFace -> Ollama -> Remote
    """
    
    def __init__(self, backend: str = "auto", **kwargs):
        """
        Initialize unified client.
        
        Args:
            backend: Backend to use ('auto', 'huggingface', 'ollama', 'remote')
            **kwargs: Arguments passed to the selected backend
        """
        self.backend_preference = backend
        self.client = None
        self.active_backend = None
        
        # Initialize based on preference
        if backend == "auto":
            self._auto_select_backend(**kwargs)
        elif backend == "huggingface":
            self._try_huggingface(**kwargs)
        elif backend == "ollama":
            self._try_ollama(**kwargs)
        elif backend == "remote":
            self._try_remote(**kwargs)
        else:
            raise ValueError(f"Unknown backend: {backend}")
        
        if self.client is None:
            raise RuntimeError("No LLM backend available")
    
    def _auto_select_backend(self, **kwargs):
        """Automatically select the best available backend."""
        # Try HuggingFace first
        if self._try_huggingface(**kwargs):
            return
        
        # Try Ollama second
        if self._try_ollama(**kwargs):
            return
        
        # Try remote as fallback
        self._try_remote(**kwargs)
    
    def _try_huggingface(self, **kwargs) -> bool:
        """Try to initialize HuggingFace client."""
        if not HF_AVAILABLE:
            logger.info("HuggingFace not available")
            return False
        
        try:
            self.client = HuggingFaceClient(**kwargs)
            if self.client.is_available():
                self.active_backend = "huggingface"
                logger.info("Using HuggingFace backend")
                return True
        except Exception as e:
            logger.warning(f"HuggingFace initialization failed: {e}")
        
        return False
    
    def _try_ollama(self, **kwargs) -> bool:
        """Try to initialize Ollama client."""
        try:
            self.client = LocalLLMClient(**kwargs)
            if self.client.is_available():
                self.active_backend = "ollama"
                logger.info("Using Ollama backend")
                return True
        except Exception as e:
            logger.warning(f"Ollama initialization failed: {e}")
        
        return False
    
    def _try_remote(self, **kwargs) -> bool:
        """Try to initialize remote client."""
        try:
            self.client = RemoteLLMClient(**kwargs)
            if self.client.is_available():
                self.active_backend = "remote"
                logger.info("Using remote backend")
                return True
        except Exception as e:
            logger.warning(f"Remote initialization failed: {e}")
        
        return False
    
    def prompt(self, messages: List[Dict[str, str]], **kwargs) -> str:
        """Send a prompt using the active backend."""
        if self.client is None:
            return "Error: No LLM backend available"
        
        return self.client.prompt(messages, **kwargs)
    
    def is_available(self) -> bool:
        """Check if any backend is available."""
        return self.client is not None and self.client.is_available()
    
    def get_model_info(self) -> Dict[str, Any]:
        """Get information about the active backend."""
        if self.client is None:
            return {"status": "no_backend"}
        
        info = self.client.get_model_info()
        info["active_backend"] = self.active_backend
        return info
    
    def switch_model(self, model_name: str) -> bool:
        """
        Switch to a different model (HuggingFace only).
        
        Args:
            model_name: Name of the model to switch to.
            
        Returns:
            True if switch was successful.
        """
        if isinstance(self.client, HuggingFaceClient):
            return self.client.load_model(model_name)
        else:
            logger.warning(f"Model switching not supported for {self.active_backend}")
            return False


# Convenience function for quick setup
def create_llm_client(backend: str = "auto", **kwargs) -> UnifiedLLMClient:
    """
    Create a unified LLM client with the specified backend.
    
    Args:
        backend: Backend preference ('auto', 'huggingface', 'ollama', 'remote')
        **kwargs: Backend-specific arguments
        
    Returns:
        Configured UnifiedLLMClient instance
    """
    return UnifiedLLMClient(backend=backend, **kwargs)


if __name__ == "__main__":
    # Quick test
    logging.basicConfig(level=logging.INFO)
    
    try:
        client = create_llm_client()
        print("LLM Client Info:", json.dumps(client.get_model_info(), indent=2))
        
        response = client.prompt([
            {"role": "user", "content": "Hello! Can you tell me about yourself?"}
        ])
        print(f"Response: {response}")
        
    except Exception as e:
        print(f"Error: {e}")