#!/usr/bin/env python3
"""
Test script for remote vector embedding models via OpenAI API.

This script tests the connection and functionality of vector embedding models
using OpenAI's embedding API.
"""

import os
import logging
import requests
import numpy as np
from typing import List, Optional, Dict, Any
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class RemoteEmbeddingClient:
    """Client for remote vector embeddings via OpenAI API."""
    
    def __init__(self, 
                 api_key: Optional[str] = None,
                 base_url: str = "https://api.openai.com/v1",
                 model_name: str = "text-embedding-3-small"):
        """
        Initialize the remote embedding client.
        
        Args:
            api_key: OpenAI API key. If None, will try to get from environment.
            base_url: Base URL for the OpenAI API
            model_name: Name of the embedding model to use
        """
        self.api_key = api_key or os.getenv("OPENAI_API_KEY")
        self.base_url = base_url
        self.model_name = model_name
        
        if not self.api_key:
            raise ValueError("OpenAI API key not found. Set OPENAI_API_KEY environment variable.")
        
        self.headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json"
        }
        
        logger.info(f"Initialized RemoteEmbeddingClient with model: {self.model_name}")
    
    def get_embedding(self, text: str) -> List[float]:
        """
        Get vector embedding for a single text.
        
        Args:
            text: Text to embed
            
        Returns:
            List of floats representing the embedding vector
        """
        return self.get_embeddings([text])[0]
    
    def get_embeddings(self, texts: List[str]) -> List[List[float]]:
        """
        Get vector embeddings for multiple texts.
        
        Args:
            texts: List of texts to embed
            
        Returns:
            List of embedding vectors
        """
        payload = {
            "model": self.model_name,
            "input": texts
        }
        
        try:
            logger.info(f"Requesting embeddings for {len(texts)} texts...")
            response = requests.post(
                f"{self.base_url}/embeddings",
                headers=self.headers,
                json=payload,
                timeout=30
            )
            
            response.raise_for_status()
            data = response.json()
            
            # Extract embeddings from response
            embeddings = []
            for item in data.get("data", []):
                embeddings.append(item.get("embedding", []))
            
            logger.info(f"Successfully retrieved {len(embeddings)} embeddings")
            return embeddings
            
        except requests.exceptions.RequestException as e:
            logger.error(f"API request failed: {e}")
            raise
        except KeyError as e:
            logger.error(f"Unexpected response format: {e}")
            logger.error(f"Response: {response.text if 'response' in locals() else 'No response'}")
            raise
    
    def test_connection(self) -> bool:
        """
        Test the connection to OpenRouter API.
        
        Returns:
            True if connection successful, False otherwise
        """
        try:
            test_text = "Hello, world!"
            embedding = self.get_embedding(test_text)
            
            if embedding and len(embedding) > 0:
                logger.info(f"Connection test successful! Embedding dimension: {len(embedding)}")
                return True
            else:
                logger.error("Connection test failed: Empty embedding returned")
                return False
                
        except Exception as e:
            logger.error(f"Connection test failed: {e}")
            return False


def test_cardinal_embedding():
    """Test embedding a cardinal's background."""
    
    # Sample cardinal background from the CSV
    cardinal_background = """Cardinal Pietro Parolin, born in 1955 in Italy, is the Vatican Secretary of State and a leading figure in the Catholic Church's diplomacy. He was ordained in 1980 and entered the Holy See's diplomatic service in 1986, serving in various countries and as Undersecretary for Relations with States. Known for his pragmatic, conciliatory approach, Parolin emphasizes dialogue, international cooperation, and peaceful resolution of conflicts. He supports Pope Francis's reforms and priorities, including social justice, care for migrants, and climate action, while upholding traditional Catholic teachings. Parolin is considered a skilled negotiator and a key architect of Vatican agreements with China and other countries."""
    
    try:
        logger.info("Testing remote embedding client with cardinal background...")
        
        # Initialize the client
        client = RemoteEmbeddingClient()
        
        # Test connection first
        if not client.test_connection():
            logger.error("Connection test failed. Exiting.")
            return False
        
        # Get embedding for the cardinal's background
        logger.info("Getting embedding for Cardinal Parolin's background...")
        embedding = client.get_embedding(cardinal_background)
        
        # Analyze the embedding
        embedding_array = np.array(embedding)
        logger.info(f"Embedding stats:")
        logger.info(f"  - Dimension: {len(embedding)}")
        logger.info(f"  - Mean: {embedding_array.mean():.6f}")
        logger.info(f"  - Std: {embedding_array.std():.6f}")
        logger.info(f"  - Min: {embedding_array.min():.6f}")
        logger.info(f"  - Max: {embedding_array.max():.6f}")
        logger.info(f"  - Norm (L2): {np.linalg.norm(embedding_array):.6f}")
        
        # Test with multiple texts
        logger.info("\nTesting batch embedding with multiple cardinal descriptions...")
        
        cardinal_descriptions = [
            "Cardinal Pietro Parolin, Vatican Secretary of State, known for diplomatic skills.",
            "Cardinal Luis Antonio Tagle, progressive Filipino leader advocating social justice.",
            "Cardinal Robert Sarah, conservative traditionalist emphasizing liturgical orthodoxy."
        ]
        
        batch_embeddings = client.get_embeddings(cardinal_descriptions)
        
        logger.info(f"Batch embedding successful!")
        logger.info(f"Number of embeddings: {len(batch_embeddings)}")
        
        # Calculate similarities between embeddings
        if len(batch_embeddings) >= 3:
            emb1 = np.array(batch_embeddings[0])
            emb2 = np.array(batch_embeddings[1])
            emb3 = np.array(batch_embeddings[2])
            
            # Cosine similarity
            def cosine_similarity(a, b):
                return np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b))
            
            sim_1_2 = cosine_similarity(emb1, emb2)
            sim_1_3 = cosine_similarity(emb1, emb3)
            sim_2_3 = cosine_similarity(emb2, emb3)
            
            logger.info(f"\nCosine similarities:")
            logger.info(f"  - Parolin vs Tagle: {sim_1_2:.4f}")
            logger.info(f"  - Parolin vs Sarah: {sim_1_3:.4f}")
            logger.info(f"  - Tagle vs Sarah: {sim_2_3:.4f}")
        
        logger.info("\n✅ Remote embedding test completed successfully!")
        return True
        
    except Exception as e:
        logger.error(f"❌ Remote embedding test failed: {e}")
        return False


def main():
    """Main test function."""
    logger.info("=" * 60)
    logger.info("REMOTE VECTOR EMBEDDING TEST")
    logger.info("=" * 60)
    
    # Check for API key
    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        logger.error("❌ OPENAI_API_KEY environment variable not set!")
        logger.info("Please set your OpenAI API key:")
        logger.info("export OPENAI_API_KEY='your-api-key-here'")
        return False
    
    logger.info(f"✅ API key found: {api_key[:8]}...{api_key[-4:]}")
    
    # Run the test
    success = test_cardinal_embedding()
    
    if success:
        logger.info("\n🎉 All tests passed! Remote embedding setup is working correctly.")
    else:
        logger.error("\n💥 Tests failed. Please check the configuration and try again.")
    
    return success


if __name__ == "__main__":
    main()
