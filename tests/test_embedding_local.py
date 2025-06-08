#!/usr/bin/env python3
"""
Test script for local vector embedding models.

This script tests local embedding models that run efficiently on M1 Mac.
Uses sentence-transformers library for high-quality embeddings.
"""

import os
import logging
import numpy as np
from typing import List, Optional, Dict, Any
import time

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class LocalEmbeddingClient:
    """Client for local vector embeddings using sentence-transformers."""
    
    def __init__(self, 
                 model_name: str = "all-MiniLM-L6-v2",
                 device: str = "mps"):  # Use MPS for M1 Mac acceleration
        """
        Initialize the local embedding client.
        
        Args:
            model_name: Name of the sentence-transformer model to use
            device: Device to run on ('mps' for M1 Mac, 'cpu' for fallback)
        """
        self.model_name = model_name
        self.device = device
        self.model = None
        
        logger.info(f"Initializing LocalEmbeddingClient with model: {self.model_name}")
        logger.info(f"Target device: {self.device}")
        
    def _load_model(self):
        """Load the sentence transformer model."""
        if self.model is not None:
            return
            
        try:
            from sentence_transformers import SentenceTransformer
            
            logger.info(f"Loading model {self.model_name}...")
            start_time = time.time()
            
            # Load model with MPS support for M1 Mac
            self.model = SentenceTransformer(self.model_name, device=self.device)
            
            load_time = time.time() - start_time
            logger.info(f"Model loaded successfully in {load_time:.2f} seconds")
            
        except ImportError:
            logger.error("sentence-transformers not installed. Installing...")
            raise ImportError("Please install sentence-transformers: uv add sentence-transformers")
        except Exception as e:
            logger.warning(f"Failed to load model with {self.device}, falling back to CPU")
            self.device = "cpu"
            self.model = SentenceTransformer(self.model_name, device=self.device)
    
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
        self._load_model()
        
        try:
            logger.info(f"Generating embeddings for {len(texts)} texts using {self.device}...")
            start_time = time.time()
            
            # Generate embeddings
            embeddings = self.model.encode(texts, convert_to_numpy=True)
            
            embed_time = time.time() - start_time
            logger.info(f"Embeddings generated in {embed_time:.2f} seconds")
            
            # Convert to list of lists
            return embeddings.tolist()
            
        except Exception as e:
            logger.error(f"Failed to generate embeddings: {e}")
            raise
    
    def test_connection(self) -> bool:
        """
        Test the local model loading and basic functionality.
        
        Returns:
            True if test successful, False otherwise
        """
        try:
            test_text = "Hello, world!"
            embedding = self.get_embedding(test_text)
            
            if embedding and len(embedding) > 0:
                logger.info(f"Connection test successful! Embedding dimension: {len(embedding)}")
                logger.info(f"Device being used: {self.device}")
                return True
            else:
                logger.error("Connection test failed: Empty embedding returned")
                return False
                
        except Exception as e:
            logger.error(f"Connection test failed: {e}")
            return False
    
    def get_model_info(self) -> Dict[str, Any]:
        """Get information about the loaded model."""
        self._load_model()
        
        try:
            # Get model dimensions
            test_embedding = self.model.encode(["test"])
            dimension = test_embedding.shape[1]
            
            return {
                "model_name": self.model_name,
                "device": self.device,
                "embedding_dimension": dimension,
                "max_sequence_length": getattr(self.model, 'max_seq_length', 'Unknown')
            }
        except Exception as e:
            logger.error(f"Failed to get model info: {e}")
            return {}


def test_cardinal_embedding():
    """Test embedding a cardinal's background."""
    
    # Sample cardinal background from the CSV
    cardinal_background = """Cardinal Pietro Parolin, born in 1955 in Italy, is the Vatican Secretary of State and a leading figure in the Catholic Church's diplomacy. He was ordained in 1980 and entered the Holy See's diplomatic service in 1986, serving in various countries and as Undersecretary for Relations with States. Known for his pragmatic, conciliatory approach, Parolin emphasizes dialogue, international cooperation, and peaceful resolution of conflicts. He supports Pope Francis's reforms and priorities, including social justice, care for migrants, and climate action, while upholding traditional Catholic teachings. Parolin is considered a skilled negotiator and a key architect of Vatican agreements with China and other countries."""
    
    try:
        logger.info("Testing local embedding client with cardinal background...")
        
        # Test different model sizes
        models_to_test = [
            "all-MiniLM-L6-v2",      # Fast, 384 dimensions
            "all-mpnet-base-v2",     # Better quality, 768 dimensions
        ]
        
        for model_name in models_to_test:
            logger.info(f"\n--- Testing model: {model_name} ---")
            
            # Initialize the client
            client = LocalEmbeddingClient(model_name=model_name)
            
            # Test connection first
            if not client.test_connection():
                logger.error(f"Connection test failed for {model_name}. Skipping.")
                continue
            
            # Get model info
            model_info = client.get_model_info()
            logger.info(f"Model info: {model_info}")
            
            # Get embedding for the cardinal's background
            logger.info("Getting embedding for Cardinal Parolin's background...")
            start_time = time.time()
            embedding = client.get_embedding(cardinal_background)
            embed_time = time.time() - start_time
            
            # Analyze the embedding
            embedding_array = np.array(embedding)
            logger.info(f"Embedding stats (generated in {embed_time:.3f}s):")
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
                "Cardinal Robert Sarah, conservative traditionalist emphasizing liturgical orthodoxy.",
                "Cardinal Marc Ouellet, Canadian theologian focused on ecclesiology and liturgy.",
                "Cardinal Peter Turkson, Ghanaian advocate for peace and sustainable development."
            ]
            
            start_time = time.time()
            batch_embeddings = client.get_embeddings(cardinal_descriptions)
            batch_time = time.time() - start_time
            
            logger.info(f"Batch embedding successful! ({batch_time:.3f}s for {len(cardinal_descriptions)} texts)")
            logger.info(f"Number of embeddings: {len(batch_embeddings)}")
            
            # Calculate similarities between embeddings
            if len(batch_embeddings) >= 3:
                embeddings_np = np.array(batch_embeddings)
                
                # Cosine similarity function
                def cosine_similarity(a, b):
                    return np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b))
                
                logger.info(f"\nCosine similarities:")
                for i in range(len(cardinal_descriptions)):
                    for j in range(i + 1, min(i + 3, len(cardinal_descriptions))):  # Limit output
                        sim = cosine_similarity(embeddings_np[i], embeddings_np[j])
                        name_i = cardinal_descriptions[i].split(',')[0].replace('Cardinal ', '')
                        name_j = cardinal_descriptions[j].split(',')[0].replace('Cardinal ', '')
                        logger.info(f"  - {name_i} vs {name_j}: {sim:.4f}")
            
            # Performance summary
            texts_per_second = len(cardinal_descriptions) / batch_time
            logger.info(f"\nPerformance: {texts_per_second:.1f} texts/second")
            
            logger.info(f"\n✅ Model {model_name} test completed successfully!")
        
        return True
        
    except Exception as e:
        logger.error(f"❌ Local embedding test failed: {e}")
        return False


def test_similarity_search():
    """Test semantic similarity search with cardinal descriptions."""
    
    try:
        logger.info("\n" + "="*50)
        logger.info("TESTING SIMILARITY SEARCH")
        logger.info("="*50)
        
        # Use the fastest model for this test
        client = LocalEmbeddingClient(model_name="all-MiniLM-L6-v2")
        
        # Sample cardinal profiles
        cardinal_profiles = [
            "Cardinal focused on traditional liturgy and conservative theology",
            "Progressive cardinal advocating for social justice and climate action", 
            "Diplomatic cardinal with experience in international relations",
            "Pastoral cardinal emphasizing care for the poor and marginalized",
            "Academic cardinal with expertise in scripture and theology",
            "Cardinal from developing nation championing global equity",
        ]
        
        # Query
        query = "Cardinal interested in environmental issues and social reform"
        
        logger.info(f"Query: {query}")
        logger.info("\nCardinal profiles to search:")
        for i, profile in enumerate(cardinal_profiles):
            logger.info(f"  {i+1}. {profile}")
        
        # Get embeddings
        logger.info("\nGenerating embeddings...")
        query_embedding = client.get_embedding(query)
        profile_embeddings = client.get_embeddings(cardinal_profiles)
        
        # Calculate similarities
        query_emb = np.array(query_embedding)
        similarities = []
        
        for i, profile_emb in enumerate(profile_embeddings):
            profile_emb = np.array(profile_emb)
            similarity = np.dot(query_emb, profile_emb) / (np.linalg.norm(query_emb) * np.linalg.norm(profile_emb))
            similarities.append((i, similarity, cardinal_profiles[i]))
        
        # Sort by similarity
        similarities.sort(key=lambda x: x[1], reverse=True)
        
        logger.info(f"\nSimilarity ranking:")
        for rank, (idx, sim, profile) in enumerate(similarities, 1):
            logger.info(f"  {rank}. [{sim:.4f}] {profile}")
        
        logger.info(f"\n✅ Similarity search test completed!")
        return True
        
    except Exception as e:
        logger.error(f"❌ Similarity search test failed: {e}")
        return False


def main():
    """Main test function."""
    logger.info("=" * 60)
    logger.info("LOCAL VECTOR EMBEDDING TEST")
    logger.info("=" * 60)
    
    # Check for required dependencies
    try:
        import sentence_transformers
        logger.info("✅ sentence-transformers is available")
    except ImportError:
        logger.error("❌ sentence-transformers not found!")
        logger.info("Install with: uv add sentence-transformers")
        return False
    
    try:
        import torch
        if torch.backends.mps.is_available():
            logger.info("✅ MPS (Metal Performance Shaders) is available for M1 acceleration")
        else:
            logger.info("⚠️  MPS not available, will use CPU")
    except ImportError:
        logger.info("⚠️  PyTorch not found, sentence-transformers will install it")
    
    # Run the tests
    success = True
    
    success &= test_cardinal_embedding()
    success &= test_similarity_search()
    
    if success:
        logger.info("\n🎉 All local embedding tests passed!")
        logger.info("Local embedding setup is working correctly on your M1 Mac.")
    else:
        logger.error("\n💥 Some tests failed. Please check the logs above.")
    
    return success


if __name__ == "__main__":
    main()
