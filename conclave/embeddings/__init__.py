"""
Vector embedding client for the Conclave simulation.

This module provides local vector embeddings using sentence-transformers,
optimized for M1 Mac with MPS acceleration.
"""

import logging
import time
from typing import List, Dict, Optional, Tuple, Union
import numpy as np

logger = logging.getLogger(__name__)


class EmbeddingClient:
    """Local vector embedding client using sentence-transformers."""
    
    def __init__(self, 
                 model_name: str = "all-MiniLM-L6-v2",
                 device: Optional[str] = None,
                 cache_embeddings: bool = True):
        """
        Initialize the embedding client.
        
        Args:
            model_name: Name of the sentence-transformer model
            device: Device to use ('mps', 'cpu', or None for auto-detection)
            cache_embeddings: Whether to cache embeddings to avoid recomputation
        """
        self.model_name = model_name
        self.device = device or self._get_optimal_device()
        self.cache_embeddings = cache_embeddings
        self.model = None
        self._embedding_cache = {} if cache_embeddings else None
        
        logger.info(f"Initialized EmbeddingClient with model: {self.model_name}")
        logger.info(f"Target device: {self.device}")
    
    def _get_optimal_device(self) -> str:
        """Determine the optimal device for this system."""
        try:
            import torch
            if torch.backends.mps.is_available():
                return "mps"  # M1/M2 Mac acceleration
            elif torch.cuda.is_available():
                return "cuda"  # NVIDIA GPU
            else:
                return "cpu"
        except ImportError:
            return "cpu"
    
    def _load_model(self):
        """Load the sentence transformer model if not already loaded."""
        if self.model is not None:
            return
            
        try:
            from sentence_transformers import SentenceTransformer
            
            logger.info(f"Loading model {self.model_name} on {self.device}...")
            start_time = time.time()
            
            self.model = SentenceTransformer(self.model_name, device=self.device)
            
            load_time = time.time() - start_time
            logger.info(f"Model loaded successfully in {load_time:.2f} seconds")
            
        except ImportError:
            raise ImportError(
                "sentence-transformers not installed. "
                "Install with: uv add sentence-transformers"
            )
        except Exception as e:
            logger.warning(f"Failed to load model with {self.device}, falling back to CPU")
            self.device = "cpu"
            self.model = SentenceTransformer(self.model_name, device=self.device)
    
    def get_embedding(self, text: str) -> np.ndarray:
        """
        Get vector embedding for a single text.
        
        Args:
            text: Text to embed
            
        Returns:
            NumPy array representing the embedding vector
        """
        # Check cache first
        if self.cache_embeddings and text in self._embedding_cache:
            return self._embedding_cache[text]
        
        self._load_model()
        
        # Generate embedding
        embedding = self.model.encode(text, convert_to_numpy=True)
        
        # Cache if enabled
        if self.cache_embeddings:
            self._embedding_cache[text] = embedding
        
        return embedding
    
    def get_embeddings(self, texts: List[str]) -> np.ndarray:
        """
        Get vector embeddings for multiple texts.
        
        Args:
            texts: List of texts to embed
            
        Returns:
            NumPy array with shape (n_texts, embedding_dim)
        """
        self._load_model()
        
        # Check cache for existing embeddings
        if self.cache_embeddings:
            cached_embeddings = {}
            texts_to_compute = []
            
            for text in texts:
                if text in self._embedding_cache:
                    cached_embeddings[text] = self._embedding_cache[text]
                else:
                    texts_to_compute.append(text)
        else:
            texts_to_compute = texts
            cached_embeddings = {}
        
        # Compute embeddings for new texts
        if texts_to_compute:
            logger.debug(f"Computing embeddings for {len(texts_to_compute)} new texts...")
            new_embeddings = self.model.encode(texts_to_compute, convert_to_numpy=True)
            
            # Cache new embeddings
            if self.cache_embeddings:
                for text, embedding in zip(texts_to_compute, new_embeddings):
                    self._embedding_cache[text] = embedding
        
        # Combine cached and new embeddings in original order
        result_embeddings = []
        new_embedding_idx = 0
        
        for text in texts:
            if text in cached_embeddings:
                result_embeddings.append(cached_embeddings[text])
            else:
                result_embeddings.append(new_embeddings[new_embedding_idx])
                new_embedding_idx += 1
        
        return np.array(result_embeddings)
    
    def cosine_similarity(self, 
                         embedding1: Union[str, np.ndarray], 
                         embedding2: Union[str, np.ndarray]) -> float:
        """
        Calculate cosine similarity between two embeddings.
        
        Args:
            embedding1: First embedding (text or embedding vector)
            embedding2: Second embedding (text or embedding vector)
            
        Returns:
            Cosine similarity score between -1 and 1
        """
        # Convert texts to embeddings if needed
        if isinstance(embedding1, str):
            embedding1 = self.get_embedding(embedding1)
        if isinstance(embedding2, str):
            embedding2 = self.get_embedding(embedding2)
        
        # Calculate cosine similarity
        dot_product = np.dot(embedding1, embedding2)
        norm1 = np.linalg.norm(embedding1)
        norm2 = np.linalg.norm(embedding2)
        
        if norm1 == 0 or norm2 == 0:
            return 0.0
        
        return dot_product / (norm1 * norm2)
    
    def find_most_similar(self, 
                         query: str, 
                         candidates: List[str], 
                         top_k: int = 5) -> List[Tuple[str, float]]:
        """
        Find the most similar candidates to a query.
        
        Args:
            query: Query text
            candidates: List of candidate texts to compare against
            top_k: Number of top results to return
            
        Returns:
            List of (candidate_text, similarity_score) tuples, sorted by similarity
        """
        query_embedding = self.get_embedding(query)
        candidate_embeddings = self.get_embeddings(candidates)
        
        similarities = []
        for candidate, candidate_embedding in zip(candidates, candidate_embeddings):
            similarity = self.cosine_similarity(query_embedding, candidate_embedding)
            similarities.append((candidate, similarity))
        
        # Sort by similarity (descending)
        similarities.sort(key=lambda x: x[1], reverse=True)
        
        return similarities[:top_k]
    
    def cluster_similar_texts(self, 
                            texts: List[str], 
                            threshold: float = 0.7) -> List[List[str]]:
        """
        Group texts into clusters based on semantic similarity.
        
        Args:
            texts: List of texts to cluster
            threshold: Similarity threshold for grouping
            
        Returns:
            List of clusters, where each cluster is a list of similar texts
        """
        embeddings = self.get_embeddings(texts)
        
        clusters = []
        used = set()
        
        for i, text in enumerate(texts):
            if i in used:
                continue
            
            cluster = [text]
            used.add(i)
            
            for j, other_text in enumerate(texts):
                if j in used or i == j:
                    continue
                
                similarity = self.cosine_similarity(embeddings[i], embeddings[j])
                if similarity >= threshold:
                    cluster.append(other_text)
                    used.add(j)
            
            clusters.append(cluster)
        
        return clusters
    
    def get_embedding_stats(self) -> Dict[str, any]:
        """Get statistics about the embedding model and cache."""
        stats = {
            "model_name": self.model_name,
            "device": self.device,
            "cache_enabled": self.cache_embeddings,
            "cached_embeddings": len(self._embedding_cache) if self._embedding_cache else 0
        }
        
        if self.model is not None:
            # Test embedding to get dimension
            test_embedding = self.get_embedding("test")
            stats["embedding_dimension"] = len(test_embedding)
        
        return stats
    
    def clear_cache(self):
        """Clear the embedding cache."""
        if self._embedding_cache:
            self._embedding_cache.clear()
            logger.info("Embedding cache cleared")


# Global client instance for easy access
_default_client = None


def get_default_client() -> EmbeddingClient:
    """Get the default embedding client instance."""
    global _default_client
    if _default_client is None:
        _default_client = EmbeddingClient()
    return _default_client


def embed_text(text: str) -> np.ndarray:
    """Convenience function to embed a single text using the default client."""
    return get_default_client().get_embedding(text)


def similarity(text1: str, text2: str) -> float:
    """Convenience function to calculate similarity between two texts."""
    return get_default_client().cosine_similarity(text1, text2)


def find_similar(query: str, 
                candidates: List[str], 
                top_k: int = 5) -> List[Tuple[str, float]]:
    """Convenience function to find similar texts using the default client."""
    return get_default_client().find_most_similar(query, candidates, top_k)
