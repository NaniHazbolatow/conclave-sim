#!/usr/bin/env python3
"""
Simple test for local embeddings on M1 Mac.
"""

import logging
from sentence_transformers import SentenceTransformer
import numpy as np
import time

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')
logger = logging.getLogger(__name__)

def test_basic_embedding():
    """Test basic embedding functionality."""
    logger.info("=== LOCAL EMBEDDING TEST ===")
    
    # Cardinal background text
    cardinal_text = """Cardinal Pietro Parolin, born in 1955 in Italy, is the Vatican Secretary of State and a leading figure in the Catholic Church's diplomacy. He was ordained in 1980 and entered the Holy See's diplomatic service in 1986, serving in various countries and as Undersecretary for Relations with States. Known for his pragmatic, conciliatory approach, Parolin emphasizes dialogue, international cooperation, and peaceful resolution of conflicts."""
    
    try:
        # Load model with MPS support
        logger.info("Loading sentence transformer model...")
        start_time = time.time()
        
        model = SentenceTransformer('intfloat/e5-large-v2', device='mps')
        
        load_time = time.time() - start_time
        logger.info(f"✅ Model loaded in {load_time:.2f} seconds")
        
        # Generate embedding
        logger.info("Generating embedding for cardinal background...")
        start_time = time.time()
        
        embedding = model.encode(cardinal_text)
        
        embed_time = time.time() - start_time
        logger.info(f"✅ Embedding generated in {embed_time:.3f} seconds")
        
        # Analyze embedding
        logger.info(f"Embedding dimension: {len(embedding)}")
        logger.info(f"Embedding stats: mean={np.mean(embedding):.4f}, std={np.std(embedding):.4f}")
        
        # Test multiple texts
        logger.info("\nTesting with multiple cardinal descriptions...")
        
        cardinal_descriptions = [
            "Cardinal Pietro Parolin, Vatican Secretary of State, diplomatic leader",
            "Cardinal Luis Antonio Tagle, progressive Filipino advocating social justice",
            "Cardinal Robert Sarah, conservative traditionalist emphasizing liturgy"
        ]
        
        start_time = time.time()
        embeddings = model.encode(cardinal_descriptions)
        batch_time = time.time() - start_time
        
        logger.info(f"✅ Batch embeddings generated in {batch_time:.3f} seconds")
        logger.info(f"Number of embeddings: {len(embeddings)}")
        
        # Calculate similarities
        def cosine_similarity(a, b):
            return np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b))
        
        logger.info("\nSimilarity matrix:")
        for i in range(len(embeddings)):
            for j in range(i+1, len(embeddings)):
                sim = cosine_similarity(embeddings[i], embeddings[j])
                name_i = cardinal_descriptions[i].split(',')[0].replace('Cardinal ', '')
                name_j = cardinal_descriptions[j].split(',')[0].replace('Cardinal ', '')
                logger.info(f"  {name_i} vs {name_j}: {sim:.4f}")
        
        logger.info("\n🎉 Local embedding test completed successfully!")
        return True
        
    except Exception as e:
        logger.error(f"❌ Test failed: {e}")
        return False

if __name__ == "__main__":
    test_basic_embedding()
