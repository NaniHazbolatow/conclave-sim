#!/usr/bin/env python3
"""
Test script for the conclave embedding module.

This script tests the embedding functionality with cardinal data.
"""

import logging
import time
import pandas as pd
from conclave.embeddings import EmbeddingClient, get_default_client, similarity, find_similar

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def test_embedding_client():
    """Test the EmbeddingClient class."""
    logger.info("=" * 60)
    logger.info("TESTING EMBEDDING CLIENT")
    logger.info("=" * 60)
    
    try:
        # Initialize client
        client = EmbeddingClient()
        
        # Test single embedding
        logger.info("Testing single text embedding...")
        text = "Cardinal Pietro Parolin is a skilled Vatican diplomat."
        embedding = client.get_embedding(text)
        logger.info(f"✅ Single embedding: dimension={len(embedding)}")
        
        # Test batch embeddings
        logger.info("Testing batch embeddings...")
        texts = [
            "Cardinal focused on social justice and climate action",
            "Conservative cardinal emphasizing traditional teachings",
            "Diplomatic cardinal with international experience"
        ]
        
        start_time = time.time()
        embeddings = client.get_embeddings(texts)
        batch_time = time.time() - start_time
        
        logger.info(f"✅ Batch embeddings: {len(embeddings)} embeddings in {batch_time:.3f}s")
        
        # Test similarity
        logger.info("Testing similarity calculation...")
        sim = client.cosine_similarity(texts[0], texts[1])
        logger.info(f"✅ Similarity between progressive and conservative: {sim:.4f}")
        
        # Test similarity search
        logger.info("Testing similarity search...")
        query = "Cardinal interested in environmental issues"
        results = client.find_most_similar(query, texts, top_k=2)
        
        logger.info(f"Query: '{query}'")
        for i, (text, score) in enumerate(results, 1):
            logger.info(f"  {i}. [{score:.4f}] {text}")
        
        # Test stats
        stats = client.get_embedding_stats()
        logger.info(f"✅ Client stats: {stats}")
        
        return True
        
    except Exception as e:
        logger.error(f"❌ EmbeddingClient test failed: {e}")
        return False


def test_convenience_functions():
    """Test the convenience functions."""
    logger.info("\n" + "=" * 60)
    logger.info("TESTING CONVENIENCE FUNCTIONS")
    logger.info("=" * 60)
    
    try:
        # Test individual functions
        logger.info("Testing convenience functions...")
        
        text1 = "Progressive cardinal advocating for social justice"
        text2 = "Conservative cardinal defending traditional doctrine"
        
        sim = similarity(text1, text2)
        logger.info(f"✅ Similarity function: {sim:.4f}")
        
        # Test find_similar
        candidates = [
            "Cardinal focused on liturgical traditions",
            "Cardinal working on climate change initiatives", 
            "Cardinal promoting interfaith dialogue",
            "Cardinal defending conservative theology"
        ]
        
        query = "Cardinal interested in environmental protection"
        results = find_similar(query, candidates, top_k=2)
        
        logger.info(f"Query: '{query}'")
        for i, (text, score) in enumerate(results, 1):
            logger.info(f"  {i}. [{score:.4f}] {text}")
        
        return True
        
    except Exception as e:
        logger.error(f"❌ Convenience functions test failed: {e}")
        return False


def test_with_cardinal_data():
    """Test with actual cardinal data from CSV."""
    logger.info("\n" + "=" * 60)
    logger.info("TESTING WITH CARDINAL DATA")
    logger.info("=" * 60)
    
    try:
        # Load cardinal data
        df = pd.read_csv('data/cardinal_electors_2025.csv')
        logger.info(f"Loaded {len(df)} cardinals from CSV")
        
        # Get client
        client = get_default_client()
        
        # Sample a few cardinals for testing
        sample_cardinals = df.head(10)
        
        cardinal_backgrounds = []
        cardinal_names = []
        
        for idx, row in sample_cardinals.iterrows():
            cardinal_names.append(row['Name'])
            cardinal_backgrounds.append(row['Background'])
        
        logger.info(f"Testing embeddings for {len(cardinal_backgrounds)} cardinals...")
        
        # Generate embeddings
        start_time = time.time()
        embeddings = client.get_embeddings(cardinal_backgrounds)
        embed_time = time.time() - start_time
        
        logger.info(f"✅ Generated embeddings in {embed_time:.2f}s")
        logger.info(f"Throughput: {len(cardinal_backgrounds)/embed_time:.1f} cardinals/second")
        
        # Test similarity search with different queries
        queries = [
            "Cardinal with diplomatic experience",
            "Cardinal focused on social justice",
            "Cardinal from developing nation",
            "Conservative traditional cardinal"
        ]
        
        for query in queries:
            logger.info(f"\nQuery: '{query}'")
            results = client.find_most_similar(query, cardinal_backgrounds, top_k=3)
            
            for i, (background, score) in enumerate(results, 1):
                # Find the cardinal name for this background
                name = cardinal_names[cardinal_backgrounds.index(background)]
                logger.info(f"  {i}. [{score:.4f}] {name}")
        
        # Test clustering
        logger.info(f"\nTesting clustering with similarity threshold 0.6...")
        clusters = client.cluster_similar_texts(cardinal_backgrounds, threshold=0.6)
        
        logger.info(f"Found {len(clusters)} clusters:")
        for i, cluster in enumerate(clusters, 1):
            if len(cluster) > 1:  # Only show clusters with multiple items
                names_in_cluster = [cardinal_names[cardinal_backgrounds.index(bg)] for bg in cluster]
                logger.info(f"  Cluster {i} ({len(cluster)} cardinals): {', '.join(names_in_cluster[:3])}{'...' if len(names_in_cluster) > 3 else ''}")
        
        return True
        
    except FileNotFoundError:
        logger.warning("❌ Cardinal CSV file not found. Skipping cardinal data test.")
        return True  # Not a failure, just missing data
    except Exception as e:
        logger.error(f"❌ Cardinal data test failed: {e}")
        return False


def test_performance():
    """Test performance characteristics."""
    logger.info("\n" + "=" * 60)
    logger.info("TESTING PERFORMANCE")
    logger.info("=" * 60)
    
    try:
        client = EmbeddingClient()
        
        # Test different batch sizes
        test_texts = [
            f"Cardinal number {i} with unique background and interests in theology and pastoral care."
            for i in range(50)
        ]
        
        for batch_size in [1, 5, 10, 20, 50]:
            test_batch = test_texts[:batch_size]
            
            start_time = time.time()
            embeddings = client.get_embeddings(test_batch)
            elapsed = time.time() - start_time
            
            throughput = batch_size / elapsed
            logger.info(f"Batch size {batch_size:2d}: {elapsed:.3f}s ({throughput:.1f} texts/sec)")
        
        # Test cache performance
        logger.info("\nTesting cache performance...")
        
        # First run (no cache)
        start_time = time.time()
        embeddings1 = client.get_embeddings(test_texts[:10])
        first_run = time.time() - start_time
        
        # Second run (with cache)
        start_time = time.time()
        embeddings2 = client.get_embeddings(test_texts[:10])
        second_run = time.time() - start_time
        
        speedup = first_run / second_run if second_run > 0 else float('inf')
        logger.info(f"First run: {first_run:.3f}s")
        logger.info(f"Cached run: {second_run:.3f}s")
        logger.info(f"Cache speedup: {speedup:.1f}x")
        
        return True
        
    except Exception as e:
        logger.error(f"❌ Performance test failed: {e}")
        return False


def main():
    """Run all embedding tests."""
    logger.info("🚀 Starting comprehensive embedding tests...")
    
    success = True
    success &= test_embedding_client()
    success &= test_convenience_functions()
    success &= test_with_cardinal_data()
    success &= test_performance()
    
    if success:
        logger.info("\n🎉 All embedding tests passed!")
        logger.info("Local embedding system is ready for use in your conclave simulation.")
    else:
        logger.error("\n💥 Some tests failed. Please check the logs above.")
    
    return success


if __name__ == "__main__":
    main()
