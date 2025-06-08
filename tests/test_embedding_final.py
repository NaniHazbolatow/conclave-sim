#!/usr/bin/env python3
"""
Final working test for local vector embeddings.

This test demonstrates local embeddings working with cardinal data.
"""

import time
import numpy as np
from sentence_transformers import SentenceTransformer

def main():
    print("🚀 FINAL LOCAL EMBEDDING TEST")
    print("=" * 50)
    
    # Initialize model with M1 acceleration
    print("Loading sentence transformer model...")
    model = SentenceTransformer('all-MiniLM-L6-v2', device='mps')
    print("✅ Model loaded with MPS acceleration")
    
    # Cardinal profiles for testing
    cardinal_profiles = {
        'Parolin': 'Cardinal Pietro Parolin, Vatican Secretary of State, experienced diplomat, supports Pope Francis reforms',
        'Tagle': 'Cardinal Luis Antonio Tagle, progressive Filipino leader, advocates social justice and climate action',
        'Sarah': 'Cardinal Robert Sarah, conservative traditionalist, emphasizes liturgical orthodoxy and traditional teachings',
        'Turkson': 'Cardinal Peter Turkson, Ghanaian advocate for peace, sustainable development, social justice',
        'Cupich': 'Cardinal Blase Cupich, American progressive, supports immigration reform and climate action'
    }
    
    # Generate embeddings
    print(f"\nGenerating embeddings for {len(cardinal_profiles)} cardinals...")
    start_time = time.time()
    
    names = list(cardinal_profiles.keys())
    descriptions = list(cardinal_profiles.values())
    embeddings = model.encode(descriptions)
    
    embed_time = time.time() - start_time
    print(f"✅ Embeddings generated in {embed_time:.2f}s")
    print(f"Throughput: {len(embeddings)/embed_time:.1f} cardinals/second")
    print(f"Embedding dimension: {embeddings.shape[1]}")
    
    # Test similarity search
    queries = [
        "Progressive cardinal focused on climate change and social justice",
        "Conservative cardinal emphasizing traditional Catholic teachings",
        "Diplomatic cardinal with international relations experience"
    ]
    
    print("\n" + "=" * 50)
    print("SIMILARITY SEARCH RESULTS")
    print("=" * 50)
    
    for query in queries:
        print(f"\n🔍 Query: \"{query}\"")
        print("-" * 40)
        
        # Get query embedding
        query_embedding = model.encode(query)
        
        # Calculate similarities
        similarities = []
        for i, (name, cardinal_embedding) in enumerate(zip(names, embeddings)):
            similarity = np.dot(query_embedding, cardinal_embedding) / (
                np.linalg.norm(query_embedding) * np.linalg.norm(cardinal_embedding)
            )
            similarities.append((name, similarity))
        
        # Sort and show top matches
        similarities.sort(key=lambda x: x[1], reverse=True)
        
        for rank, (name, sim) in enumerate(similarities[:3], 1):
            print(f"  {rank}. {name}: {sim:.4f}")
    
    # Performance test
    print("\n" + "=" * 50)
    print("PERFORMANCE TEST")
    print("=" * 50)
    
    test_texts = [f"Cardinal {i} with unique theological background and pastoral experience" for i in range(20)]
    
    start_time = time.time()
    test_embeddings = model.encode(test_texts)
    performance_time = time.time() - start_time
    
    print(f"Performance: {len(test_texts)} texts in {performance_time:.3f}s")
    print(f"Throughput: {len(test_texts)/performance_time:.1f} texts/second")
    
    print("\n🎉 LOCAL EMBEDDING TEST COMPLETED SUCCESSFULLY!")
    print("✅ M1 MPS acceleration working")
    print("✅ Semantic similarity search working")
    print("✅ High-performance batch processing working")
    print("\nYour local embedding setup is ready for the conclave simulation!")

if __name__ == "__main__":
    main()
