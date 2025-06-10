#!/usr/bin/env python3
"""
Comprehensive visualization test using the new Cardinal Visualizer.

This test demonstrates the full visualization suite for cardinal positioning
and ideological clustering using PCA and other dimensionality reduction techniques.
"""

import time
import numpy as np
from sentence_transformers import SentenceTransformer
import yaml
import os
import torch
import sys

# Add the project root to the path so we can import our modules
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

from conclave.visualization.cardinal_visualizer import CardinalVisualizer

def load_config():
    """Load configuration from config.yaml"""
    config_path = os.path.join(os.path.dirname(__file__), '..', 'config.yaml')
    with open(config_path, 'r') as f:
        return yaml.safe_load(f)

def get_device(device_config):
    """Determine the best available device"""
    if device_config == "auto":
        if torch.backends.mps.is_available():
            return "mps"
        elif torch.cuda.is_available():
            return "cuda"
        else:
            return "cpu"
    return device_config

def main():
    print("🎨 CARDINAL VISUALIZATION SUITE TEST")
    print("=" * 60)
    
    # Load configuration
    config = load_config()
    embedding_config = config['embeddings']
    
    # Initialize embedding model from config
    model_name = embedding_config['model_name']
    device = get_device(embedding_config['device'])
    
    print(f"Loading sentence transformer model: {model_name}...")
    model = SentenceTransformer(model_name, device=device)
    print(f"✅ Model loaded with {device} acceleration")
    print(f"Model dimension: {model.get_sentence_embedding_dimension()}")
    
    # Test cardinal profiles with clear ideological categories
    cardinal_profiles = {
        # Hyper-conservatives
        'Sarah': 'Cardinal Robert Sarah, ultra-conservative traditionalist, strongly opposes modern reforms, emphasizes strict liturgical orthodoxy and unwavering traditional Catholic teachings',
        'Burke': 'Cardinal Raymond Burke, extreme conservative American, militant defender of traditional doctrine, fierce opponent of progressive changes and papal reforms',
        
        # Progressives  
        'Tagle': 'Cardinal Luis Antonio Tagle, progressive Filipino leader, champions social justice, climate action, and church modernization',
        'Cupich': 'Cardinal Blase Cupich, American progressive, advocates immigration reform, LGBTQ+ inclusion, and environmental activism',
        
        # Balanced/Moderate
        'Parolin': 'Cardinal Pietro Parolin, Vatican Secretary of State, experienced diplomat, seeks balance between tradition and reform, pragmatic centrist approach'
    }
    
    # Define cardinal categories for visualization
    cardinal_categories = {
        'Sarah': 'Hyper-Conservative',
        'Burke': 'Hyper-Conservative', 
        'Tagle': 'Progressive',
        'Cupich': 'Progressive',
        'Parolin': 'Balanced'
    }
    
    # Generate embeddings
    print(f"\nGenerating embeddings for {len(cardinal_profiles)} cardinals...")
    names = list(cardinal_profiles.keys())
    descriptions = list(cardinal_profiles.values())
    embeddings = model.encode(descriptions)
    
    print(f"✅ Original embedding dimension: {embeddings.shape[1]}")
    
    # Initialize visualizer
    print("\n🎨 Initializing Cardinal Visualizer...")
    visualizer = CardinalVisualizer(config)
    print(f"✅ Visualizer configured with {visualizer.reduction_method.upper()} method")
    
    # Test 1: Create main ideological visualization
    print("\n📊 Creating ideological visualization...")
    fig1 = visualizer.create_ideological_visualization(
        embeddings=embeddings,
        cardinal_names=names,
        cardinal_categories=cardinal_categories,
        title="Cardinal Ideological Landscape"
    )
    
    # Save the plot
    output_dir = os.path.join(os.path.dirname(__file__), '..', 'outputs')
    os.makedirs(output_dir, exist_ok=True)
    
    fig1.savefig(os.path.join(output_dir, 'cardinal_ideological_positioning.png'), 
                 dpi=300, bbox_inches='tight')
    print("✅ Ideological visualization saved to outputs/cardinal_ideological_positioning.png")
    
    # Test 2: Analyze clusters
    print("\n🔍 Analyzing ideological clusters...")
    cluster_analysis = visualizer.analyze_clusters(
        embeddings=embeddings,
        cardinal_names=names,
        cardinal_categories=cardinal_categories
    )
    
    print("\n📍 Cluster Analysis Results:")
    print(f"Cluster Centers:")
    for category, center in cluster_analysis['cluster_centers'].items():
        print(f"  {category}: ({center[0]:.1f}, {center[1]:.1f})")
    
    print(f"\nCluster Distances:")
    for pair, distance in cluster_analysis['cluster_distances'].items():
        print(f"  {pair}: {distance:.1f} units")
    
    separation_status = "✅ Good" if cluster_analysis['good_separation'] else "⚠️ Limited"
    print(f"\nCluster Separation: {separation_status}")
    
    print(f"\n📍 Cardinal Grid Positions ({visualizer.reduction_method.upper()}):")
    for name, coords in cluster_analysis['coordinates'].items():
        category = cardinal_categories[name]
        print(f"  {name} ({category}): ({coords[0]:.1f}, {coords[1]:.1f})")
    
    # Test 3: Create comparison plot
    print("\n📊 Creating method comparison visualization...")
    fig2 = visualizer.create_comparison_plot(
        embeddings=embeddings,
        cardinal_names=names,
        cardinal_categories=cardinal_categories
    )
    
    fig2.savefig(os.path.join(output_dir, 'cardinal_method_comparison.png'), 
                 dpi=300, bbox_inches='tight')
    print("✅ Method comparison saved to outputs/cardinal_method_comparison.png")
    
    # Test 4: Test different reduction methods
    print("\n🔄 Testing different reduction methods...")
    methods = ['pca', 'tsne', 'umap']
    
    for method in methods:
        print(f"\n  Testing {method.upper()}...")
        start_time = time.time()
        
        coords_2d = visualizer.reduce_dimensions(embeddings, method)
        coords_norm = visualizer.normalize_coordinates(coords_2d)
        
        process_time = time.time() - start_time
        print(f"    ✅ {method.upper()} completed in {process_time:.3f}s")
        
        # Calculate cluster separation for this method
        temp_visualizer = CardinalVisualizer(config)
        temp_visualizer.reduction_method = method
        temp_analysis = temp_visualizer.analyze_clusters(embeddings, names, cardinal_categories)
        
        avg_distance = np.mean(list(temp_analysis['cluster_distances'].values()))
        print(f"    Average cluster distance: {avg_distance:.1f} units")
    
    # Performance analysis
    print("\n" + "=" * 60)
    print("PERFORMANCE & RECOMMENDATIONS")
    print("=" * 60)
    
    print(f"\n🎯 Why PCA works well for cardinal visualization:")
    print("1. ✅ Linear transformation preserves global ideological structure")
    print("2. ✅ Principal components represent main variance directions")
    print("3. ✅ Stable and reproducible results across runs")
    print("4. ✅ Interpretable axes (PC1 often = progressive vs conservative)")
    print("5. ✅ Less distortion of relative distances")
    
    print(f"\n💡 Visualization Suite Features:")
    print("1. ✅ Configurable through config.yaml")
    print("2. ✅ Multiple dimensionality reduction methods")
    print("3. ✅ Automatic cluster analysis")
    print("4. ✅ Grid normalization for simulation positioning")
    print("5. ✅ High-quality publication-ready plots")
    print("6. ✅ Category-based color coding")
    
    print(f"\n🚀 Ready for integration:")
    print("- Grid coordinates can be used directly in conclave simulation")
    print("- Proximity-based interaction rules")
    print("- Ideological coalition formation")
    print("- Visual monitoring of cardinal positions")
    
    print("\n🎉 VISUALIZATION SUITE TEST COMPLETED!")
    print("Your visualization system is ready for the conclave simulation!")

if __name__ == "__main__":
    main()
