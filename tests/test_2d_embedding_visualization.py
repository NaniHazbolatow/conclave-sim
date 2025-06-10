#!/usr/bimport time
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sentence_transformers import SentenceTransformer
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
import umap
import yaml
import os
import torch
"""
Test script for converting high-dimensional embeddings to 2D grid positions.

This demonstrates various dimensionality reduction techniques to place
cardinals on a 2D spatial grid based on their ideological similarities.
"""

import time
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sentence_transformers import SentenceTransformer
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
import umap
import yaml
import os

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
    print("🗺️  CARDINAL 2D POSITIONING TEST")
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
    
    # Simplified cardinal profiles: 2 hyper-conservatives, 2 progressives, 1 balanced
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
    
    # Generate embeddings
    print(f"\nGenerating embeddings for {len(cardinal_profiles)} cardinals...")
    names = list(cardinal_profiles.keys())
    descriptions = list(cardinal_profiles.values())
    embeddings = model.encode(descriptions)
    
    print(f"✅ Original embedding dimension: {embeddings.shape[1]}")
    
    # Test different dimensionality reduction techniques
    techniques = {
        'PCA': lambda x: PCA(n_components=2, random_state=42).fit_transform(x),
        'TSNE': lambda x: TSNE(n_components=2, random_state=42, perplexity=min(5, len(x)-1)).fit_transform(x),
        'UMAP': lambda x: umap.UMAP(n_components=2, random_state=42, n_neighbors=min(5, len(x)-1)).fit_transform(x)
    }
    
    # Create visualization
    fig, axes = plt.subplots(2, 2, figsize=(15, 12))
    axes = axes.flatten()
    
    print("\n🔄 Applying dimensionality reduction techniques...")
    
    for idx, (technique_name, technique_func) in enumerate(techniques.items()):
        print(f"  Processing {technique_name}...")
        start_time = time.time()
        
        # Apply dimensionality reduction
        coords_2d = technique_func(embeddings)
        
        # Normalize coordinates to grid (0-100 range for easy positioning)
        x_coords = coords_2d[:, 0]
        y_coords = coords_2d[:, 1]
        
        # Normalize to 0-100 range
        x_norm = ((x_coords - x_coords.min()) / (x_coords.max() - x_coords.min())) * 100
        y_norm = ((y_coords - y_coords.min()) / (y_coords.max() - y_coords.min())) * 100
        
        process_time = time.time() - start_time
        print(f"    ✅ {technique_name} completed in {process_time:.3f}s")
        
        # Plot
        ax = axes[idx]
        scatter = ax.scatter(x_norm, y_norm, c=range(len(names)), cmap='tab20', s=100, alpha=0.7)
        
        # Add labels
        for i, name in enumerate(names):
            ax.annotate(name, (x_norm[i], y_norm[i]), xytext=(5, 5), 
                       textcoords='offset points', fontsize=8, alpha=0.8)
        
        ax.set_title(f'{technique_name} - Cardinal Positioning', fontsize=12, fontweight='bold')
        ax.set_xlabel('Ideological Dimension 1 (0-100)')
        ax.set_ylabel('Ideological Dimension 2 (0-100)')
        ax.grid(True, alpha=0.3)
        ax.set_xlim(-5, 105)
        ax.set_ylim(-5, 105)
        
        # Print coordinates for this technique
        print(f"\n    📍 {technique_name} Grid Positions (0-100 scale):")
        for i, name in enumerate(names):
            print(f"      {name}: ({x_norm[i]:.1f}, {y_norm[i]:.1f})")
    
    # Remove empty subplot
    axes[3].remove()
    
    plt.tight_layout()
    plt.suptitle('Cardinal Ideological Positioning in 2D Space', fontsize=16, fontweight='bold', y=0.98)
    plt.show()
    
    # Analyze ideological clusters
    print("\n" + "=" * 60)
    print("IDEOLOGICAL CLUSTER ANALYSIS")
    print("=" * 60)
    
    # Use TSNE results for analysis (often gives good semantic clustering)
    tsne_coords = TSNE(n_components=2, random_state=42, perplexity=min(5, len(embeddings)-1)).fit_transform(embeddings)
    x_norm = ((tsne_coords[:, 0] - tsne_coords[:, 0].min()) / (tsne_coords[:, 0].max() - tsne_coords[:, 0].min())) * 100
    y_norm = ((tsne_coords[:, 1] - tsne_coords[:, 1].min()) / (tsne_coords[:, 1].max() - tsne_coords[:, 1].min())) * 100
    
    # Identify progressive vs conservative clusters
    progressive_cardinals = ['Tagle', 'Cupich']
    conservative_cardinals = ['Sarah', 'Burke']
    balanced_cardinals = ['Parolin']
    
    print("\n🔍 Expected vs Actual Clustering:")
    
    # Calculate average positions
    prog_positions = [(x_norm[i], y_norm[i]) for i, name in enumerate(names) if name in progressive_cardinals]
    cons_positions = [(x_norm[i], y_norm[i]) for i, name in enumerate(names) if name in conservative_cardinals]
    balanced_positions = [(x_norm[i], y_norm[i]) for i, name in enumerate(names) if name in balanced_cardinals]
    
    if prog_positions and cons_positions:
        prog_center = (np.mean([p[0] for p in prog_positions]), np.mean([p[1] for p in prog_positions]))
        cons_center = (np.mean([p[0] for p in cons_positions]), np.mean([p[1] for p in cons_positions]))
        
        distance = np.sqrt((prog_center[0] - cons_center[0])**2 + (prog_center[1] - cons_center[1])**2)
        
        print(f"\nProgressive cluster center (2 cardinals): ({prog_center[0]:.1f}, {prog_center[1]:.1f})")
        print(f"Conservative cluster center (2 cardinals): ({cons_center[0]:.1f}, {cons_center[1]:.1f})")
        
        if balanced_positions:
            balanced_pos = balanced_positions[0]
            print(f"Balanced cardinal position: ({balanced_pos[0]:.1f}, {balanced_pos[1]:.1f})")
            
            # Calculate distances from balanced to each cluster
            dist_to_prog = np.sqrt((balanced_pos[0] - prog_center[0])**2 + (balanced_pos[1] - prog_center[1])**2)
            dist_to_cons = np.sqrt((balanced_pos[0] - cons_center[0])**2 + (balanced_pos[1] - cons_center[1])**2)
            
            print(f"Distance from balanced to progressive cluster: {dist_to_prog:.1f}")
            print(f"Distance from balanced to conservative cluster: {dist_to_cons:.1f}")
        
        print(f"Ideological separation distance: {distance:.1f} units")
        
        if distance > 30:
            print("✅ Good ideological separation detected!")
        else:
            print("⚠️  Limited ideological separation - consider different reduction technique")
    
    # Grid positioning recommendations
    print("\n" + "=" * 60)
    print("GRID POSITIONING RECOMMENDATIONS")
    print("=" * 60)
    
    print("\n💡 For your conclave simulation:")
    print("1. Use TSNE or UMAP coordinates as base positions")
    print("2. Cardinals closer in 2D space will interact more frequently")
    print("3. Consider adding small random jitter to avoid overlaps")
    print("4. You can use these coordinates for:")
    print("   - Visual representation of ideological landscape")
    print("   - Proximity-based interaction rules")
    print("   - Coalition formation algorithms")
    print("   - Influence propagation models")
    
    print("\n🎯 POSITIONING TEST COMPLETED!")
    print("Your embeddings can be successfully converted to 2D grid positions!")

if __name__ == "__main__":
    main()
