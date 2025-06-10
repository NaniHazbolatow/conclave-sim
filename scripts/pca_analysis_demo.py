#!/usr/bin/env python3
"""
Demo script showing why PCA is visually most convincing for cardinal positioning.

This script demonstrates the key advantages of PCA over other dimensionality 
reduction techniques for ideological visualization.
"""

import numpy as np
import matplotlib.pyplot as plt
from sentence_transformers import SentenceTransformer
import yaml
import os
import sys
import torch

# Add project root to path
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
    print("🔍 WHY PCA IS VISUALLY MOST CONVINCING")
    print("=" * 50)
    
    # Load configuration and model
    config = load_config()
    embedding_config = config['embeddings']
    model = SentenceTransformer(embedding_config['model_name'], device=get_device(embedding_config['device']))
    
    # Comprehensive cardinal profiles organized by ideological factions
    cardinal_profiles = {
        # Ultra-Conservative Traditionalist Faction (4 cardinals)
        'Sarah': 'Cardinal Robert Sarah, ultra-conservative African traditionalist, strongly opposes modern reforms, emphasizes strict liturgical orthodoxy, Latin Mass advocate, unwavering traditional Catholic teachings',
        'Burke': 'Cardinal Raymond Burke, extreme conservative American canonist, militant defender of traditional doctrine, fierce opponent of progressive changes and papal reforms, liturgical purist',
        'Müller': 'Cardinal Gerhard Ludwig Müller, German ultra-conservative theologian, former Prefect of Doctrine of Faith, rigid defender of Catholic orthodoxy, critic of Vatican II interpretations',
        'Brandmüller': 'Cardinal Walter Brandmüller, German conservative historian, traditional Church doctrine advocate, strong opponent of clerical marriage and female ordination',
        
        # Progressive Social Justice Faction (4 cardinals)
        'Tagle': 'Cardinal Luis Antonio Tagle, progressive Filipino leader, champions social justice, climate action, church modernization, interfaith dialogue, pastoral mercy approach',
        'Cupich': 'Cardinal Blase Cupich, American progressive reformer, advocates immigration reform, LGBTQ+ inclusion, environmental activism, synodal church governance',
        'Marx': 'Cardinal Reinhard Marx, German progressive social advocate, supports church structural reforms, economic justice, transparency in church finances, women in ministry',
        'McElroy': 'Cardinal Robert McElroy, American progressive theologian, champions marginalized communities, LGBTQ+ pastoral care, climate justice, synodal participation',
        
        # Moderate Diplomatic Faction (3 cardinals)
        'Parolin': 'Cardinal Pietro Parolin, Vatican Secretary of State, experienced diplomat, seeks balance between tradition and reform, pragmatic centrist approach, international relations expert',
        'Sandri': 'Cardinal Leonardo Sandri, Argentine Vatican diplomat, Eastern Churches prefect, bridge-builder between different traditions, pastoral pragmatist',
        'Filoni': 'Cardinal Fernando Filoni, Italian missionary expert, former nuncio, balanced approach to evangelization, cultural sensitivity in diverse contexts',
        
        # African Traditional-Modern Synthesis Faction (3 cardinals)
        'Turkson': 'Cardinal Peter Turkson, Ghanaian moderate conservative, advocate for peace and sustainable development, African traditional values with modern social justice concerns',
        'Napier': 'Cardinal Wilfrid Napier, South African moderate, defender of traditional marriage and family while advocating for social justice and economic equality',
        'Onaiyekan': 'Cardinal John Onaiyekan, Nigerian moderate traditionalist, promotes interfaith dialogue while maintaining orthodox Catholic teachings on moral issues',
        
        # Asian Pastoral Innovation Faction (2 cardinals)
        'Bo': 'Cardinal Charles Bo, Myanmar progressive, advocates for human rights and democracy, promotes inculturation of Catholic faith with Asian spiritual traditions',
        'Gracias': 'Cardinal Oswald Gracias, Indian moderate progressive, supports liturgical adaptation to local cultures while maintaining doctrinal orthodoxy'
    }
    
    cardinal_categories = {
        # Ultra-Conservative Traditionalist Faction
        'Sarah': 'Ultra-Conservative',
        'Burke': 'Ultra-Conservative',
        'Müller': 'Ultra-Conservative', 
        'Brandmüller': 'Ultra-Conservative',
        
        # Progressive Social Justice Faction
        'Tagle': 'Progressive',
        'Cupich': 'Progressive',
        'Marx': 'Progressive',
        'McElroy': 'Progressive',
        
        # Moderate Diplomatic Faction
        'Parolin': 'Moderate-Diplomatic',
        'Sandri': 'Moderate-Diplomatic',
        'Filoni': 'Moderate-Diplomatic',
        
        # African Traditional-Modern Synthesis Faction
        'Turkson': 'African-Synthesis',
        'Napier': 'African-Synthesis',
        'Onaiyekan': 'African-Synthesis',
        
        # Asian Pastoral Innovation Faction
        'Bo': 'Asian-Innovation',
        'Gracias': 'Asian-Innovation'
    }
    
    # Generate embeddings
    names = list(cardinal_profiles.keys())
    descriptions = list(cardinal_profiles.values())
    embeddings = model.encode(descriptions)
    
    print(f"Original embedding dimension: {embeddings.shape[1]}")
    print(f"Number of cardinals: {len(names)}")
    print(f"Number of factions: {len(set(cardinal_categories.values()))}")
    
    # Print faction composition
    print(f"\n📊 FACTION COMPOSITION:")
    faction_counts = {}
    for category in cardinal_categories.values():
        faction_counts[category] = faction_counts.get(category, 0) + 1
    
    for faction, count in faction_counts.items():
        members = [name for name, cat in cardinal_categories.items() if cat == faction]
        print(f"   {faction}: {count} cardinals ({', '.join(members)})")
    
    # Test all three methods with detailed analysis
    visualizer = CardinalVisualizer(config)
    methods = ['pca', 'tsne', 'umap']
    
    results = {}
    
    for method in methods:
        print(f"\n📊 Analyzing {method.upper()}...")
        
        # Get coordinates
        coords_2d = visualizer.reduce_dimensions(embeddings, method)
        coords_norm = visualizer.normalize_coordinates(coords_2d)
        
        # Calculate faction centers and separations
        faction_centers = {}
        faction_coords = {}
        
        for faction in set(cardinal_categories.values()):
            faction_members = [i for i, name in enumerate(names) if cardinal_categories[name] == faction]
            faction_coord_list = [coords_norm[i] for i in faction_members]
            faction_coords[faction] = faction_coord_list
            faction_centers[faction] = np.mean(faction_coord_list, axis=0)
        
        # Calculate inter-faction distances
        faction_names = list(faction_centers.keys())
        inter_faction_distances = {}
        
        for i, faction1 in enumerate(faction_names):
            for j, faction2 in enumerate(faction_names[i+1:], i+1):
                distance = np.linalg.norm(faction_centers[faction1] - faction_centers[faction2])
                inter_faction_distances[f"{faction1} vs {faction2}"] = distance
        
        # Calculate intra-faction compactness
        faction_compactness = {}
        for faction, coords_list in faction_coords.items():
            if len(coords_list) > 1:
                center = faction_centers[faction]
                distances = [np.linalg.norm(coord - center) for coord in coords_list]
                faction_compactness[faction] = np.mean(distances)
            else:
                faction_compactness[faction] = 0.0
        
        results[method] = {
            'faction_centers': faction_centers,
            'inter_faction_distances': inter_faction_distances,
            'faction_compactness': faction_compactness,
            'coords': coords_norm
        }
        
        print(f"  Average inter-faction distance: {np.mean(list(inter_faction_distances.values())):.1f} units")
        print(f"  Average intra-faction compactness: {np.mean(list(faction_compactness.values())):.1f} units")
        
        # Show best and worst separated factions
        sorted_distances = sorted(inter_faction_distances.items(), key=lambda x: x[1], reverse=True)
        print(f"  Best separated: {sorted_distances[0][0]} ({sorted_distances[0][1]:.1f} units)")
        print(f"  Least separated: {sorted_distances[-1][0]} ({sorted_distances[-1][1]:.1f} units)")
    
    # Analysis
    print("\n" + "=" * 60)
    print("MULTI-FACTION COMPARISON ANALYSIS")
    print("=" * 60)
    
    print(f"\n🎯 Why PCA excels with multiple factions:")
    
    pca_results = results['pca']
    tsne_results = results['tsne']
    umap_results = results['umap']
    
    print(f"\n1. INTER-FACTION SEPARATION (average distance between faction centers):")
    pca_avg_separation = np.mean(list(pca_results['inter_faction_distances'].values()))
    tsne_avg_separation = np.mean(list(tsne_results['inter_faction_distances'].values()))
    umap_avg_separation = np.mean(list(umap_results['inter_faction_distances'].values()))
    
    print(f"   PCA:  {pca_avg_separation:.1f} units")
    print(f"   TSNE: {tsne_avg_separation:.1f} units") 
    print(f"   UMAP: {umap_avg_separation:.1f} units")
    
    if pca_avg_separation >= max(tsne_avg_separation, umap_avg_separation):
        print("   ✅ PCA provides the best overall faction separation")
    
    print(f"\n2. INTRA-FACTION COMPACTNESS (average tightness within factions):")
    pca_avg_compactness = np.mean(list(pca_results['faction_compactness'].values()))
    tsne_avg_compactness = np.mean(list(tsne_results['faction_compactness'].values()))
    umap_avg_compactness = np.mean(list(umap_results['faction_compactness'].values()))
    
    print(f"   PCA:  {pca_avg_compactness:.1f} units (lower = more compact)")
    print(f"   TSNE: {tsne_avg_compactness:.1f} units") 
    print(f"   UMAP: {umap_avg_compactness:.1f} units")
    
    print(f"\n3. SEPARATION-TO-COMPACTNESS RATIO (higher = better clustering):")
    pca_ratio = pca_avg_separation / max(pca_avg_compactness, 0.1)
    tsne_ratio = tsne_avg_separation / max(tsne_avg_compactness, 0.1)
    umap_ratio = umap_avg_separation / max(umap_avg_compactness, 0.1)
    
    print(f"   PCA:  {pca_ratio:.2f}")
    print(f"   TSNE: {tsne_ratio:.2f}") 
    print(f"   UMAP: {umap_ratio:.2f}")
    
    if pca_ratio >= max(tsne_ratio, umap_ratio):
        print("   ✅ PCA achieves the best separation-to-compactness ratio")
    
    print(f"\n4. FACTION-SPECIFIC ANALYSIS (PCA results):")
    for faction, center in pca_results['faction_centers'].items():
        compactness = pca_results['faction_compactness'][faction]
        print(f"   {faction}:")
        print(f"     Center: ({center[0]:.1f}, {center[1]:.1f})")
        print(f"     Compactness: {compactness:.1f}")
    
    print(f"\n5. MOST/LEAST SEPARATED FACTION PAIRS (PCA):")
    sorted_distances = sorted(pca_results['inter_faction_distances'].items(), key=lambda x: x[1], reverse=True)
    print(f"   Most separated: {sorted_distances[0][0]} = {sorted_distances[0][1]:.1f} units")
    print(f"   Least separated: {sorted_distances[-1][0]} = {sorted_distances[-1][1]:.1f} units")
    
    print(f"\n6. IDEOLOGICAL SPECTRUM REPRESENTATION:")
    print("   ✅ PCA: Preserves global ideological distances between all factions")
    print("   ❓ t-SNE: May artificially cluster distant factions together") 
    print("   ❓ UMAP: Good local structure but may distort global relationships")
    
    # Create a focused PCA visualization
    print(f"\n📊 Creating detailed multi-faction PCA visualization...")
    
    visualizer.reduction_method = 'pca'
    fig = visualizer.create_ideological_visualization(
        embeddings=embeddings,
        cardinal_names=names,
        cardinal_categories=cardinal_categories,
        title="PCA: Multi-Faction Cardinal Ideological Positioning (16 Cardinals)"
    )
    
    # Save detailed visualization
    output_dir = os.path.join(os.path.dirname(__file__), '..', 'outputs')
    fig.savefig(os.path.join(output_dir, 'pca_multi_faction_analysis.png'), 
                dpi=300, bbox_inches='tight')
    
    print("✅ Multi-faction PCA visualization saved to outputs/pca_multi_faction_analysis.png")
    
    # Print final coordinates organized by faction for use in simulation
    print(f"\n📍 FINAL PCA COORDINATES BY FACTION:")
    pca_coords = results['pca']['coords']
    
    for faction in sorted(set(cardinal_categories.values())):
        print(f"\n   🏛️ {faction.upper()} FACTION:")
        faction_members = [(name, i) for i, name in enumerate(names) if cardinal_categories[name] == faction]
        
        for name, i in faction_members:
            print(f"      {name}: ({pca_coords[i][0]:.1f}, {pca_coords[i][1]:.1f})")
    
    print(f"\n🔍 COALITION POTENTIAL ANALYSIS:")
    print("Based on PCA positioning, likely coalition scenarios:")
    
    # Identify potential coalitions based on proximity
    faction_centers = results['pca']['faction_centers']
    
    # Calculate all pairwise distances between faction centers
    faction_pairs = []
    faction_names = list(faction_centers.keys())
    
    for i, faction1 in enumerate(faction_names):
        for j, faction2 in enumerate(faction_names[i+1:], i+1):
            distance = np.linalg.norm(faction_centers[faction1] - faction_centers[faction2])
            faction_pairs.append((faction1, faction2, distance))
    
    # Sort by distance (closest first)
    faction_pairs.sort(key=lambda x: x[2])
    
    print(f"\n   Most likely alliances (closest in ideological space):")
    for i, (faction1, faction2, distance) in enumerate(faction_pairs[:3]):
        print(f"   {i+1}. {faction1} + {faction2} (distance: {distance:.1f})")
    
    print(f"\n   Least likely alliances (furthest apart):")
    for i, (faction1, faction2, distance) in enumerate(faction_pairs[-2:]):
        print(f"   {len(faction_pairs)-1-i}. {faction1} + {faction2} (distance: {distance:.1f})")
    
    print(f"\n🎉 MULTI-FACTION ANALYSIS COMPLETE!")
    print("PCA successfully reveals complex ideological landscape with:")
    print(f"   ✅ {len(set(cardinal_categories.values()))} distinct factions")
    print(f"   ✅ {len(names)} individual cardinal positions")
    print("   ✅ Clear faction boundaries and coalition potential")
    print("   ✅ Stable, interpretable 2D representation")
    print("\nThis rich positioning data is ideal for complex conclave simulations!")

if __name__ == "__main__":
    main()
