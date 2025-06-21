"""
Generate Bocconi Network using Cardinal IDs

This script recreates the bocconi graph using the Cardinal_ID system
from the adjusted data files instead of the surname-based approach.
This ensures compatibility with the utility_utils grouping system.
"""

import os
import sys
import logging
import random
import pickle
from typing import Dict, Set, List
from collections import defaultdict
from itertools import combinations

import pandas as pd
import networkx as nx
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import colormaps
from matplotlib.colors import Normalize
from matplotlib.patches import Patch

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def load_cardinal_master_data(file_path: str) -> Dict[int, Dict]:
    """Load cardinal master data and return mapping from Cardinal_ID to info."""
    df = pd.read_csv(file_path)
    cardinal_info = {}
    
    for _, row in df.iterrows():
        cardinal_info[row['Cardinal_ID']] = {
            'Name': row['Name'],
            'Background': row['Background'],
            'Internal_Persona': row['Internal_Persona'],
            'Public_Profile': row['Public_Profile'],
            'Profile_Blurb': row['Profile_Blurb'],
            'Persona_Tag': row['Persona_Tag']
        }
    
    logger.info(f"Loaded {len(cardinal_info)} cardinals from master data")
    return cardinal_info

def load_consecration_network(file_path: str) -> nx.Graph:
    """Load episcopal consecration network (if available).
    
    Note: This network still uses surnames as node identifiers.
    We'll skip it for now to maintain Cardinal_ID consistency.
    """
    try:
        consecration_df = pd.read_excel(file_path, index_col=0)
        # Skip this network for now since it uses surnames, not Cardinal_IDs
        logger.info("Skipping consecration network (uses surnames, not Cardinal_IDs)")
        return nx.Graph()
    except FileNotFoundError:
        logger.warning("Consecration network file not found, creating empty graph")
        return nx.Graph()

def load_comembership_network(file_path: str) -> nx.Graph:
    """Load Vatican co-membership network using Cardinal_ID."""
    try:
        official_comemberships = pd.read_excel(file_path)
        logger.info(f"Loaded {len(official_comemberships)} co-membership records")
        
        # Create co-membership network using Cardinal_IDs
        G_comembership = nx.Graph()
        
        # Add all cardinals as nodes using their Cardinal_IDs
        all_cardinal_ids = set(official_comemberships['Cardinal_ID'].unique())
        G_comembership.add_nodes_from(all_cardinal_ids)
        logger.info(f"Added {len(all_cardinal_ids)} cardinal nodes")
        
        # Create edges based on shared memberships
        weights = defaultdict(int)
        groups = official_comemberships.groupby('Membership')['Cardinal_ID'].apply(list)
        
        for membership, members in groups.items():
            if len(members) > 1:
                # Create edges between all pairs in the same group
                for pair in combinations(sorted(members), 2):
                    weights[pair] += 1
        
        # Add edges with weights to the graph
        edge_count = 0
        for (cardinal1, cardinal2), weight in weights.items():
            G_comembership.add_edge(cardinal1, cardinal2, weight=weight)
            edge_count += 1
        
        logger.info(f"Created {edge_count} weighted edges from co-memberships")
        return G_comembership
        
    except FileNotFoundError:
        logger.warning("Formal relationships file not found, creating empty graph")
        return nx.Graph()

def combine_networks(G1: nx.Graph, G2: nx.Graph, weight_attr='weight', default_weight=1.0) -> nx.Graph:
    """Combine two weighted graphs by summing edge weights."""
    result = nx.Graph()
    
    # Add all nodes from both graphs
    result.add_nodes_from(G1.nodes(data=True))
    result.add_nodes_from(G2.nodes(data=True))
    
    # Combine node attributes
    for node in result.nodes():
        result.nodes[node].update(G1.nodes.get(node, {}))
        result.nodes[node].update(G2.nodes.get(node, {}))
    
    def add_or_update_edge(u, v, data, target):
        weight = data.get(weight_attr, default_weight)
        other_attrs = {k: v for k, v in data.items() if k != weight_attr}
        
        if target.has_edge(u, v):
            current_weight = target[u][v].get(weight_attr, default_weight)
            target[u][v][weight_attr] = current_weight + weight
            target[u][v].update(other_attrs)
        else:
            target.add_edge(u, v, **{weight_attr: weight}, **other_attrs)
    
    # Add edges from both graphs
    for u, v, data in G1.edges(data=True):
        add_or_update_edge(u, v, data, result)
    for u, v, data in G2.edges(data=True):
        add_or_update_edge(u, v, data, result)
    
    return result

def load_node_attributes(G: nx.Graph, file_path: str, cardinal_info: Dict[int, Dict]) -> nx.Graph:
    """Load node attributes from Excel and attach to NetworkX graph using Cardinal_ID."""
    try:
        node_df = pd.read_excel(file_path)
        
        # Set Cardinal_ID as index for easier mapping
        node_df = node_df.set_index('Cardinal_ID')
        
        # Add attributes to network nodes
        for cardinal_id in G.nodes():
            if cardinal_id in node_df.index:
                # Add all attributes from the node_df
                for attr, value in node_df.loc[cardinal_id].items():
                    G.nodes[cardinal_id][attr] = value
                
                # Add name and other info from master data if available
                if cardinal_id in cardinal_info:
                    G.nodes[cardinal_id]['FullName'] = cardinal_info[cardinal_id]['Name']
                    G.nodes[cardinal_id]['Background'] = cardinal_info[cardinal_id]['Background']
                    
        logger.info(f"Added attributes to {len([n for n in G.nodes() if 'FullName' in G.nodes[n]])} nodes")
        return G
        
    except FileNotFoundError:
        logger.warning("Node info file not found, node attributes will be missing")
        return G

def present_in_conclave(data: Dict) -> bool:
    """Determine whether a cardinal is present in conclave."""
    lean = str(data.get('Lean', '')).strip().lower()
    age = data.get('Age', 0)
    return lean != 'non-voting' and age < 80

def add_ideology_scores(G: nx.Graph) -> nx.Graph:
    """Add random ideology scores to nodes that don't have them."""
    for node in G.nodes():
        if 'IdeologyScore' not in G.nodes[node]:
            # Use political lean if available to influence ideology score
            lean = str(G.nodes[node].get('Lean', '')).strip().lower()
            lean_score_map = {
                'liberal': -0.8 + random.uniform(0, 0.4),
                'soft liberal': -0.4 + random.uniform(0, 0.4),
                'moderate': -0.2 + random.uniform(0, 0.4),
                'soft conservative': 0.2 + random.uniform(0, 0.4),
                'conservative': 0.6 + random.uniform(0, 0.4)
            }
            
            if lean in lean_score_map:
                G.nodes[node]['IdeologyScore'] = lean_score_map[lean]
            else:
                G.nodes[node]['IdeologyScore'] = random.uniform(-1, 1)
    
    return G

def calculate_centrality(G: nx.Graph) -> Dict[int, float]:
    """Calculate eigenvector centrality for the graph."""
    if len(G.edges) > 0:
        try:
            centrality = nx.eigenvector_centrality(G, weight='weight', max_iter=1000)
        except nx.PowerIterationFailedConvergence:
            logger.warning("Eigenvector centrality failed to converge, using degree centrality")
            centrality = nx.degree_centrality(G)
        except nx.NetworkXError:
            logger.warning("Network error in centrality calculation, using degree centrality")
            centrality = nx.degree_centrality(G)
    else:
        centrality = {node: 0.0 for node in G.nodes()}
    
    return centrality

def create_visualization(G: nx.Graph, output_path: str = "cardinal_network_updated.png"):
    """Create and save a visualization of the cardinal network."""
    # Define political lean → numerical score for coloring
    lean_score_map = {
        'liberal': -1,
        'soft liberal': -0.5,
        'moderate': 0,
        'soft conservative': 0.5,
        'conservative': 1
    }
    
    # Filter graph to only include papal candidates (electors)
    G_electors = G.subgraph([
        n for n, d in G.nodes(data=True) if present_in_conclave(d)
    ]).copy()
    
    logger.info(f"Visualization includes {len(G_electors.nodes)} electors out of {len(G.nodes)} total cardinals")
    
    # Extract and score political lean for each node
    partition = {
        n: str(d.get('Lean', 'Unknown')).strip().lower()
        for n, d in G.nodes(data=True)
    }
    default_color = (0.9, 0.85, 0.75, 1.0)
    
    node_scores = [
        lean_score_map.get(partition.get(n, ''), None)
        for n in G.nodes()
    ]
    
    # Color map from lean score
    cmap = colormaps.get_cmap('coolwarm')
    norm = Normalize(vmin=-1, vmax=1)
    node_colors = [
        cmap(norm(score)) if score is not None else default_color
        for score in node_scores
    ]
    
    # Get centrality for node sizing
    centrality = {n: G.nodes[n].get('EigenvectorCentrality', 0.1) for n in G.nodes()}
    
    # Node layout and size
    pos = nx.spring_layout(G, seed=42, k=3, scale=4)
    node_sizes = [centrality[n] * 20000 for n in G.nodes()]
    
    # Edge transparency normalization
    min_c, max_c = min(centrality.values()), max(centrality.values())
    normalize = lambda c: (c - min_c) / (max_c - min_c) if max_c > min_c else 0.5
    
    # Plot setup
    plt.figure(figsize=(16, 14))
    nx.draw_networkx_nodes(
        G,
        pos,
        node_color=node_colors,
        node_size=node_sizes,
        alpha=0.85,
        linewidths=0.3,
        edgecolors='black'
    )
    
    # Draw only weighted edges (weight > 1)
    filtered_edges = [
        (u, v) for u, v in G.edges()
        if G[u][v].get('weight', 1) > 1
    ]
    
    # Draw edges with adaptive transparency
    for u, v in filtered_edges:
        alpha = 0.01 + normalize((centrality[u] + centrality[v]) / 2) * 0.15
        width = G[u][v].get('weight', 1)
        nx.draw_networkx_edges(G, pos, edgelist=[(u, v)], width=width, alpha=alpha)
    
    # Label top 10 most central candidates
    top_labels = {
        n: G.nodes[n].get('FullName', f"Cardinal_{n}")[:20]  # Truncate long names
        for n, _ in sorted(centrality.items(), key=lambda x: x[1], reverse=True)[:10]
    }
    nx.draw_networkx_labels(G, pos, labels=top_labels, font_size=10, font_weight='bold')
    
    # Legend for political lean
    legend_handles = [
        Patch(facecolor=cmap(norm(score)), edgecolor='k', label=label.title())
        for label, score in lean_score_map.items()
    ] + [Patch(facecolor=default_color, edgecolor='k', label='Non-voting')]
    
    plt.legend(
        handles=legend_handles,
        title="Political Lean",
        loc='lower left',
        bbox_to_anchor=(1, 0),
        fontsize=10,
        title_fontsize=12,
        frameon=True
    )
    
    # Calculate statistics for electors
    elector_scores = [
        lean_score_map.get(str(d.get('Lean', '')).strip().lower())
        for n, d in G_electors.nodes(data=True)
        if str(d.get('Lean', '')).strip().lower() in lean_score_map
    ]
    
    if elector_scores:
        f_score = sum(elector_scores) / len(elector_scores)
        logger.info(f"F-score (mean lean among electors): {f_score:.3f}")
    
    # Count by political lean
    for lean_type in lean_score_map.keys():
        count = sum(
            1 for d in G.nodes.values()
            if str(d.get('Lean', '')).strip().lower() == lean_type
        )
        logger.info(f"Number of {lean_type}s in the college: {count}")
    
    # Final polish and export
    plt.title("Cardinal Network — Grouped by Political Lean (Cardinal ID Based)", fontsize=18)
    plt.axis('off')
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    logger.info(f"Visualization saved to {output_path}")
    plt.close()

def main():
    """Main function to generate the updated bocconi network."""
    # Set paths
    base_path = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    data_path = os.path.join(base_path, 'data')
    network_path = os.path.join(data_path, 'network')
    output_path = os.path.join(data_path, 'network')  # Changed from conclave/network/archive to data/network
    
    logger.info(f"Working with data from: {data_path}")
    logger.info(f"Output will be saved to: {output_path}")
    
    # Load cardinal master data
    cardinal_info = load_cardinal_master_data(os.path.join(data_path, 'cardinals_master_data.csv'))
    
    # Load networks
    G_consecration = load_consecration_network(os.path.join(network_path, 'adjacency_matrix_consecrator.xlsx'))
    G_comembership = load_comembership_network(os.path.join(network_path, 'formal.xlsx'))
    
    # Combine networks
    logger.info("Combining networks...")
    G_multiplex = combine_networks(G_consecration, G_comembership, weight_attr='weight')
    logger.info(f"Combined network has {len(G_multiplex.nodes)} nodes and {len(G_multiplex.edges)} edges")
    
    # Load node attributes
    G_multiplex = load_node_attributes(G_multiplex, os.path.join(network_path, 'node_info.xlsx'), cardinal_info)
    
    # Add ideology scores
    G_multiplex = add_ideology_scores(G_multiplex)
    
    # Calculate centrality
    logger.info("Calculating centrality...")
    centrality = calculate_centrality(G_multiplex)
    nx.set_node_attributes(G_multiplex, centrality, 'EigenvectorCentrality')
    
    # Save the graph
    os.makedirs(output_path, exist_ok=True)
    graph_file = os.path.join(output_path, 'bocconi_graph.gpickle')
    
    with open(graph_file, 'wb') as f:
        pickle.dump(G_multiplex, f, pickle.HIGHEST_PROTOCOL)
    logger.info(f"Graph saved to {graph_file}")
    
    # Create visualization
    create_visualization(G_multiplex, os.path.join(output_path, "cardinal_network_updated.png"))
    
    # Print summary statistics
    logger.info(f"Final network statistics:")
    logger.info(f"  Total nodes: {len(G_multiplex.nodes)}")
    logger.info(f"  Total edges: {len(G_multiplex.edges)}")
    logger.info(f"  Nodes with attributes: {len([n for n in G_multiplex.nodes() if 'FullName' in G_multiplex.nodes[n]])}")
    logger.info(f"  Average degree: {sum(dict(G_multiplex.degree()).values()) / len(G_multiplex.nodes):.2f}")
    
    # Show top 5 most central cardinals
    top_central = sorted(centrality.items(), key=lambda x: x[1], reverse=True)[:5]
    logger.info("Top 5 most central cardinals:")
    for cardinal_id, cent_score in top_central:
        name = G_multiplex.nodes[cardinal_id].get('FullName', f'Cardinal_{cardinal_id}')
        logger.info(f"  {name} (ID: {cardinal_id}): {cent_score:.3f}")

if __name__ == "__main__":
    # Set random seed for reproducibility
    random.seed(42)
    np.random.seed(42)
    
    main()
