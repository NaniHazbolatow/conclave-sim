"""
Utility-Based Non-Overlapping Group Formation for Conclave Simulation

This module creates non-overlapping groups where high-centrality cardinals form groups around themselves.
Each cardinal is assigned to exactly one group.

Updated to use Cardinal IDs from enriched network files instead of surname-based matching.
Network nodes are now identified by their Cardinal_ID from the master data.
"""

import random
import logging
from typing import List, Dict, Set, Tuple
import networkx as nx
import pandas as pd
from collections import defaultdict
from itertools import combinations
import os

logger = logging.getLogger(__name__)

# This is a placeholder for the actual network loading logic.
# In a real scenario, you would import the graph from your network module.
def load_network():
    """Loads the cardinal network graph using Cardinal IDs from enriched network files."""
    # Build path to data files
    base_path = os.path.join(os.path.dirname(__file__), '..', '..', 'data')
    network_path = os.path.join(base_path, 'network')
    
    # Load master data for full names (Cardinal_ID is now in network files)
    id_to_full_name = {}
    try:
        master_data = pd.read_csv(os.path.join(base_path, 'cardinals_master_data.csv'))
        
        for _, row in master_data.iterrows():
            id_to_full_name[row['Cardinal_ID']] = row['Name']
        
    except FileNotFoundError:
        pass  # Master data not available, full names will be missing
    
    # Initialize episcopal consecration matrix
    try:
        consecration_df = pd.read_excel(os.path.join(network_path, 'adjacency_matrix_consecrator.xlsx'), index_col=0)
        G_consecration = nx.from_pandas_adjacency(consecration_df, create_using=nx.Graph())
    except FileNotFoundError:
        logger.debug("Consecration network file not found, skipping")
        G_consecration = nx.Graph()

    # Initialize Vatican co-memberships matrix using Cardinal IDs
    try:
        official_comemberships = pd.read_excel(os.path.join(network_path, 'formal.xlsx'))
        
        # Create co-membership network using Cardinal IDs
        G_comembership = nx.Graph()
        
        # Add all cardinals as nodes using their Cardinal IDs
        all_cardinal_ids = set(official_comemberships['Cardinal_ID'].unique())
        G_comembership.add_nodes_from(all_cardinal_ids)
        
        # Create edges based on shared memberships
        weights = defaultdict(int)
        groups = official_comemberships.groupby('Membership')['Cardinal_ID'].apply(list)
        
        for members in groups:
            if len(members) > 1:
                # Create edges between all pairs in the same group
                for pair in combinations(sorted(members), 2):
                    weights[pair] += 1
        
        # Add edges with weights to the graph
        for (cardinal1, cardinal2), weight in weights.items():
            G_comembership.add_edge(cardinal1, cardinal2, weight=weight)
            
    except FileNotFoundError:
        logger.debug("Formal relationships file not found, skipping co-membership network")
        G_comembership = nx.Graph()

    # Combine networks (note: consecration network still uses surnames, so we may need to map it)
    # For now, we'll start with the co-membership network and add consecration later if needed
    G_multiplex = G_comembership.copy()

    # Load node attributes using Cardinal IDs
    try:
        node_df = pd.read_excel(os.path.join(network_path, 'node_info.xlsx'))
        
        # Set Cardinal_ID as index for easier mapping
        node_df = node_df.set_index('Cardinal_ID')
        
        # Add attributes to network nodes
        for cardinal_id in G_multiplex.nodes():
            if cardinal_id in node_df.index:
                # Add all attributes from the node_df
                for attr, value in node_df.loc[cardinal_id].items():
                    G_multiplex.nodes[cardinal_id][attr] = value
                
                # Add full name from master data if available
                if cardinal_id in id_to_full_name:
                    G_multiplex.nodes[cardinal_id]['FullName'] = id_to_full_name[cardinal_id]
                    
    except FileNotFoundError:
        logger.debug("Node info file not found, node attributes will be missing")

    # Calculate Eigenvector Centrality
    if len(G_multiplex.edges) > 0:
        try:
            centrality = nx.eigenvector_centrality(G_multiplex, weight='weight', max_iter=1000)
            nx.set_node_attributes(G_multiplex, centrality, 'EigenvectorCentrality')
        except nx.PowerIterationFailedConvergence:
            centrality = nx.eigenvector_centrality(G_multiplex)
            nx.set_node_attributes(G_multiplex, centrality, 'EigenvectorCentrality')
        except nx.NetworkXError:
            centrality = nx.degree_centrality(G_multiplex)
            nx.set_node_attributes(G_multiplex, centrality, 'EigenvectorCentrality')
    else:
        centrality = {node: 0.0 for node in G_multiplex.nodes()}
        nx.set_node_attributes(G_multiplex, centrality, 'EigenvectorCentrality')

    return G_multiplex

def ensure_ideology_scores(network_graph: nx.Graph):
    """Ensure all nodes have an 'IdeologyScore' attribute."""
    for node in network_graph.nodes:
        if 'IdeologyScore' not in network_graph.nodes[node]:
            network_graph.nodes[node]['IdeologyScore'] = random.uniform(-1, 1)

def compute_utility(G, source, beta=(1.0, 1.0, 1.0, 1.0)):
    """
    Computes utility from source to all other nodes in G.
    """
    utilities = {}
    w1, w2, w3, w4 = beta
    x_i = G.nodes[source].get('IdeologyScore', 0)

    cost_graph = G.copy()
    for u, v, data in cost_graph.edges(data=True):
        weight = data.get('weight', 1.0)
        data['cost'] = 1.0 / max(weight, 1e-5)

    shortest_paths = nx.single_source_dijkstra_path_length(cost_graph, source, weight='cost')

    for target in G.nodes:
        if target == source:
            continue

        connection_strength = 1.0 / (shortest_paths.get(target, float('inf')) + 1e-5)
        x_j = G.nodes[target].get('IdeologyScore', 0)
        proximity = -abs(x_i - x_j)
        centrality_j = G.nodes[target].get('EigenvectorCentrality', 1e-5)
        influenceability = 1.0 / (centrality_j + 1e-5)
        
        # For simplicity, interaction term is omitted here, but can be added.
        utility = (w1 * connection_strength +
                   w2 * proximity +
                   w3 * influenceability)
        utilities[target] = utility
        
    return utilities

def normalize_utility_matrix(utility_matrix: Dict[str, Dict[str, float]]) -> Dict[str, Dict[str, float]]:
    """Normalizes the utility matrix to a 0-1 scale for each source cardinal."""
    normalized_matrix = {}
    for source, utilities in utility_matrix.items():
        normalized_matrix[source] = {}
        if not utilities:
            continue
        
        min_val = min(utilities.values())
        max_val = max(utilities.values())
        
        if max_val == min_val:
            for target in utilities:
                normalized_matrix[source][target] = 0.5
        else:
            for target, value in utilities.items():
                normalized_matrix[source][target] = (value - min_val) / (max_val - min_val)
                
    return normalized_matrix

def form_non_overlapping_group(
    leader: str,
    potential_members: Set[str],
    normalized_utilities: Dict[str, Dict[str, float]],
    group_size: int,
    stochasticity: float = 0.0
) -> List[str]:
    """
    Forms a single non-overlapping group around a leader.
    Introduces stochasticity to member selection.
    stochasticity = 0: purely deterministic based on utility
    stochasticity = 1: purely random
    """
    group = [leader]
    leader_utilities = normalized_utilities.get(leader, {})

    if stochasticity > 0:
        # Probabilistic selection
        candidates_with_scores = []
        for member in potential_members:
            utility_score = leader_utilities.get(member, 0)
            random_score = random.random()
            # Blend utility with randomness
            final_score = (1 - stochasticity) * utility_score + stochasticity * random_score
            candidates_with_scores.append((member, final_score))
        
        # Sort by the blended score
        candidates_with_scores.sort(key=lambda x: x[1], reverse=True)
        candidates = [c[0] for c in candidates_with_scores]
    else:
        # Original deterministic logic
        candidates = sorted(
            list(potential_members),
            key=lambda m: leader_utilities.get(m, -1),
            reverse=True
        )

    members_to_add = min(len(candidates), group_size - 1)
    chosen_members = candidates[:members_to_add]
    group.extend(chosen_members)

    return group


def generate_non_overlapping_utility_groups(
    network_graph: nx.Graph, 
    cardinal_list: List[str], 
    group_size: int, 
    beta: Tuple[float, float, float, float] = (1.0, 1.0, 1.0, 1.0),
    seed: int = None,
    stochasticity: float = 0.0,
    leader_stochasticity: float = 0.0
) -> List[List[str]]:
    """
    Generate non-overlapping groups where high centrality cardinals form groups around themselves.
    Aims to create as many full groups as possible and avoids singleton groups where possible.
    """
    if seed is not None:
        random.seed(seed)
    
    ensure_ideology_scores(network_graph)
    
    available_cardinals = [c for c in cardinal_list if c in network_graph.nodes]
    
    centrality_scores = {c: network_graph.nodes[c].get('EigenvectorCentrality', 0) for c in available_cardinals}
    
    utility_matrix = {c: compute_utility(network_graph, c, beta) for c in available_cardinals}
    normalized_utilities = normalize_utility_matrix(utility_matrix)
    
    groups = []
    remaining_cardinals = set(available_cardinals)
    
    while remaining_cardinals:
        # Leader Selection
        if random.random() < leader_stochasticity and len(remaining_cardinals) > 1:
            # Choose a random leader from the remaining cardinals
            leader = random.choice(list(remaining_cardinals))
        else:
            # Choose the highest-centrality leader from the remaining cardinals
            leader = max(remaining_cardinals, key=lambda c: centrality_scores.get(c, 0))

        # The leader is available, form a group
        potential_members = remaining_cardinals - {leader}
        
        group = form_non_overlapping_group(
            leader,
            potential_members,
            normalized_utilities,
            group_size,
            stochasticity
        )
        
        groups.append(group)
        for member in group:
            if member in remaining_cardinals:
                remaining_cardinals.remove(member)

        if not remaining_cardinals:
            break  # All cardinals have been grouped

    # Post-processing: Re-balance to avoid singleton groups
    if len(groups) > 1 and len(groups[-1]) == 1:
        # If the last group is a singleton, borrow from the previous group.
        last_group = groups[-1]
        second_to_last_group = groups[-2]

        if len(second_to_last_group) > 1:
            # Move the last member (lowest utility for that group's leader)
            member_to_move = second_to_last_group.pop()
            last_group.append(member_to_move)

    return groups


