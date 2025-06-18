# Custom functions
from network_utils import add_networks, load_node_attributes, pope_candidate
import os
import logging

# Configure logging
logging.basicConfig(level=logging.INFO)

# Packages
import polars as pl
import pandas as pd
import networkx as nx
from collections import defaultdict
from itertools import combinations
import matplotlib.pyplot as plt
from matplotlib import colormaps
from matplotlib.colors import Normalize
from matplotlib.patches import Patch

# Initialize episcopal consecration matrix from Soda et al. (2025) 
# Episcopal consecration gives us insight into who ordained who
G_consecration = nx.from_pandas_adjacency(
    pd.read_excel('adjacency_matrix_consecrator.xlsx', index_col=0), 
    create_using=nx.Graph()
)

# Initialize Vatican co-memberships matrix from Soda et al. (2025)
# Vatican co-membership gives us insight into who works with who
official_comemberships = pd.read_excel('formal.xlsx')
all_cardinals = set(official_comemberships['Person'].unique())

G_comembership = nx.Graph()
G_comembership.add_nodes_from(all_cardinals)

# Initialize weights as membership ties
weights = defaultdict(int)

groups = official_comemberships.groupby('Membership')['Person'].apply(list)

for members in groups:
    if len(members) > 1:
        for pair in combinations(sorted(members), 2):
            weights[pair] += 1

# Add weighted edges to the formal graph
for (cardinal1, cardinal2), weight in weights.items():
    G_comembership.add_edge(cardinal1, cardinal2, weight=weight)


# We have information of both networks in 1 now
G_multiplex = add_networks(G_consecration, G_comembership, weight_attr='weight', combine_node_attrs=True)

# Enrich nodes with information (full name)
G_multiplex = load_node_attributes(G_multiplex, 'node_info.xlsx')

# Meta-information about each node
centrality = nx.eigenvector_centrality(G_multiplex, weight='weight')
nx.set_node_attributes(G_multiplex, centrality, 'EigenvectorCentrality')

#####################
### Utility Model ###
#####################
def compute_utility(G, source, beta=(1.0, 1.0, 1.0, 1.0)):
    """
    Computes utility from source to all other nodes in G using Dijkstra-based connection strength.

    Parameters:
    - G: NetworkX graph with 'IdeologyScore' and 'EigenvectorCentrality' node attributes
    - source: source node
    - beta: (w1, w2, w3, w4) = (connection strength, ideological proximity, influenceability, interaction)

    Returns:
    - Dictionary mapping target node to utility score
    """
    utilities = {}
    w1, w2, w3, w4 = beta
    x_i = G.nodes[source].get('IdeologyScore', 0)

    # 1. Compute Dijkstra costs with cost = 1 / weight
    cost_graph = G.copy()
    for u, v, data in cost_graph.edges(data=True):
        weight = data.get('weight', 1.0)
        data['cost'] = 1.0 / max(weight, 1e-5)  # avoid div by zero

    # 2. Compute shortest path lengths
    shortest_paths = nx.single_source_dijkstra_path_length(cost_graph, source, weight='cost')

    for target in G.nodes:
        if target == source:
            continue

        # 3. Connection strength = inverse of shortest path length
        if target in shortest_paths:
            connection_strength = 1.0 / (shortest_paths[target] + 1e-5)
        else:
            connection_strength = 0.0

        # 4. Ideological proximity
        x_j = G.nodes[target].get('IdeologyScore', 0)
        proximity = -abs(x_i - x_j)

        # 5. Influenceability = inverse of EigenvectorCentrality
        centrality_j = G.nodes[target].get('EigenvectorCentrality', 1e-5)
        influenceability = 1.0 / centrality_j if centrality_j > 0 else 0

        # 6. Interaction term
        interaction = connection_strength * proximity

        # 7. Linear utility with interaction
        utility = (
            w1 * connection_strength +
            w2 * proximity +
            w3 * influenceability +
            w4 * interaction
        )

        utilities[target] = utility

    return utilities


utilities = compute_utility(G_multiplex, source='Prevost')
top_targets = sorted(utilities.items(), key=lambda x: x[1], reverse=True)[:10]
logging.info(f"Top 10 targets by utility: {top_targets}")

#######################################
### Appendix: Extra Plot for Report ###
#######################################

from matplotlib.colors import Normalize
from matplotlib.patches import Patch
from matplotlib import colormaps

# Filter graph to only include papal candidates
G_electors = G_multiplex.subgraph(
    [n for n, d in G_multiplex.nodes(data=True) if pope_candidate(d)]
).copy()

# Define political lean → numerical score
lean_score_map = {
    'liberal': -1,
    'soft liberal': -0.5,
    'moderate': 0,
    'soft conservative': 0.5,
    'conservative': 1
}

# Extract and score political lean for each node
partition = {
    n: str(d.get('Lean', 'Unknown')).strip().lower()
    for n, d in G_electors.nodes(data=True)
}
default_color = (0.9, 0.85, 0.75, 1.0)

node_scores = [
    lean_score_map.get(partition.get(n, ''), None)
    for n in G_multiplex.nodes()
]

# Color map from lean score
cmap = colormaps.get_cmap('coolwarm')
norm = Normalize(vmin=-1, vmax=1)
node_colors = [
    cmap(norm(score)) if score is not None else default_color
    for score in node_scores
]

# Node layout and size
pos = nx.spring_layout(G_multiplex, seed=32, k=3, scale=4)
node_sizes = [centrality[n] * 20000 for n in G_multiplex.nodes()]

# Edge transparency normalization
min_c, max_c = min(centrality.values()), max(centrality.values())
normalize = lambda c: (c - min_c) / (max_c - min_c) if max_c > min_c else 0.5

# Plot setup
plt.figure(figsize=(16, 14))
nx.draw_networkx_nodes(
    G_multiplex,
    pos,
    node_color=node_colors,
    node_size=node_sizes,
    alpha=0.85,
    linewidths=0.3,
    edgecolors='black'
)

filtered_edges = [
    (u, v) for u, v in G_multiplex.edges()
    if G_multiplex[u][v].get('weight', 1) > 1
]

# Draw edges with adaptive transparency
for u, v in filtered_edges: # G_multiplex.edges():
    alpha = 0.01 + normalize((centrality[u] + centrality[v]) / 2) * 0.15
    width = G_multiplex[u][v].get('weight', 1)
    nx.draw_networkx_edges(G_multiplex, pos, edgelist=[(u, v)], width=width, alpha=alpha)

# Label top 10 most central candidates
top_labels = {
    n: G_multiplex.nodes[n].get('Full Name', n)
    for n, _ in sorted(centrality.items(), key=lambda x: x[1], reverse=True)[:10]
}
nx.draw_networkx_labels(G_multiplex, pos, labels=top_labels, font_size=10, font_weight='bold')

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


# Get lean scores for electors
elector_scores = [
    lean_score_map.get(str(d.get('Lean', '')).strip().lower())
    for n, d in G_electors.nodes(data=True)
    if str(d.get('Lean', '')).strip().lower() in lean_score_map
]

# F-score: average of lean scores
f_score = sum(elector_scores) / len(elector_scores)

# Count conservatives and soft conservatives
conservatives = sum(
    1 for d in G_multiplex.nodes.values()
    if str(d.get('Lean', '')).strip().lower() == 'conservative'
)

soft_conservatives = sum(
    1 for d in G_multiplex.nodes.values()
    if str(d.get('Lean', '')).strip().lower() == 'soft conservative'
)

liberals = sum(
    1 for d in G_multiplex.nodes.values()
    if str(d.get('Lean', '')).strip().lower() == 'liberal'
)

soft_liberals = sum(
    1 for d in G_multiplex.nodes.values()
    if str(d.get('Lean', '')).strip().lower() == 'soft liberal'
)

logging.info(f"F-score (mean lean among electors): {f_score:.3f}")
logging.info(f"Number of conservatives in the college: {conservatives}")
logging.info(f"Number of soft conservatives in the college: {soft_conservatives}")
logging.info(f"Number of soft liberals in the college: {soft_liberals}")
logging.info(f"Number of liberals in the college: {liberals}")

# Final polish and export
plt.title("Cardinal Network — Grouped by Political Lean", fontsize=18)
plt.axis('off')
plt.tight_layout()
plt.savefig("network/cardinal_network.png", dpi=300)
