import networkx as nx
import pandas as pd

def add_networks(G1, G2, weight_attr='weight', default_weight=1.0, combine_node_attrs=True):
    """Combine two weighted graphs by summing edge weights and optionally merging node attributes."""
    result = type(G1)()
    result.add_nodes_from(G1.nodes(data=True))
    result.add_nodes_from(G2.nodes(data=True))

    if combine_node_attrs:
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

    for u, v, data in G1.edges(data=True):
        add_or_update_edge(u, v, data, result)
    for u, v, data in G2.edges(data=True):
        add_or_update_edge(u, v, data, result)

    return result

def load_node_attributes(G, excel_file, node_col='Surname'):
    """Load node attributes from Excel and attach to NetworkX graph."""
    node_df = pd.read_excel(excel_file)
    node_df = node_df.set_index(node_col)
    node_attrs = node_df.to_dict('index')
    nx.set_node_attributes(G, node_attrs)
    return G

def pope_candidate(data):
    """Determine whether a cardinal is a viable candidate for the papacy."""
    lean = str(data.get('Lean', '')).strip().lower()
    age = data.get('Age', 0)
    return lean != 'non-voting' or age < 80
