# Utility-Based Group Formation for Conclave Simulation

## Current Utility Function Analysis

The existing `compute_utility()` function provides a sophisticated multi-factor scoring system:

- **Connection Strength**: Network distance between cardinals (closer = higher utility)
- **Ideological Proximity**: Political alignment similarity (closer = higher utility) 
- **Influenceability**: Inverse centrality (less influential = higher utility to target)
- **Interaction Effects**: Network connections × ideological alignment

## Implementation Approach

### Option A: Direct Utility-Based Grouping (Recommended)

```python
def generate_utility_groups(network_graph, cardinal_list, group_sizes, utility_threshold=0.5, beta=(1.0, 1.0, 1.0, 1.0)):
    """
    Generate groups based on pairwise utility scores between cardinals.
    
    Parameters:
    - network_graph: NetworkX graph with cardinal network data
    - cardinal_list: List of cardinal names/IDs 
    - group_sizes: List of desired group sizes [3, 4, 3, ...]
    - utility_threshold: 0-1, minimum utility for grouping preference (higher = more selective)
    - beta: Utility function weights (connection, ideology, influenceability, interaction)
    
    Returns:
    - List of groups: [[cardinal1, cardinal2, cardinal3], [cardinal4, ...], ...]
    """
    
    # 1. Prepare network data
    ensure_ideology_scores(network_graph)  # Convert Lean to numeric IdeologyScore
    
    # 2. Compute all pairwise utilities
    utility_matrix = {}
    for cardinal in cardinal_list:
        if cardinal in network_graph.nodes:
            utilities = compute_utility(network_graph, cardinal, beta)
            utility_matrix[cardinal] = utilities
    
    # 3. Normalize utilities to 0-1 range
    normalized_utilities = normalize_utility_matrix(utility_matrix)
    
    # 4. Group formation algorithm
    groups = []
    remaining_cardinals = set(cardinal_list)
    
    for group_size in group_sizes:
        if len(remaining_cardinals) < group_size:
            # Add remaining cardinals to last group
            if groups:
                groups[-1].extend(list(remaining_cardinals))
            break
            
        # Find best group using utility-based clustering
        group = form_utility_group(
            remaining_cardinals, 
            normalized_utilities,
            group_size, 
            utility_threshold
        )
        
        groups.append(group)
        remaining_cardinals -= set(group)
    
    return groups

def ensure_ideology_scores(graph):
    """Convert Lean categorical values to numeric IdeologyScore if needed."""
    lean_to_score = {
        'liberal': -1.0,
        'soft liberal': -0.5, 
        'moderate': 0.0,
        'soft conservative': 0.5,
        'conservative': 1.0,
        'non-voting': 0.0  # Neutral for non-voting cardinals
    }
    
    for node in graph.nodes():
        if 'IdeologyScore' not in graph.nodes[node]:
            lean = str(graph.nodes[node].get('Lean', 'moderate')).lower().strip()
            graph.nodes[node]['IdeologyScore'] = lean_to_score.get(lean, 0.0)

def normalize_utility_matrix(utility_matrix):
    """Normalize all utility scores to 0-1 range."""
    all_scores = []
    for cardinal_utilities in utility_matrix.values():
        all_scores.extend(cardinal_utilities.values())
    
    min_score = min(all_scores)
    max_score = max(all_scores)
    score_range = max_score - min_score
    
    if score_range == 0:
        return utility_matrix  # All scores are the same
    
    normalized = {}
    for cardinal, utilities in utility_matrix.items():
        normalized[cardinal] = {
            target: (score - min_score) / score_range 
            for target, score in utilities.items()
        }
    
    return normalized

def form_utility_group(available_cardinals, utility_matrix, group_size, threshold):
    """Form a single group using utility-based selection."""
    
    if group_size == 1:
        return [available_cardinals.pop()]
    
    # Start with a random cardinal or most central one
    group = [next(iter(available_cardinals))]
    available_cardinals.remove(group[0])
    
    # Add cardinals based on utility scores
    while len(group) < group_size and available_cardinals:
        best_candidate = None
        best_avg_utility = -1
        
        for candidate in available_cardinals:
            # Calculate average utility between candidate and current group members
            utilities_to_group = []
            for group_member in group:
                if group_member in utility_matrix and candidate in utility_matrix[group_member]:
                    utilities_to_group.append(utility_matrix[group_member][candidate])
                if candidate in utility_matrix and group_member in utility_matrix[candidate]:
                    utilities_to_group.append(utility_matrix[candidate][group_member])
            
            if utilities_to_group:
                avg_utility = sum(utilities_to_group) / len(utilities_to_group)
                if avg_utility > best_avg_utility:
                    best_avg_utility = avg_utility
                    best_candidate = candidate
        
        # Add best candidate if above threshold, otherwise add random
        if best_candidate and best_avg_utility >= threshold:
            group.append(best_candidate)
            available_cardinals.remove(best_candidate)
        else:
            # Fallback to random selection if no good utility matches
            random_candidate = next(iter(available_cardinals))
            group.append(random_candidate)
            available_cardinals.remove(random_candidate)
    
    return group
```

### Option B: Simplified Distance-Based Grouping

```python
def generate_simple_utility_groups(network_graph, cardinal_list, group_sizes, homophily=0.7):
    """
    Simplified approach using just connection strength + ideological proximity.
    
    Parameters:
    - homophily: 0-1, where 1 = only group with highest utility, 0 = random
    """
    
    # Use simplified utility: just connection + ideology  
    beta = (1.0, 1.0, 0.0, 0.5)  # (connection, ideology, no influenceability, some interaction)
    
    return generate_utility_groups(network_graph, cardinal_list, group_sizes, 
                                   utility_threshold=homophily, beta=beta)
```

## Advantages of Utility-Based Approach

✅ **Uses existing sophisticated model** - leverages the multi-factor utility computation
✅ **Incorporates all key factors** - network distance, ideology, influence, interactions  
✅ **Tunable via threshold** - utility_threshold acts like homophily parameter
✅ **Flexible weighting** - can adjust beta weights for different scenarios
✅ **Realistic groupings** - based on actual cardinal relationships and characteristics

## Potential Missing Information

🤔 **Nationality/Regional factors** - not explicitly in utility function (could be added)
🤔 **Age cohort effects** - generational similarity not captured
🤔 **Institutional experience** - Vatican dicastery relationships only partly captured

## Recommendation

**Start with Option A** (Direct Utility-Based Grouping) because:
1. It uses the sophisticated utility model you already have
2. The utility_threshold parameter gives you the same control as homophily (0-1)
3. You can easily tune the beta weights to emphasize different factors
4. It's a natural extension of your existing network analysis

The utility scores effectively capture the most important grouping factors:
- **Social connections** (who knows whom)
- **Political alignment** (ideological similarity) 
- **Strategic considerations** (influence levels)
- **Interaction effects** (connected AND ideologically similar)

**Missing regional/age factors** could be added later if needed, but the current utility model captures the most crucial relationships for realistic conclave groupings.
