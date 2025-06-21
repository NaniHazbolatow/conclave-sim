#!/usr/bin/env python3
"""
Generate Predefined Groups for Conclave Simulation

This script analyzes the Bocconi network to create predefined groups
based on centrality and ideological balance.
"""

import pickle
import pandas as pd
import networkx as nx
from pathlib import Path
import yaml
from typing import Dict, List, Tuple, Optional
import numpy as np


class PredefinedGroupsGenerator:
    """Generate predefined groups based on network analysis."""
    
    def __init__(self, base_path=None):
        """Initialize the generator with paths."""
        if base_path is None:
            base_path = Path(__file__).parent.parent
        else:
            base_path = Path(base_path)
            
        self.base_path = base_path
        self.network_path = base_path / 'data' / 'network' / 'bocconi_graph.gpickle'
        self.config_path = base_path / 'config' / 'predefined_groups.yaml'
        
    def load_network(self) -> nx.Graph:
        """Load the Bocconi network."""
        with open(self.network_path, 'rb') as f:
            return pickle.load(f)
    
    def get_voting_cardinals(self, G: nx.Graph) -> List[int]:
        """Get list of voting cardinals (exclude non-voting)."""
        voting_cardinals = [
            n for n, d in G.nodes(data=True) 
            if d.get('Lean', '').lower() != 'non-voting'
        ]
        return voting_cardinals
    
    def categorize_ideology(self, lean: str) -> str:
        """Categorize political lean into progressive/conservative."""
        lean = lean.lower().strip()
        progressive = ['liberal', 'soft liberal']
        conservative = ['conservative', 'soft conservative']
        
        if lean in progressive:
            return 'progressive'
        elif lean in conservative:
            return 'conservative'
        else:
            return 'moderate'
    
    def balance_ideology(self, cardinals: List[int], G: nx.Graph, target_size: int) -> List[int]:
        """Balance ideology in group selection."""
        # Categorize cardinals by ideology
        progressive = []
        conservative = []
        moderate = []
        
        for cardinal in cardinals:
            lean = G.nodes[cardinal].get('Lean', '')
            category = self.categorize_ideology(lean)
            
            if category == 'progressive':
                progressive.append(cardinal)
            elif category == 'conservative':
                conservative.append(cardinal)
            else:
                moderate.append(cardinal)
        
        # Sort each category by centrality (descending)
        def sort_by_centrality(cardinals_list):
            return sorted(cardinals_list, 
                         key=lambda x: G.nodes[x].get('EigenvectorCentrality', 0), 
                         reverse=True)
        
        progressive = sort_by_centrality(progressive)
        conservative = sort_by_centrality(conservative)
        moderate = sort_by_centrality(moderate)
        
        # Select balanced representation
        selected = []
        prog_count = 0
        cons_count = 0
        mod_count = 0
        
        # Aim for roughly equal representation, prioritizing higher centrality
        target_prog = target_size // 3
        target_cons = target_size // 3
        target_mod = target_size - target_prog - target_cons
        
        # Select from each category
        while len(selected) < target_size:
            # Add progressive if available and under target
            if prog_count < target_prog and prog_count < len(progressive):
                selected.append(progressive[prog_count])
                prog_count += 1
            # Add conservative if available and under target
            elif cons_count < target_cons and cons_count < len(conservative):
                selected.append(conservative[cons_count])
                cons_count += 1
            # Add moderate if available and under target
            elif mod_count < target_mod and mod_count < len(moderate):
                selected.append(moderate[mod_count])
                mod_count += 1
            # Fill remaining slots with highest centrality available
            else:
                remaining = (progressive[prog_count:] + 
                           conservative[cons_count:] + 
                           moderate[mod_count:])
                if remaining:
                    remaining = sort_by_centrality(remaining)
                    selected.append(remaining[0])
                    # Update counters
                    if remaining[0] in progressive:
                        prog_count += 1
                    elif remaining[0] in conservative:
                        cons_count += 1
                    else:
                        mod_count += 1
                else:
                    break
        
        return selected[:target_size]
    
    def select_candidates(self, cardinals: List[int], G: nx.Graph, num_candidates: int) -> List[int]:
        """Select candidates based on highest centrality."""
        # Sort by centrality (descending)
        sorted_cardinals = sorted(
            cardinals, 
            key=lambda x: G.nodes[x].get('EigenvectorCentrality', 0), 
            reverse=True
        )
        return sorted_cardinals[:num_candidates]
    
    def generate_group(self, G: nx.Graph, voting_cardinals: List[int], 
                      group_size: int, num_candidates: int, use_balance: bool = True) -> Dict:
        """Generate a single group."""
        # For full group, use all voting cardinals
        if group_size >= len(voting_cardinals):
            selected_cardinals = voting_cardinals.copy()
        else:
            if use_balance:
                # Balance ideology for smaller groups
                selected_cardinals = self.balance_ideology(voting_cardinals, G, group_size)
            else:
                # Just take top centrality cardinals
                sorted_cardinals = sorted(
                    voting_cardinals, 
                    key=lambda x: G.nodes[x].get('EigenvectorCentrality', 0), 
                    reverse=True
                )
                selected_cardinals = sorted_cardinals[:group_size]
        
        # Select candidates from the group
        candidates = self.select_candidates(selected_cardinals, G, num_candidates)
        electors = [c for c in selected_cardinals if c not in candidates]
        
        return {
            'cardinal_ids': selected_cardinals,
            'candidate_ids': candidates,
            'elector_ids': electors,
            'total_cardinals': len(selected_cardinals),
            'num_candidates': len(candidates),
            'num_electors': len(electors)
        }
    
    def analyze_group_balance(self, cardinals: List[int], G: nx.Graph) -> Dict:
        """Analyze ideological balance of a group."""
        ideology_counts = {'progressive': 0, 'conservative': 0, 'moderate': 0}
        
        for cardinal in cardinals:
            lean = G.nodes[cardinal].get('Lean', '')
            category = self.categorize_ideology(lean)
            ideology_counts[category] += 1
        
        return ideology_counts
    
    def generate_predefined_groups(self) -> Dict:
        """Generate all predefined groups."""
        G = self.load_network()
        voting_cardinals = self.get_voting_cardinals(G)
        
        print(f"Network loaded: {len(G.nodes())} total cardinals, {len(voting_cardinals)} voting")
        
        # Define group specifications
        group_specs = {
            'small': {'size': 5, 'candidates': 2},
            'medium': {'size': 15, 'candidates': 3},
            'large': {'size': 25, 'candidates': 5},
            'xlarge': {'size': 50, 'candidates': 7},
            'full': {'size': len(voting_cardinals), 'candidates': 10}
        }
        
        predefined_groups = {}
        
        for group_name, spec in group_specs.items():
            print(f"\nGenerating {group_name} group...")
            
            # Use balance for all groups except full
            use_balance = group_name != 'full'
            
            group_data = self.generate_group(
                G, voting_cardinals, 
                spec['size'], spec['candidates'], 
                use_balance
            )
            
            # Add analysis
            balance = self.analyze_group_balance(group_data['cardinal_ids'], G)
            
            # Create group configuration
            predefined_groups[group_name] = {
                'total_cardinals': group_data['total_cardinals'],
                'cardinal_ids': group_data['cardinal_ids'],
                'candidate_ids': group_data['candidate_ids'],
                'elector_ids': group_data['elector_ids'],
                'description': f"{group_name.title()} group with {group_data['total_cardinals']} cardinals and {group_data['num_candidates']} candidates",
                'ideology_balance': balance
            }
            
            print(f"  Cardinals: {group_data['total_cardinals']}")
            print(f"  Candidates: {group_data['num_candidates']}")
            print(f"  Ideology balance: {balance}")
        
        return {'predefined_groups': predefined_groups}
    
    def save_groups(self, groups_data: Dict):
        """Save predefined groups to YAML file."""
        # Convert numpy integers to regular Python integers for clean YAML
        def convert_numpy_ints(obj):
            if isinstance(obj, dict):
                return {k: convert_numpy_ints(v) for k, v in obj.items()}
            elif isinstance(obj, list):
                return [convert_numpy_ints(item) for item in obj]
            elif isinstance(obj, np.integer):
                return int(obj)
            else:
                return obj
        
        clean_groups_data = convert_numpy_ints(groups_data)
        
        with open(self.config_path, 'w') as f:
            yaml.dump(clean_groups_data, f, default_flow_style=False, sort_keys=False)
        
        print(f"\nPredefined groups saved to: {self.config_path}")
    
    def run(self):
        """Run the complete group generation process."""
        print("Generating predefined groups based on network analysis...")
        groups_data = self.generate_predefined_groups()
        self.save_groups(groups_data)
        print("Done!")

if __name__ == "__main__":
    generator = PredefinedGroupsGenerator()
    generator.run()
