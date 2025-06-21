"""
Advanced Multi-Round Grouping Scheduler for Conclave Simulation

This module provides a sophisticated grouping scheduler that tracks interaction history
and uses utility-based assignment with penalty systems to avoid repeated pairings.
"""

import random
import itertools
import logging
from collections import defaultdict
from typing import Any, Dict, List, Set, Tuple

import networkx as nx
import numpy as np

logger = logging.getLogger(__name__)


class BreakoutScheduler:
    """
    Advanced scheduler for multi-round discussion groups that tracks interaction history
    and uses utility-based assignment with penalties for repeated pairings.
    """
    
    def __init__(
        self,
        graph: nx.Graph,
        room_size: int,
        weights: Tuple[float, float, float, float] = (1.0, 1.0, 1.0, 1.0),
        rationality: float = 1.0,
        penalty_weight: float = 0.5,
    ):
        """
        Initialize the BreakoutScheduler.
        
        Args:
            graph: NetworkX graph with cardinal nodes and attributes
            room_size: Target size for each discussion group
            weights: Utility weights (connection, ideology, influence, interaction)
            rationality: Balance between utility-based (1.0) and random (0.0) selection
            penalty_weight: How strongly to avoid repeated pairings
        """
        assert room_size > 1, "room_size must be ≥2"
        self.G = graph
        self.room_size = room_size
        self.weights = weights
        self.lam = rationality  # rationality parameter
        self.penalty_weight = penalty_weight

        # Precompute raw utilities for all node pairs
        logger.info("Precomputing utility matrix for BreakoutScheduler...")
        self.utilities: Dict[Any, Dict[Any, float]] = {
            node: self._calc_utils_from(node)
            for node in graph.nodes()
        }

        # Track past co-assignments
        self.pair_history: Dict[frozenset, int] = defaultdict(int)
        self._round = 0
        
        logger.info(f"BreakoutScheduler initialized with {len(self.G.nodes)} nodes, "
                   f"room_size={room_size}, rationality={rationality}, "
                   f"penalty_weight={penalty_weight}")

    def _calc_utils_from(self, source: Any) -> Dict[Any, float]:
        """
        Calculate utility scores from a source node to all other nodes.
        
        Args:
            source: Source node ID
            
        Returns:
            Dictionary mapping target nodes to utility scores
        """
        w_conn, w_prox, w_inf, w_int = self.weights
        x_i = self.G.nodes[source].get('IdeologyScore', 0)

        # Create cost graph for shortest path calculations
        costG = self.G.copy()
        for u, v, d in costG.edges(data=True):
            w = d.get('weight', 1e-5)
            d['cost'] = 1 / max(w, 1e-5)

        # Calculate shortest path distances
        dists = nx.single_source_dijkstra_path_length(costG, source, weight='cost')
        
        utils = {}
        for tgt in self.G.nodes():
            if tgt == source:
                continue
                
            # Connection strength (inverse of network distance)
            cs = 1.0 / (dists.get(tgt, np.inf) + 1e-5)
            
            # Ideological proximity (negative absolute difference)
            x_j = self.G.nodes[tgt].get('IdeologyScore', 0)
            prox = -abs(x_i - x_j)
            
            # Influenceability (inverse of centrality)
            cent = self.G.nodes[tgt].get('EigenvectorCentrality', 1e-5)
            infl = 1.0 / cent if cent > 0 else 0
            
            # Interaction term (connection * proximity)
            inter = cs * prox
            
            # Combined utility
            utils[tgt] = w_conn*cs + w_prox*prox + w_inf*infl + w_int*inter
        
        # Apply sigmoid transformation to normalize utilities
        sigmoid_utils = {
            tgt: 1.0 / (1.0 + np.exp(-score))
            for tgt, score in utils.items()
        }
        return sigmoid_utils

    def next_round(self) -> List[List[Any]]:
        """
        Generate groups for the next discussion round.
        
        Returns:
            List of groups, where each group is a list of node IDs
        """
        self._round += 1
        logger.info(f"Generating groups for round {self._round}")
        
        assigned: Set[Any] = set()
        rooms: List[List[Any]] = []

        # Get centrality scores for leader selection
        centrality = nx.eigenvector_centrality_numpy(self.G)
            
        # Sort nodes by centrality (highest first) for leader selection
        seeds = sorted(centrality, key=lambda n: -centrality[n])

        for source in seeds:
            if source in assigned:
                continue

            # 1) Build filtered utility pool (exclude assigned & self)
            raw_utils = self.utilities[source]
            util_candidates = [
                tgt for tgt in raw_utils
                if tgt not in assigned and tgt != source
            ]
            
            # Apply penalty for past pairings
            util_pool = sorted(
                util_candidates,
                key=lambda tgt: (raw_utils[tgt] 
                               - self.penalty_weight * self.pair_history[frozenset({source, tgt})]),
                reverse=True
            )

            # 2) Build filtered random pool
            remaining = [
                n for n in self.G.nodes()
                if n not in assigned and n != source
            ]
            rand_pool = random.sample(
                remaining,
                k=min(len(remaining), self.room_size - 1)
            )

            # 3) Pick up to room_size - 1 members using network-based selection only
            picks: List[Any] = []
            while len(picks) < self.room_size - 1:
                if util_pool and random.random() < self.lam:
                    # Use utility-based selection (primary method)
                    nxt = util_pool.pop(0)
                elif rand_pool:
                    # Use random selection from network nodes
                    nxt = rand_pool.pop(0)
                else:
                    # No more candidates available
                    break
                    
                if nxt in picks:
                    continue
                picks.append(nxt)

            # Create the room with leader + selected members
            room = [source] + picks
            rooms.append(room)
            assigned.update(room)

            if len(assigned) >= self.G.number_of_nodes():
                break

        # Update pair-history for penalty system
        for room in rooms:
            for a, b in itertools.combinations(room, 2):
                self.pair_history[frozenset({a, b})] += 1

        logger.info(f"Generated {len(rooms)} groups for round {self._round}, "
                   f"assigned {len(assigned)} nodes")

        return rooms

    def get_round_number(self) -> int:
        """Get the current round number."""
        return self._round

    def get_pair_history(self) -> Dict[frozenset, int]:
        """Get the interaction history between all pairs."""
        return dict(self.pair_history)

    def reset_history(self) -> None:
        """Reset the interaction history (for new simulations)."""
        self.pair_history.clear()
        self._round = 0
        logger.info("BreakoutScheduler history reset")
