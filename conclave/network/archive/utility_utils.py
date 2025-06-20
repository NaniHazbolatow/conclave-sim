import random
import itertools
from collections import defaultdict
from typing import Any, Dict, List, Set, Tuple

import networkx as nx
import numpy as np

class BreakoutScheduler:
    def __init__(
        self,
        graph: nx.Graph,
        room_size: int,
        weights: Tuple[float, float, float, float] = (1,1,1,1),
        rationality: float = 1.0,
        penalty_weight: float = 0.5,   # ← how strongly to avoid repeats
    ):
        assert room_size > 1, "room_size must be ≥2"
        self.G = graph
        self.room_size = room_size
        self.weights = weights
        self.lam = rationality
        self.penalty_weight = penalty_weight

        # Precompute raw utilities
        self.utilities: Dict[Any, Dict[Any, float]] = {
            node: self._calc_utils_from(node)
            for node in graph.nodes()
        }

        # Track past co-assignments
        self.pair_history: Dict[frozenset, int] = defaultdict(int)
        self._round = 0

    def _calc_utils_from(self, source: Any) -> Dict[Any, float]:
        w_conn, w_prox, w_inf, w_int = self.weights
        x_i = self.G.nodes[source].get('IdeologyScore', 0)

        costG = self.G.copy()
        for u, v, d in costG.edges(data=True):
            w = d.get('weight',1e-5)
            d['cost'] = 1 / max(w,1e-5)

        dists = nx.single_source_dijkstra_path_length(costG, source, weight='cost')
        utils = {}
        for tgt in self.G.nodes():
            if tgt == source:
                continue
            cs = 1.0 / (dists.get(tgt, np.inf) + 1e-5)
            x_j = self.G.nodes[tgt].get('IdeologyScore', 0)
            prox = -abs(x_i - x_j)
            cent = self.G.nodes[tgt].get('EigenvectorCentrality', 1e-5)
            infl = 1.0 / cent if cent > 0 else 0
            inter = cs * prox
            utils[tgt] = w_conn*cs + w_prox*prox + w_inf*infl + w_int*inter
        
        sigmoid_utils = {
            tgt: 1.0 / (1.0 + np.exp(-score))
            for tgt, score in utils.items()
        }
        return sigmoid_utils

    def next_round(self) -> List[List[Any]]:
        self._round += 1
        assigned: Set[Any] = set()
        rooms: List[List[Any]] = []

        centrality = nx.eigenvector_centrality_numpy(self.G)
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
            util_pool = sorted(
                util_candidates,
                key=lambda tgt: raw_utils[tgt]
                                 - self.penalty_weight * self.pair_history[frozenset({source, tgt})],
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

            # 3) Pick up to room_size - 1, never repeating picks
            picks: List[Any] = []
            while len(picks) < self.room_size - 1:
                if util_pool and random.random() < self.lam:
                    nxt = util_pool.pop(0)
                elif rand_pool:
                    nxt = rand_pool.pop(0)
                else:
                    # fallback over everyone still available
                    leftovers = [
                        n for n in self.G.nodes()
                        if n not in assigned and n != source and n not in picks
                    ]
                    if not leftovers:
                        break
                    nxt = random.choice(leftovers)
                if nxt in picks:
                    continue
                picks.append(nxt)

            room = [source] + picks
            rooms.append(room)
            assigned.update(room)

            if len(assigned) >= self.G.number_of_nodes():
                break

        # Update pair-history
        for room in rooms:
            for a, b in itertools.combinations(room, 2):
                self.pair_history[frozenset({a, b})] += 1

        return rooms


# Read graph
import pickle
with open('bocconi_graph.gpickle', 'rb') as f:
    G_multiplex = pickle.load(f)

scheduler = BreakoutScheduler(G_multiplex, room_size=5, rationality=1, penalty_weight=0.1)
for _ in range(6):
    print("Round", scheduler._round+1, "rooms:", scheduler.next_round())
