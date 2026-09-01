"""Hierarchical Navigable Small World (HNSW) graph for approximate nearest
neighbour search over normalized vectors (cosine via dot product).

A dependency-free reference implementation (Malkov & Yashunin, 2016):
nodes get a random level from a geometric distribution; upper layers form
coarse express lanes and layer 0 holds the full graph. Queries greedily
descend the layers, then run a best-first search with an ``ef`` beam at the
bottom.

Suited to mid-sized collections (up to a few hundred thousand vectors);
insertion is pure Python. The graph is rebuilt on load rather than
persisted.
"""

import heapq
import math
import random
from typing import List, Tuple

import numpy as np


class HNSW:
    def __init__(self, dim: int, m: int = 16, ef_construction: int = 200,
                 seed: int = 0):
        self.dim = dim
        self.m = m
        self.m0 = 2 * m
        self.ef_construction = ef_construction
        self.level_mult = 1.0 / math.log(m)
        self.rng = random.Random(seed)
        self.vectors: List[np.ndarray] = []
        # graph[node][level] -> list of neighbour ids (levels 0..node_level)
        self.graph: List[List[List[int]]] = []
        self.entry: int = -1
        self.max_level: int = -1

    def __len__(self) -> int:
        return len(self.vectors)

    def _sim(self, node: int, q: np.ndarray) -> float:
        return float((self.vectors[node] * q).sum())

    def _search_layer(self, q: np.ndarray, entry_points: List[int],
                      ef: int, level: int) -> List[Tuple[float, int]]:
        """Best-first search on one layer; returns up to ef (sim, id) pairs."""
        visited = set(entry_points)
        candidates = [(-self._sim(e, q), e) for e in entry_points]
        heapq.heapify(candidates)
        # Min-heap of results: the worst kept result sits on top.
        best = [(self._sim(e, q), e) for e in entry_points]
        heapq.heapify(best)
        while len(best) > ef:
            heapq.heappop(best)

        while candidates:
            neg_sim, node = heapq.heappop(candidates)
            if len(best) >= ef and -neg_sim < best[0][0]:
                break
            for neighbour in self.graph[node][level]:
                if neighbour in visited:
                    continue
                visited.add(neighbour)
                sim = self._sim(neighbour, q)
                if len(best) < ef or sim > best[0][0]:
                    heapq.heappush(candidates, (-sim, neighbour))
                    heapq.heappush(best, (sim, neighbour))
                    if len(best) > ef:
                        heapq.heappop(best)
        return list(best)

    def _select_neighbours(self, candidates: List[Tuple[float, int]],
                           m: int, anchor: int = -1) -> List[int]:
        """Heuristic neighbour selection (Malkov & Yashunin, Alg. 4).

        A candidate is kept only if it is closer to the query than to any
        already-selected neighbour. This preserves long-range links between
        clusters; pruning purely by similarity disconnects the graph on
        clustered data. Remaining slots are filled with the best leftovers.
        """
        selected: List[Tuple[float, int]] = []
        ordered = sorted(candidates, reverse=True)
        for sim, cand in ordered:
            if cand == anchor:
                continue
            if len(selected) >= m:
                break
            diverse = True
            for _, kept in selected:
                if float((self.vectors[cand]
                          * self.vectors[kept]).sum()) > sim:
                    diverse = False
                    break
            if diverse:
                selected.append((sim, cand))
        if len(selected) < m:
            chosen = {i for _, i in selected}
            for sim, cand in ordered:
                if len(selected) >= m:
                    break
                if cand != anchor and cand not in chosen:
                    selected.append((sim, cand))
        return [i for _, i in selected]

    def add(self, vector: np.ndarray) -> int:
        """Insert one (normalized) vector; returns its node id."""
        vector = np.asarray(vector, dtype=np.float32)
        node = len(self.vectors)
        self.vectors.append(vector)
        level = int(-math.log(self.rng.random() + 1e-12) * self.level_mult)
        self.graph.append([[] for _ in range(level + 1)])

        if self.entry < 0:
            self.entry = node
            self.max_level = level
            return node

        entry_points = [self.entry]
        # Greedy descent through layers above the new node's level.
        for layer in range(self.max_level, level, -1):
            best = self._search_layer(vector, entry_points, 1, layer)
            entry_points = [max(best)[1]]

        for layer in range(min(level, self.max_level), -1, -1):
            best = self._search_layer(
                vector, entry_points, self.ef_construction, layer)
            max_links = self.m0 if layer == 0 else self.m
            neighbours = self._select_neighbours(best, max_links, anchor=node)
            self.graph[node][layer] = list(neighbours)
            for neighbour in neighbours:
                links = self.graph[neighbour][layer]
                links.append(node)
                if len(links) > max_links:
                    ranked = [
                        (float((self.vectors[x]
                                * self.vectors[neighbour]).sum()), x)
                        for x in links
                    ]
                    links[:] = self._select_neighbours(
                        ranked, max_links, anchor=neighbour)
            entry_points = [i for _, i in best]

        if level > self.max_level:
            self.max_level = level
            self.entry = node
        return node

    def search(self, q: np.ndarray, k: int,
               ef: int = 0) -> List[Tuple[float, int]]:
        """Return up to k (similarity, node_id) pairs, best first."""
        if self.entry < 0:
            return []
        q = np.asarray(q, dtype=np.float32)
        ef = max(ef, 50, k)
        entry_points = [self.entry]
        for layer in range(self.max_level, 0, -1):
            best = self._search_layer(q, entry_points, 1, layer)
            entry_points = [max(best)[1]]
        best = self._search_layer(q, entry_points, ef, 0)
        return sorted(best, reverse=True)[:k]

