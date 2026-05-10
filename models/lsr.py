"""
Latent Space Roadmap (LSR).

Graph construction:
  1. Encode all training frames → latent cloud Z.
  2. For each node, find its k nearest neighbours.
  3. Keep an edge (i, j) only when the pair appears as consecutive
     timesteps in at least one training trajectory (i.e. a real action
     connects them).  This prevents the planner from taking shortcuts
     through physically unreachable states.

Planning:
  Given z_start and z_goal, snap to nearest graph nodes then run
  Dijkstra weighted by Euclidean latent distance.
"""

import heapq
from typing import Dict, List, Optional, Tuple

import numpy as np
from scipy.spatial import KDTree


class LSR:
    def __init__(self, k: int = 10):
        self.k = k
        self.latents: Optional[np.ndarray] = None
        self.tree: Optional[KDTree] = None
        # adjacency: node → [(neighbour, euclidean_weight), ...]
        self.graph: Dict[int, List[Tuple[int, float]]] = {}
        # valid_transitions: (i, j) → action that takes state_i → state_j
        self.valid_transitions: Dict[Tuple[int, int], np.ndarray] = {}

    # ------------------------------------------------------------------
    # Graph construction
    # ------------------------------------------------------------------

    def build(
        self,
        latents: np.ndarray,
        episodes: List[List[int]],
        actions: np.ndarray,
    ) -> None:
        """
        latents  : (N, z_dim)  — encoded training frames
        episodes : list of index-lists, one list per trajectory
        actions  : (N, action_dim) — action[i] moves frame i → frame i+1
        """
        self.latents = latents.copy()
        self.tree = KDTree(latents)
        self.graph = {}
        self.valid_transitions = {}

        # Record every consecutive transition
        for ep in episodes:
            for t in range(len(ep) - 1):
                i, j = ep[t], ep[t + 1]
                self.valid_transitions[(i, j)] = actions[i]

        valid_set = set(self.valid_transitions.keys())
        n = len(latents)

        for i in range(n):
            _, nbrs = self.tree.query(latents[i], k=min(self.k + 1, n))
            for j in nbrs[1:]:          # skip self
                j = int(j)
                if (i, j) in valid_set or (j, i) in valid_set:
                    w = float(np.linalg.norm(latents[i] - latents[j]))
                    self.graph.setdefault(i, []).append((j, w))
                    self.graph.setdefault(j, []).append((i, w))

    # ------------------------------------------------------------------
    # Planning
    # ------------------------------------------------------------------

    def _dijkstra(self, src: int, dst: int) -> Optional[List[int]]:
        dist: Dict[int, float] = {src: 0.0}
        prev: Dict[int, int] = {}
        pq = [(0.0, src)]
        visited: set = set()

        while pq:
            d, u = heapq.heappop(pq)
            if u in visited:
                continue
            visited.add(u)

            if u == dst:
                path, cur = [], dst
                while cur != src:
                    path.append(cur)
                    cur = prev[cur]
                path.append(src)
                return path[::-1]

            for v, w in self.graph.get(u, []):
                nd = d + w
                if nd < dist.get(v, float("inf")):
                    dist[v] = nd
                    prev[v] = u
                    heapq.heappush(pq, (nd, v))

        return None  # no path

    def plan(
        self,
        start_z: np.ndarray,
        goal_z: np.ndarray,
    ) -> Optional[Tuple[List[int], np.ndarray]]:
        """
        Returns (node_indices_along_path, latent_path) or None if no path.
        """
        if self.tree is None or self.latents is None:
            raise RuntimeError("Call build() before plan().")

        _, (n_start,) = self.tree.query(start_z.reshape(1, -1), k=1)
        _, (n_goal,)  = self.tree.query(goal_z.reshape(1, -1),  k=1)
        n_start, n_goal = int(n_start), int(n_goal)

        if n_start == n_goal:
            return [n_start], self.latents[[n_start]]

        path = self._dijkstra(n_start, n_goal)
        if path is None:
            return None

        return path, self.latents[path]

    def get_actions_for_path(self, path: List[int]) -> List[Optional[np.ndarray]]:
        """Return the training action for each consecutive node pair in path."""
        actions = []
        for i in range(len(path) - 1):
            key = (path[i], path[i + 1])
            rev = (path[i + 1], path[i])
            if key in self.valid_transitions:
                actions.append(self.valid_transitions[key])
            elif rev in self.valid_transitions:
                actions.append(self.valid_transitions[rev])
            else:
                actions.append(None)
        return actions

    # ------------------------------------------------------------------
    # Persistence
    # ------------------------------------------------------------------

    def save(self, path: str) -> None:
        import pickle
        state = {
            "k": self.k,
            "latents": self.latents,
            "graph": self.graph,
            "valid_transitions": self.valid_transitions,
        }
        with open(path, "wb") as f:
            pickle.dump(state, f)

    @classmethod
    def load(cls, path: str) -> "LSR":
        import pickle
        with open(path, "rb") as f:
            state = pickle.load(f)
        lsr = cls(k=state["k"])
        lsr.latents = state["latents"]
        lsr.graph = state["graph"]
        lsr.valid_transitions = state["valid_transitions"]
        if lsr.latents is not None:
            lsr.tree = KDTree(lsr.latents)
        return lsr
