
import numpy as np
import random as _random
from collections import defaultdict
from typing import Optional


class PathSampler:

    def __init__(self, n_users, n_items, interactions, timestamps=None,
                 restart_prob=0.3, path_length=5, seed: Optional[int] = None):
        self.n_users = n_users
        self.n_items = n_items
        self.restart_prob = restart_prob
        self.path_length = path_length
        self.timestamps = timestamps if timestamps else {}


        self._rng = np.random.default_rng(seed)
        self._py_rng = _random.Random(seed)


        self.neighbors = defaultdict(list)
        self.edge_ts = {}

        for u, i in interactions:
            item_nid = i + n_users
            self.neighbors[u].append(item_nid)
            self.neighbors[item_nid].append(u)
            ts = self.timestamps.get((u, i), 0.0)
            self.edge_ts[(u, item_nid)] = ts
            self.edge_ts[(item_nid, u)] = ts


    def random_walk_with_restart(self, start_node, num_paths=10):
        paths = []
        for _ in range(num_paths):
            path = [start_node]
            current = start_node
            while len(path) < self.path_length:
                if self._py_rng.random() < self.restart_prob:
                    current = start_node
                    path = [start_node]
                else:
                    nbrs = self.neighbors[current]
                    if not nbrs:
                        break
                    current = self._py_rng.choice(nbrs)
                    path.append(current)
            if len(path) >= 2:
                paths.append(path)
        return paths


    @staticmethod
    def get_positive_nodes(paths, alpha=2):
        counts = defaultdict(int)
        for p in paths:
            seen = set()
            for n in p:
                if n not in seen:
                    counts[n] += 1
                    seen.add(n)
        return {n for n, c in counts.items() if c >= alpha}


    def target_guided_random_walk(self, center_path, beta=4, num_positive_paths=5):
        positive_paths = []
        for _ in range(num_positive_paths):
            det_len = min(beta, len(center_path))
            p1 = center_path[:det_len]

            current = p1[-1]
            p2 = []
            remaining = self.path_length - len(p1)
            for _ in range(remaining):
                nbrs = self.neighbors[current]
                if not nbrs:
                    break
                current = self._temporal_sample(current, nbrs)
                p2.append(current)
            full = p1 + p2
            if len(full) >= 2:
                positive_paths.append(full)
        return positive_paths

    def _temporal_sample(self, current, neighbors):
        if not self.timestamps or len(neighbors) == 1:
            return self._py_rng.choice(neighbors)

        scores = []
        for nbr in neighbors:
            ts = self.edge_ts.get((current, nbr), 0.0)
            scores.append(ts + 1.0)

        scores = np.array(scores, dtype=np.float64)
        scores = scores - scores.max()
        exp_scores = np.exp(scores)
        probs = exp_scores / exp_scores.sum()
        return self._rng.choice(neighbors, p=probs)

    def sample_paths_for_node(self, node, num_center_paths=10, alpha=2,
                              beta=4, num_positive_paths=5):
        center_paths = self.random_walk_with_restart(node, num_center_paths)
        positive_nodes = self.get_positive_nodes(center_paths, alpha)

        positive_paths_dict = {}
        for cp in center_paths:
            pos_paths = self.target_guided_random_walk(cp, beta, num_positive_paths)
            positive_paths_dict[tuple(cp)] = pos_paths

        return center_paths, positive_paths_dict, positive_nodes
