
import torch
import numpy as np
from collections import defaultdict
from typing import List, Tuple, Dict, Set, Optional


class SubgraphSample:

    __slots__ = ["center_node", "nodes", "edges", "edge_timestamps", "priority", "batch_id"]

    def __init__(
        self,
        center_node: int,
        nodes: Set[int],
        edges: List[Tuple[int, int]],
        edge_timestamps: Dict[Tuple[int, int], float],
        priority: float = 1.0,
        batch_id: int = 0,
    ):
        self.center_node = center_node
        self.nodes = nodes
        self.edges = edges
        self.edge_timestamps = edge_timestamps
        self.priority = priority
        self.batch_id = batch_id


class TopologyPreservingReplayBuffer:

    def __init__(self, max_size: int = 500, k_hop: int = 2):
        self.max_size = max_size
        self.k_hop = k_hop
        self._buffer: List[SubgraphSample] = []
        self._seen_count = 0

    def update(
        self,
        new_interactions: List[Tuple[int, int]],
        graph_neighbors: Dict[int, Set[int]],
        edge_timestamps: Dict[Tuple[int, int], float],
        batch_id: int = 0,
        n_users: int = 0,
        max_candidates: int = 50,
    ) -> None:
        users_in_batch = list(set(u for u, _ in new_interactions))
        if len(users_in_batch) > max_candidates:
            users_in_batch = list(np.random.choice(
                users_in_batch, size=max_candidates, replace=False,
            ))

        for u in users_in_batch:
            self._seen_count += 1
            subgraph = self._extract_subgraph(u, graph_neighbors, edge_timestamps)
            subgraph.batch_id = batch_id
            subgraph.priority = self._compute_priority(u, graph_neighbors)

            if len(self._buffer) < self.max_size:
                self._buffer.append(subgraph)
            else:
                j = np.random.randint(0, self._seen_count)
                if j < self.max_size:
                    min_idx = min(range(len(self._buffer)), key=lambda i: self._buffer[i].priority)
                    if subgraph.priority > self._buffer[min_idx].priority:
                        self._buffer[min_idx] = subgraph

    def _extract_subgraph(
        self,
        center: int,
        graph_neighbors: Dict[int, Set[int]],
        edge_timestamps: Dict[Tuple[int, int], float],
    ) -> SubgraphSample:
        visited: Set[int] = {center}
        frontier: Set[int] = {center}
        edges: List[Tuple[int, int]] = []
        sg_timestamps: Dict[Tuple[int, int], float] = {}

        for _ in range(self.k_hop):
            next_frontier: Set[int] = set()


            for n in sorted(frontier):
                for nb in sorted(graph_neighbors.get(n, set())):
                    edges.append((n, nb))
                    sg_timestamps[(n, nb)] = edge_timestamps.get((n, nb), 0.0)
                    if nb not in visited:
                        visited.add(nb)
                        next_frontier.add(nb)
            frontier = next_frontier

        return SubgraphSample(
            center_node=center,
            nodes=visited,
            edges=edges,
            edge_timestamps=sg_timestamps,
        )

    @staticmethod
    def _compute_priority(node: int, graph_neighbors: Dict[int, Set[int]]) -> float:
        return float(len(graph_neighbors.get(node, set())))

    def sample(self, n: int) -> List[SubgraphSample]:
        if not self._buffer:
            return []
        n = min(n, len(self._buffer))
        priorities = np.array([s.priority for s in self._buffer], dtype=np.float64)
        total = priorities.sum()
        if total == 0:
            probs = np.ones(len(self._buffer)) / len(self._buffer)
        else:
            probs = priorities / total
        indices = np.random.choice(len(self._buffer), size=n, replace=False, p=probs)
        return [self._buffer[i] for i in indices]

    def get_replay_paths(
        self,
        n_replay: int,
        path_sampler,
        num_paths_per_node: int = 5,
        alpha: int = 2,
        beta: int = 4,
        num_positive_paths: int = 3,
    ) -> Tuple[List[int], List[List[int]], Dict]:
        subgraphs = self.sample(n_replay)
        replay_nodes = []
        replay_paths = []

        for sg in subgraphs:
            center = sg.center_node
            replay_nodes.append(center)
            try:
                center_paths, _, _ = path_sampler.sample_paths_for_node(
                    center, num_paths_per_node, alpha, beta, num_positive_paths
                )
                for cp in center_paths:
                    replay_paths.append(cp)
            except Exception:
                continue

        return replay_nodes, replay_paths, {"n_subgraphs": len(subgraphs)}

    def __len__(self) -> int:
        return len(self._buffer)

    def stats(self) -> Dict:
        if not self._buffer:
            return {"size": 0, "seen": self._seen_count}
        priorities = [s.priority for s in self._buffer]
        return {
            "size": len(self._buffer),
            "seen": self._seen_count,
            "mean_priority": float(np.mean(priorities)),
            "max_priority": float(np.max(priorities)),
            "unique_batches": len(set(s.batch_id for s in self._buffer)),
        }
