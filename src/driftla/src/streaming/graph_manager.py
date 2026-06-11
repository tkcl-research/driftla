
import torch
import numpy as np
from scipy.sparse import coo_matrix
from collections import defaultdict
from typing import List, Tuple, Set, Dict, Optional


class StreamingGraphManager:

    def __init__(self, n_users: int, n_items: int, k_hop: int = 2):
        self.n_users = n_users
        self.n_items = n_items
        self.total_nodes = n_users + n_items
        self.k_hop = k_hop

        self.neighbors: Dict[int, Set[int]] = defaultdict(set)
        self.edge_timestamps: Dict[Tuple[int, int], float] = {}
        self.degrees = np.zeros(self.total_nodes, dtype=np.float64)

        self._rows: List[int] = []
        self._cols: List[int] = []


        self._edge_ts_vec: List[float] = []

        self._adj_tensor: Optional[torch.Tensor] = None

        self._cached_interactions: Optional[List[Tuple[int, int]]] = None
        self._cached_timestamps: Optional[Dict[Tuple[int, int], float]] = None
        self._interactions_dirty = True

    def init_from_interactions(
        self,
        interactions: List[Tuple[int, int]],
        timestamps: Optional[Dict[Tuple[int, int], float]] = None,
    ) -> torch.Tensor:
        timestamps = timestamps or {}
        rows = []
        cols = []
        edge_ts: List[float] = []
        for u, i in interactions:
            item_nid = i + self.n_users
            self.neighbors[u].add(item_nid)
            self.neighbors[item_nid].add(u)
            ts = timestamps.get((u, i), 0.0)
            self.edge_timestamps[(u, item_nid)] = ts
            self.edge_timestamps[(item_nid, u)] = ts
            rows.extend([u, item_nid])
            cols.extend([item_nid, u])
            edge_ts.extend([ts, ts])

        self._rows = rows
        self._cols = cols
        self._edge_ts_vec = edge_ts

        self.degrees = np.array(
            [len(self.neighbors[n]) for n in range(self.total_nodes)], dtype=np.float64
        )
        self._interactions_dirty = True
        self._rebuild_full_adjacency()
        return self._adj_tensor

    def add_edges(
        self, edges: List[Tuple[int, int, float]]
    ) -> Tuple[torch.Tensor, Set[int]]:
        endpoint_nodes: Set[int] = set()
        new_rows = []
        new_cols = []

        for u, i, ts in edges:
            item_nid = i + self.n_users
            is_new = item_nid not in self.neighbors[u]

            self.neighbors[u].add(item_nid)
            self.neighbors[item_nid].add(u)
            self.edge_timestamps[(u, item_nid)] = ts
            self.edge_timestamps[(item_nid, u)] = ts

            if is_new:
                self.degrees[u] = len(self.neighbors[u])
                self.degrees[item_nid] = len(self.neighbors[item_nid])
                new_rows.extend([u, item_nid])
                new_cols.extend([item_nid, u])
                self._edge_ts_vec.extend([ts, ts])

            endpoint_nodes.add(u)
            endpoint_nodes.add(item_nid)

        self._rows.extend(new_rows)
        self._cols.extend(new_cols)
        self._interactions_dirty = True

        affected_nodes = self._k_hop_neighbors_batch(endpoint_nodes)

        self._rebuild_full_adjacency()
        return self._adj_tensor, affected_nodes

    def _k_hop_neighbors(self, node: int) -> Set[int]:
        visited: Set[int] = {node}
        frontier: Set[int] = {node}
        for _ in range(self.k_hop):
            next_frontier: Set[int] = set()
            for n in frontier:
                for nb in self.neighbors[n]:
                    if nb not in visited:
                        visited.add(nb)
                        next_frontier.add(nb)
            frontier = next_frontier
        return visited

    def _k_hop_neighbors_batch(self, seed_nodes: Set[int]) -> Set[int]:
        visited: Set[int] = set(seed_nodes)
        frontier: Set[int] = set(seed_nodes)
        for _ in range(self.k_hop):
            if len(visited) > self.total_nodes * 0.5:
                return set(range(self.total_nodes))
            next_frontier: Set[int] = set()
            for n in frontier:
                for nb in self.neighbors.get(n, set()):
                    if nb not in visited:
                        visited.add(nb)
                        next_frontier.add(nb)
            frontier = next_frontier
        return visited

    def _rebuild_full_adjacency(self) -> None:
        if not self._rows:
            self._adj_tensor = torch.sparse_coo_tensor(
                torch.zeros(2, 0, dtype=torch.long),
                torch.zeros(0),
                (self.total_nodes, self.total_nodes),
            )
            return

        rows_np = np.array(self._rows, dtype=np.int64)
        cols_np = np.array(self._cols, dtype=np.int64)
        vals = np.ones(len(rows_np), dtype=np.float64)

        adj = coo_matrix(
            (vals, (rows_np, cols_np)), shape=(self.total_nodes, self.total_nodes)
        )
        adj = adj.tocsr()
        adj.data[:] = 1.0
        adj = adj.tocoo()

        rowsum = np.array(adj.sum(1)).flatten()
        with np.errstate(divide="ignore"):
            d_inv_sqrt = np.power(rowsum, -0.5)
        d_inv_sqrt[np.isinf(d_inv_sqrt)] = 0.0

        d_mat = coo_matrix(
            (d_inv_sqrt, (np.arange(len(d_inv_sqrt)), np.arange(len(d_inv_sqrt)))),
            shape=adj.shape,
        )
        adj_norm = d_mat.dot(adj).dot(d_mat).tocoo()

        indices = torch.from_numpy(np.vstack([adj_norm.row, adj_norm.col]).astype(np.int64)).long()
        values = torch.from_numpy(adj_norm.data.astype(np.float32)).float()
        self._adj_tensor = torch.sparse_coo_tensor(
            indices, values, (self.total_nodes, self.total_nodes)
        )

    def get_adjacency(self) -> torch.Tensor:
        return self._adj_tensor

    def get_time_decayed_adjacency(
        self,
        now: float,
        lambda_e: float,
        ts_scale: float = 1.0,
        floor: float = 0.0,
    ) -> torch.Tensor:
        if not self._rows:
            return torch.sparse_coo_tensor(
                torch.zeros(2, 0, dtype=torch.long),
                torch.zeros(0),
                (self.total_nodes, self.total_nodes),
            )

        rows_np = np.array(self._rows, dtype=np.int64)
        cols_np = np.array(self._cols, dtype=np.int64)
        ts_np = np.array(self._edge_ts_vec, dtype=np.float64)

        scale = float(max(ts_scale, 1e-9))
        ages = (float(now) - ts_np) / scale

        ages = np.clip(ages, 0.0, None)
        weights = np.exp(-float(lambda_e) * ages)
        floor = float(np.clip(floor, 0.0, 0.999))
        if floor > 0.0:
            weights = floor + (1.0 - floor) * weights
        weights = np.clip(weights, 1e-6, 1.0).astype(np.float64)

        adj = coo_matrix(
            (weights, (rows_np, cols_np)),
            shape=(self.total_nodes, self.total_nodes),
        )

        adj = adj.tocsr()
        adj = adj.tocoo()

        rowsum = np.array(adj.sum(1)).flatten()
        with np.errstate(divide="ignore"):
            d_inv_sqrt = np.power(rowsum, -0.5)
        d_inv_sqrt[np.isinf(d_inv_sqrt)] = 0.0

        d_mat = coo_matrix(
            (d_inv_sqrt, (np.arange(len(d_inv_sqrt)), np.arange(len(d_inv_sqrt)))),
            shape=adj.shape,
        )
        adj_norm = d_mat.dot(adj).dot(d_mat).tocoo()

        indices = torch.from_numpy(
            np.vstack([adj_norm.row, adj_norm.col]).astype(np.int64)
        ).long()
        values = torch.from_numpy(adj_norm.data.astype(np.float32)).float()
        return torch.sparse_coo_tensor(
            indices, values, (self.total_nodes, self.total_nodes)
        )

    def get_neighbors(self, node: int) -> Set[int]:
        return self.neighbors[node]

    def get_sorted_neighbors(self, node: int) -> List[int]:
        return sorted(self.neighbors.get(node, set()))

    def get_sorted_edges(self) -> List[Tuple[int, int]]:
        out: List[Tuple[int, int]] = []
        for u in range(self.n_users):
            for item_nid in sorted(self.neighbors.get(u, set())):
                out.append((u, item_nid - self.n_users))
        return out

    def get_edge_timestamp(self, u: int, v: int) -> float:
        return self.edge_timestamps.get((u, v), 0.0)

    def get_all_interactions(self) -> List[Tuple[int, int]]:
        if self._interactions_dirty or self._cached_interactions is None:
            self._cached_interactions = []
            for u in range(self.n_users):
                for item_nid in sorted(self.neighbors.get(u, set())):
                    self._cached_interactions.append((u, item_nid - self.n_users))
            self._interactions_dirty = False
        return self._cached_interactions

    def get_timestamps_dict(self) -> Dict[Tuple[int, int], float]:
        if self._cached_timestamps is None or self._interactions_dirty:
            self._cached_timestamps = {}
            for u in range(self.n_users):
                for item_nid in sorted(self.neighbors.get(u, set())):
                    i = item_nid - self.n_users
                    self._cached_timestamps[(u, i)] = self.edge_timestamps.get((u, item_nid), 0.0)
        return self._cached_timestamps

    def num_edges(self) -> int:
        return sum(len(nbrs) for nbrs in self.neighbors.values()) // 2

    def summary(self) -> Dict:
        return {
            "n_users": self.n_users,
            "n_items": self.n_items,
            "n_edges": self.num_edges(),
            "avg_degree": float(self.degrees.mean()) if self.degrees.sum() > 0 else 0.0,
        }
