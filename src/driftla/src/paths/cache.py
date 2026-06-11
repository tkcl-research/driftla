
import time
import torch
import numpy as np
from collections import defaultdict
from typing import List, Tuple, Dict, Set, Optional


class CachedPath:

    __slots__ = [
        "path", "embedding_snapshot", "timestamp", "priority",
        "drift_score", "weight", "last_access", "created_at_batch",
    ]

    def __init__(
        self,
        path: List[int],
        embedding_snapshot: Optional[torch.Tensor] = None,
        created_at_batch: int = 0,
    ):
        self.path = path
        self.embedding_snapshot = embedding_snapshot
        self.timestamp = time.monotonic()
        self.priority = 1.0
        self.drift_score = 0.0
        self.weight = 1.0
        self.last_access = self.timestamp
        self.created_at_batch = created_at_batch


class DynamicPathCache:

    def __init__(
        self,
        max_paths_per_node: int = 100,
        drift_threshold_high: float = 0.5,
        drift_threshold_low: float = 0.1,
    ):
        self.max_paths_per_node = max_paths_per_node
        self.drift_threshold_high = drift_threshold_high
        self.drift_threshold_low = drift_threshold_low

        self._cache: Dict[int, List[CachedPath]] = defaultdict(list)
        self._stats = {"hits": 0, "misses": 0, "invalidations": 0, "evictions": 0}

        self.current_batch: int = 0

    def get_paths(self, node: int) -> List[CachedPath]:
        entries = self._cache.get(node, [])
        now = time.monotonic()
        for e in entries:
            e.last_access = now
        if entries:
            self._stats["hits"] += 1
        else:
            self._stats["misses"] += 1
        return entries

    def put_paths(
        self,
        node: int,
        paths: List[List[int]],
        embedding_snapshots: Optional[List[torch.Tensor]] = None,
    ) -> None:
        new_entries = []
        for idx, p in enumerate(paths):
            snap = embedding_snapshots[idx] if embedding_snapshots else None
            new_entries.append(CachedPath(p, snap, created_at_batch=self.current_batch))

        existing = self._cache.get(node, [])
        combined = existing + new_entries

        if len(combined) > self.max_paths_per_node:
            combined.sort(key=lambda e: e.priority, reverse=True)
            evicted = len(combined) - self.max_paths_per_node
            combined = combined[: self.max_paths_per_node]
            self._stats["evictions"] += evicted

        self._cache[node] = combined

    def invalidate_by_affected_nodes(
        self, affected_nodes: Set[int], total_nodes: int = 0,
    ) -> Dict[int, int]:
        if total_nodes > 0 and len(affected_nodes) > total_nodes * 0.5:
            total = sum(len(v) for v in self._cache.values())
            self._stats["invalidations"] += total
            self._cache.clear()
            return {}

        invalidated: Dict[int, int] = {}
        affected_frozen = frozenset(affected_nodes)
        for node, entries in list(self._cache.items()):
            before = len(entries)
            entries = [
                e for e in entries
                if not affected_frozen.intersection(e.path)
            ]
            removed = before - len(entries)
            if removed > 0:
                self._cache[node] = entries
                invalidated[node] = removed
                self._stats["invalidations"] += removed
        return invalidated

    def score_drift(
        self,
        node: int,
        encode_fn,
        all_emb: torch.Tensor,
        temporal_encoder=None,
        timestamps_dict: Optional[Dict] = None,
        n_users: int = 0,
    ) -> List[float]:
        entries = self._cache.get(node, [])
        drift_scores = []

        for entry in entries:
            if entry.embedding_snapshot is None:
                entry.drift_score = 0.0
                entry.weight = 1.0
                drift_scores.append(0.0)
                continue

            current_emb = encode_fn(
                entry.path, all_emb, temporal_encoder, timestamps_dict, n_users
            )
            cos_sim = torch.nn.functional.cosine_similarity(
                current_emb.unsqueeze(0),
                entry.embedding_snapshot.unsqueeze(0),
            ).item()
            drift = 1.0 - cos_sim
            entry.drift_score = drift

            if drift > self.drift_threshold_high:
                entry.weight = 0.0
            elif drift < self.drift_threshold_low:
                entry.weight = 1.0
            else:
                entry.weight = max(0.0, 1.0 - drift / self.drift_threshold_high)

            entry.priority = entry.weight * (1.0 / (1.0 + drift))
            drift_scores.append(drift)

        return drift_scores

    def invalidate_high_drift(self, node: int) -> int:
        entries = self._cache.get(node, [])
        before = len(entries)
        entries = [e for e in entries if e.drift_score <= self.drift_threshold_high]
        removed = before - len(entries)
        self._cache[node] = entries
        self._stats["invalidations"] += removed
        return removed

    def update_snapshots(self, node: int, new_snapshots: List[torch.Tensor]) -> None:
        entries = self._cache.get(node, [])
        for i, entry in enumerate(entries):
            if i < len(new_snapshots):
                entry.embedding_snapshot = new_snapshots[i].detach().clone()
                entry.drift_score = 0.0
                entry.weight = 1.0

    def get_weighted_paths(
        self,
        node: int,
        ttl_batches: Optional[int] = None,
        min_weight: float = 0.0,
    ) -> List[Tuple[List[int], float]]:
        entries = self._cache.get(node, [])
        if ttl_batches is None:
            return [(e.path, e.weight) for e in entries if e.weight > 0]

        cutoff = self.current_batch - ttl_batches
        out: List[Tuple[List[int], float]] = []
        for e in entries:
            if e.weight <= 0:
                continue
            if e.created_at_batch < cutoff and e.weight < min_weight:
                continue
            out.append((e.path, e.weight))
        return out

    def clear(self) -> None:
        self._cache.clear()

    def stats(self) -> Dict:
        total_cached = sum(len(v) for v in self._cache.values())
        return {**self._stats, "total_cached_paths": total_cached, "nodes_cached": len(self._cache)}
