
from __future__ import annotations

import time
from collections import defaultdict
from typing import Any, Dict, List, Optional, Set, Tuple

import numpy as np
import torch
import torch.optim as optim

from driftla.src.encoder.lightgcn import LightGCN
from driftla.src.metrics import evaluate_on_batch
from driftla.src.streaming.data import create_negative_samples
from driftla.src.streaming.graph_manager import StreamingGraphManager
from driftla.src.utils import set_seed


def _affected_replay(
    batch: List[Tuple[int, int]],
    history: List[Tuple[int, int]],
    user_items: Dict[int, Set[int]],
    item_users: Dict[int, Set[int]],
    max_replay: int,
    rng: np.random.Generator,
) -> List[Tuple[int, int]]:
    if not history or max_replay <= 0:
        return []

    affected_users: Set[int] = set()
    affected_items: Set[int] = set()
    for u, i in batch:
        affected_users.add(u)
        affected_items.add(i)
        affected_users.update(item_users.get(i, set()))
        affected_items.update(user_items.get(u, set()))

    candidates: List[Tuple[int, int]] = []
    for u, i in history:
        if u in affected_users or i in affected_items:
            candidates.append((u, i))

    if not candidates:
        k = min(max_replay, len(history))
        sel = rng.choice(len(history), size=k, replace=False)
        return [history[i] for i in sel]

    if len(candidates) <= max_replay:
        return candidates
    sel = rng.choice(len(candidates), size=max_replay, replace=False)
    return [candidates[i] for i in sel]


def run_ergnn_warmstart(
    init_data: List[Tuple[int, int]],
    batches: List[List[Tuple[int, int]]],
    timestamps: Optional[Dict[Tuple[int, int], float]],
    n_users: int,
    n_items: int,
    device: torch.device,
    embed_dim: int = 64,
    n_layers: int = 3,
    warmup_epochs: int = 3,
    streaming_passes: int = 3,
    history_replay: int = 4096,
    batch_size: int = 2048,
    lr: float = 1e-3,
    seed: int = 42,
) -> Dict[str, Any]:
    set_seed(seed)
    rng = np.random.default_rng(seed)
    ts_map = timestamps or {}

    model = LightGCN(n_users, n_items, embed_dim=embed_dim, n_layers=n_layers).to(device)
    opt = optim.Adam(model.parameters(), lr=lr)

    user_pos: Dict[int, set] = defaultdict(set)
    user_items: Dict[int, Set[int]] = defaultdict(set)
    item_users: Dict[int, Set[int]] = defaultdict(set)
    for u, i in init_data:
        user_pos[u].add(i)
        user_items[u].add(i)
        item_users[i].add(u)
    history = list(init_data)

    graph_mgr = StreamingGraphManager(n_users, n_items, k_hop=1)
    init_triples = [(u, i, ts_map.get((u, i), 0.0)) for u, i in init_data]
    graph_mgr.add_edges(init_triples)
    adj = graph_mgr.get_adjacency().to(device)

    def _train_pass(triples: List[Tuple[int, int]]) -> float:
        idx = rng.permutation(len(triples))
        n_mb = max(1, (len(idx) + batch_size - 1) // batch_size)
        total = 0.0
        for b in range(n_mb):
            sub = idx[b * batch_size: (b + 1) * batch_size]
            us = np.array([triples[j][0] for j in sub])
            ps = np.array([triples[j][1] for j in sub])
            ns = create_negative_samples(us, user_pos, n_items)
            u_t = torch.tensor(us, dtype=torch.long, device=device)
            p_t = torch.tensor(ps, dtype=torch.long, device=device)
            n_t = torch.tensor(ns, dtype=torch.long, device=device)
            opt.zero_grad()
            loss, reg = model.bpr_loss(u_t, p_t, n_t, adj)
            (loss + reg).backward()
            opt.step()
            total += float(loss.detach())
        return total / max(1, n_mb)

    t0 = time.time()
    for _ in range(warmup_epochs):
        _train_pass(history)
    warmup_time = time.time() - t0

    results = []
    stream_time = 0.0

    for b_idx, batch in enumerate(batches):
        t_step = time.time()

        model.eval()
        with torch.no_grad():
            u_emb, i_emb = model(adj)
            metrics = evaluate_on_batch(u_emb, i_emb, batch, history)

        new_triples = [(u, i, ts_map.get((u, i), 0.0)) for u, i in batch]
        graph_mgr.add_edges(new_triples)
        adj = graph_mgr.get_adjacency().to(device)

        extra = _affected_replay(
            batch, history, user_items, item_users, history_replay, rng,
        )
        train_set = list(batch) + extra

        history.extend(batch)
        for u, i in batch:
            user_pos[u].add(i)
            user_items[u].add(i)
            item_users[i].add(u)

        model.train()
        for _ in range(streaming_passes):
            _train_pass(train_set)

        stream_time += time.time() - t_step
        results.append(
            {"batch": b_idx + 1, "metrics": {k: float(v) for k, v in metrics.items()}}
        )
        print(
            f"  [ERGNN-WS] batch {b_idx + 1} R@10={metrics.get('Recall@10', 0):.4f} "
            f"NDCG@10={metrics.get('NDCG@10', 0):.4f}",
            flush=True,
        )

    r10 = float(np.mean([x["metrics"].get("Recall@10", 0) for x in results]))
    n10 = float(np.mean([x["metrics"].get("NDCG@10", 0) for x in results]))
    r20 = float(np.mean([x["metrics"].get("Recall@20", 0) for x in results]))
    n20 = float(np.mean([x["metrics"].get("NDCG@20", 0) for x in results]))
    return {
        "method": "ERGNN warm-start",
        "warmup_time_s": round(warmup_time, 1),
        "stream_time_s": round(stream_time, 1),
        "total_time_s": round(warmup_time + stream_time, 1),
        "avg_recall10": r10,
        "avg_ndcg10": n10,
        "avg_recall20": r20,
        "avg_ndcg20": n20,
        "batches": results,
        "seed": seed,
        "hyperparams": {"history_replay": history_replay},
    }
