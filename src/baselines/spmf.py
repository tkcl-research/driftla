
from __future__ import annotations

import time
from collections import defaultdict
from typing import Any, Dict, List, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

from driftla.src.metrics import evaluate_on_batch
from driftla.src.streaming.data import create_negative_samples
from driftla.src.utils import set_seed


class SPMF(nn.Module):

    def __init__(self, n_users: int, n_items: int, embed_dim: int = 64):
        super().__init__()
        self.user_emb = nn.Embedding(n_users, embed_dim)
        self.item_emb = nn.Embedding(n_items, embed_dim)
        self.user_bias = nn.Embedding(n_users, 1)
        self.item_bias = nn.Embedding(n_items, 1)
        nn.init.normal_(self.user_emb.weight, std=0.1)
        nn.init.normal_(self.item_emb.weight, std=0.1)
        nn.init.zeros_(self.user_bias.weight)
        nn.init.zeros_(self.item_bias.weight)

    def score(self, u: torch.Tensor, i: torch.Tensor) -> torch.Tensor:
        return (
            (self.user_emb(u) * self.item_emb(i)).sum(-1)
            + self.user_bias(u).squeeze(-1)
            + self.item_bias(i).squeeze(-1)
        )

    def all_user_item_emb(self) -> Tuple[torch.Tensor, torch.Tensor]:
        return self.user_emb.weight, self.item_emb.weight

    def bpr(self, u, pos, neg, reg=1e-5):
        ps = self.score(u, pos)
        ns = self.score(u, neg)
        loss = F.softplus(ns - ps).mean()
        if reg > 0:
            loss = loss + reg * (
                self.user_emb(u).norm(2).pow(2)
                + self.item_emb(pos).norm(2).pow(2)
                + self.item_emb(neg).norm(2).pow(2)
            ) / max(1, len(u))
        return loss


def run_spmf_streaming(
    init_data: List[Tuple[int, int]],
    batches: List[List[Tuple[int, int]]],
    n_users: int,
    n_items: int,
    device: torch.device,
    embed_dim: int = 64,
    warmup_epochs: int = 3,
    streaming_passes: int = 3,
    history_replay: int = 4096,
    batch_size: int = 2048,
    lr: float = 1e-3,
    seed: int = 42,
) -> Dict[str, Any]:
    set_seed(seed)
    rng = np.random.default_rng(seed)
    model = SPMF(n_users, n_items, embed_dim).to(device)
    opt = optim.Adam(model.parameters(), lr=lr)

    user_pos: Dict[int, set] = defaultdict(set)
    for u, i in init_data:
        user_pos[u].add(i)
    history = list(init_data)

    def _train_pass(triples: List[Tuple[int, int]]):
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
            loss = model.bpr(u_t, p_t, n_t)
            loss.backward()
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
        ts = time.time()
        model.eval()
        with torch.no_grad():
            u_emb, i_emb = model.all_user_item_emb()
            metrics = evaluate_on_batch(u_emb, i_emb, batch, history)

        history_extra: List[Tuple[int, int]] = []
        if history_replay > 0 and history:
            k = min(history_replay, len(history))
            sel = rng.choice(len(history), size=k, replace=False)
            history_extra = [history[i] for i in sel]
        train_set = list(batch) + history_extra

        history.extend(batch)
        for u, i in batch:
            user_pos[u].add(i)

        model.train()
        for _ in range(streaming_passes):
            _train_pass(train_set)
        stream_time += time.time() - ts
        results.append(
            {"batch": b_idx + 1, "metrics": {k: float(v) for k, v in metrics.items()}}
        )
        print(
            f"  [SPMF] batch {b_idx + 1} R@10={metrics.get('Recall@10', 0):.4f} "
            f"NDCG@10={metrics.get('NDCG@10', 0):.4f}",
            flush=True,
        )

    r10 = float(np.mean([x["metrics"].get("Recall@10", 0) for x in results]))
    n10 = float(np.mean([x["metrics"].get("NDCG@10", 0) for x in results]))
    return {
        "method": "SPMF",
        "warmup_time_s": round(warmup_time, 1),
        "stream_time_s": round(stream_time, 1),
        "total_time_s": round(warmup_time + stream_time, 1),
        "avg_recall10": r10,
        "avg_ndcg10": n10,
        "batches": results,
        "seed": seed,
    }
