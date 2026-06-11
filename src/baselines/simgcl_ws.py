
from __future__ import annotations

import os
import time
from collections import defaultdict
from typing import TYPE_CHECKING, Any, Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.nn.functional as F
import torch.optim as optim

from driftla.src.encoder.lightgcn import LightGCN
from driftla.src.metrics import evaluate_on_batch
from driftla.src.streaming.data import create_negative_samples
from driftla.src.streaming.graph_manager import StreamingGraphManager
from driftla.src.utils import set_seed


def _simgcl_loss(
    h1: torch.Tensor,
    h2: torch.Tensor,
    temperature: float = 0.2,
) -> torch.Tensor:
    h1 = F.normalize(h1, dim=-1)
    h2 = F.normalize(h2, dim=-1)

    sim = torch.mm(h1, h2.t()) / temperature

    labels = torch.arange(h1.size(0), device=h1.device)
    loss = F.cross_entropy(sim, labels)
    return loss


def _eval_simgcl_on_val(
    model: "LightGCN",
    val_tail: List[Tuple[int, int]],
    history: List[Tuple[int, int]],
    adj: "torch.Tensor",
    device: "torch.device",
) -> float:
    model.eval()
    with torch.no_grad():
        u_emb, i_emb = model(adj)
        metrics = evaluate_on_batch(u_emb, i_emb, val_tail, history)
    model.train()
    return float(metrics.get("Recall@10", 0.0))


def _tune_simgcl_on_val(
    init_data: List[Tuple[int, int]],
    val_tail: List[Tuple[int, int]],
    n_users: int,
    n_items: int,
    device: "torch.device",
    embed_dim: int,
    n_layers: int,
    warmup_epochs: int,
    history_replay: int,
    batch_size: int,
    lr: float,
    seed: int,
) -> Dict[str, float]:
    if os.getenv("REVISION_FAST", "").strip() in ("1", "true", "True", "yes"):

        eps_grid = [0.1, 0.2]
        tau_grid = [0.2, 0.3]
        lambda_cl_grid = [0.1, 0.2]
    else:
        eps_grid = [0.1, 0.2, 0.3]
        tau_grid = [0.1, 0.2, 0.3]
        lambda_cl_grid = [0.1, 0.2, 0.5]

    best_r10  = -1.0
    best_hp: Dict[str, float] = {"eps": 0.2, "temperature": 0.2, "lambda_cl": 0.2}

    user_pos_val: Dict[int, set] = defaultdict(set)
    for u, i in init_data:
        user_pos_val[u].add(i)

    for eps in eps_grid:
        for tau in tau_grid:
            for lam in lambda_cl_grid:
                set_seed(seed)
                rng_g = np.random.default_rng(seed)
                m_tmp = LightGCN(n_users, n_items, embed_dim=embed_dim, n_layers=n_layers).to(device)
                opt_tmp = optim.Adam(m_tmp.parameters(), lr=lr)
                gm_tmp = StreamingGraphManager(n_users, n_items, k_hop=1)
                ts_map_tmp: Dict[Tuple[int, int], float] = {}
                gm_tmp.add_edges([(u, i, 0.0) for u, i in init_data])
                adj_tmp = gm_tmp.get_adjacency().to(device)

                def _aug_tmp(h: torch.Tensor) -> torch.Tensor:
                    return h + torch.empty_like(h).uniform_(-eps, eps)

                for _ in range(warmup_epochs):
                    idx = rng_g.permutation(len(init_data))
                    n_mb = max(1, (len(idx) + batch_size - 1) // batch_size)
                    for b in range(n_mb):
                        sub = idx[b * batch_size: (b + 1) * batch_size]
                        us = np.array([init_data[j][0] for j in sub])
                        ps = np.array([init_data[j][1] for j in sub])
                        ns = create_negative_samples(us, user_pos_val, n_items)
                        u_t = torch.tensor(us, dtype=torch.long, device=device)
                        p_t = torch.tensor(ps, dtype=torch.long, device=device)
                        n_t = torch.tensor(ns, dtype=torch.long, device=device)
                        opt_tmp.zero_grad()
                        bpr_l, reg = m_tmp.bpr_loss(u_t, p_t, n_t, adj_tmp)
                        cl_l = torch.tensor(0.0, device=device)
                        if lam > 0:
                            u_all, i_all = m_tmp(adj_tmp)
                            u_v1 = _aug_tmp(u_all[u_t])
                            u_v2 = _aug_tmp(u_all[u_t])
                            i_v1 = _aug_tmp(i_all[p_t])
                            i_v2 = _aug_tmp(i_all[p_t])
                            cl_l = 0.5 * (
                                _simgcl_loss(u_v1, u_v2, tau)
                                + _simgcl_loss(i_v1, i_v2, tau)
                            )
                        (bpr_l + reg + lam * cl_l).backward()
                        opt_tmp.step()

                r10 = _eval_simgcl_on_val(m_tmp, val_tail, list(init_data), adj_tmp, device)
                if r10 > best_r10:
                    best_r10 = r10
                    best_hp = {"eps": eps, "temperature": tau, "lambda_cl": lam}
                del m_tmp, opt_tmp

    print(
        f"[SimGCL-WS val-tune] best hp={best_hp} val_R@10={best_r10:.4f}",
        flush=True,
    )
    return best_hp


def run_simgcl_warmstart(
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

    eps: float = 0.2,
    lambda_cl: float = 0.2,
    temperature: float = 0.2,

    val_tail: Optional[List[Tuple[int, int]]] = None,
) -> Dict[str, Any]:

    if val_tail is not None and len(val_tail) > 0:
        best_hp = _tune_simgcl_on_val(
            init_data, val_tail, n_users, n_items, device,
            embed_dim=embed_dim, n_layers=n_layers,
            warmup_epochs=warmup_epochs, history_replay=history_replay,
            batch_size=batch_size, lr=lr, seed=seed,
        )
        eps         = best_hp["eps"]
        temperature = best_hp["temperature"]
        lambda_cl   = best_hp["lambda_cl"]

    set_seed(seed)
    rng = np.random.default_rng(seed)

    model = LightGCN(n_users, n_items, embed_dim=embed_dim, n_layers=n_layers).to(device)
    opt = optim.Adam(model.parameters(), lr=lr)

    user_pos: Dict[int, set] = defaultdict(set)
    for u, i in init_data:
        user_pos[u].add(i)
    history = list(init_data)

    graph_mgr = StreamingGraphManager(n_users, n_items, k_hop=1)
    ts_map = timestamps or {}
    init_triples = [(u, i, ts_map.get((u, i), 0.0)) for u, i in init_data]
    graph_mgr.add_edges(init_triples)
    adj = graph_mgr.get_adjacency().to(device)

    def _augment(h: torch.Tensor) -> torch.Tensor:
        noise = torch.empty_like(h).uniform_(-eps, eps)
        return h + noise

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


            bpr_loss, reg = model.bpr_loss(u_t, p_t, n_t, adj)


            cl_loss = torch.tensor(0.0, device=device)
            if lambda_cl > 0:
                u_emb_all, i_emb_all = model(adj)

                u_batch = u_emb_all[u_t]
                u_v1 = _augment(u_batch)
                u_v2 = _augment(u_batch)

                i_batch = i_emb_all[p_t]
                i_v1 = _augment(i_batch)
                i_v2 = _augment(i_batch)
                cl_loss = 0.5 * (
                    _simgcl_loss(u_v1, u_v2, temperature)
                    + _simgcl_loss(i_v1, i_v2, temperature)
                )

            loss = bpr_loss + reg + lambda_cl * cl_loss
            loss.backward()
            opt.step()
            total += float(bpr_loss.detach())
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

        extra: List[Tuple[int, int]] = []
        if history_replay > 0 and history:
            k = min(history_replay, len(history))
            sel = rng.choice(len(history), size=k, replace=False)
            extra = [history[i] for i in sel]
        train_set = list(batch) + extra

        history.extend(batch)
        for u, i in batch:
            user_pos[u].add(i)

        model.train()
        for _ in range(streaming_passes):
            _train_pass(train_set)

        step_time = time.time() - t_step
        stream_time += step_time
        results.append(
            {"batch": b_idx + 1, "metrics": {k: float(v) for k, v in metrics.items()}}
        )
        print(
            f"  [SimGCL-WS] batch {b_idx + 1} R@10={metrics.get('Recall@10', 0):.4f} "
            f"NDCG@10={metrics.get('NDCG@10', 0):.4f}",
            flush=True,
        )

    r10 = float(np.mean([x["metrics"].get("Recall@10", 0) for x in results]))
    n10 = float(np.mean([x["metrics"].get("NDCG@10", 0) for x in results]))
    r20 = float(np.mean([x["metrics"].get("Recall@20", 0) for x in results]))
    n20 = float(np.mean([x["metrics"].get("NDCG@20", 0) for x in results]))
    return {
        "method": "SimGCL warm-start",
        "warmup_time_s": round(warmup_time, 1),
        "stream_time_s": round(stream_time, 1),
        "total_time_s": round(warmup_time + stream_time, 1),
        "avg_recall10": r10,
        "avg_ndcg10": n10,
        "avg_recall20": r20,
        "avg_ndcg20": n20,
        "batches": results,
        "seed": seed,
        "hyperparams": {
            "eps": eps,
            "lambda_cl": lambda_cl,
            "temperature": temperature,
        },
    }
