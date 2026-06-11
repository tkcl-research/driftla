

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from collections import defaultdict

import numpy as np
import torch
import torch.optim as optim

_PECL_ROOT = os.path.dirname(os.path.abspath(__file__))
if _PECL_ROOT not in sys.path:
    sys.path.insert(0, _PECL_ROOT)


_STANDALONE_ROOT = os.path.abspath(os.path.join(_PECL_ROOT, "..", ".."))
if _STANDALONE_ROOT not in sys.path:
    sys.path.insert(0, _STANDALONE_ROOT)
from driftla.src.metrics import evaluate_on_batch as _driftla_batch_metrics

from data_loader import (
    create_negative_samples,
    load_ciao_chronological_streaming,
    load_gowala_chronological_streaming,
    load_ml1m_chronological_streaming,
    load_yelp_chronological_streaming,
)
from src.lightgcn import create_adjacency_matrix
from src.model import PECL
from src.utils import set_seed


def evaluate_on_batch(user_emb, item_emb, test_interactions, train_set):
    if not test_interactions:
        return {}
    return _driftla_batch_metrics(
        user_emb, item_emb, test_interactions, list(train_set)
    )


def train_one_epoch(
    model, interactions, user_pos, adj_matrix, n_items,
    batch_size, timestamps, device, optimizer,
    lambda1=0.1, lambda2=0.1, lambda3=1e-4,
):
    model.train()
    indices = np.random.permutation(len(interactions))
    n_batches = max(1, (len(indices) + batch_size - 1) // batch_size)
    total_loss = 0.0

    for b in range(n_batches):
        lo = b * batch_size
        hi = min(lo + batch_size, len(indices))
        batch_idx = indices[lo:hi]

        users_np = np.array([interactions[j][0] for j in batch_idx])
        items_np = np.array([interactions[j][1] for j in batch_idx])
        users = torch.tensor(users_np, dtype=torch.long, device=device)
        pos_items = torch.tensor(items_np, dtype=torch.long, device=device)
        neg_items_np = create_negative_samples(users_np, user_pos, n_items)
        neg_items = torch.tensor(neg_items_np, dtype=torch.long, device=device)

        optimizer.zero_grad()
        loss, _ = model.compute_total_loss(
            users, pos_items, neg_items, adj_matrix,
            lambda1, lambda2, lambda3, timestamps,
        )
        loss.backward()
        optimizer.step()
        total_loss += loss.item()

    return total_loss / n_batches


def run_streaming_pecl(
    init_data,
    batches,
    timestamps,
    n_users,
    n_items,
    device: torch.device,
    warmup_epochs: int = 10,
    batch_size: int = 2048,
    epochs_per_stream_batch: int = 10,
):
    if torch.cuda.is_available() and device.type == "cuda":
        torch.cuda.reset_peak_memory_stats()

    model = PECL(n_users, n_items, embed_dim=64, n_layers=3, alpha=2, beta=4, tau=0.05)
    model.set_path_sampler(init_data, timestamps)
    model = model.to(device)
    optimizer = optim.Adam(model.parameters(), lr=0.001)

    all_interactions = list(init_data)
    user_pos = defaultdict(set)
    for u, i in all_interactions:
        user_pos[u].add(i)

    adj = create_adjacency_matrix(n_users, n_items, all_interactions).to(device)

    t_warmup_start = time.time()
    for ep in range(1, warmup_epochs + 1):
        loss = train_one_epoch(
            model, all_interactions, user_pos, adj, n_items,
            batch_size, timestamps, device, optimizer,
        )
        print(f"  Warmup epoch {ep}: loss={loss:.4f}", flush=True)
    warmup_time = time.time() - t_warmup_start
    print(f"  Warmup time: {warmup_time:.1f}s", flush=True)

    results = []
    total_stream_time = 0.0

    for b_idx, batch in enumerate(batches):
        t_start = time.time()

        model.eval()
        with torch.no_grad():
            u_emb, i_emb = model(adj)
            metrics = evaluate_on_batch(u_emb, i_emb, batch, all_interactions)

        all_interactions.extend(batch)
        for u, i in batch:
            user_pos[u].add(i)

        t_rebuild = time.time()
        adj = create_adjacency_matrix(n_users, n_items, all_interactions).to(device)
        model.set_path_sampler(all_interactions, timestamps)
        rebuild_time = time.time() - t_rebuild

        t_train = time.time()
        ep_losses = []
        for _ in range(epochs_per_stream_batch):
            ep_losses.append(
                train_one_epoch(
                    model, all_interactions, user_pos, adj, n_items,
                    batch_size, timestamps, device, optimizer,
                )
            )
        loss = float(np.mean(ep_losses))
        train_time = time.time() - t_train

        step_time = time.time() - t_start
        total_stream_time += step_time

        peak_mb = (
            torch.cuda.max_memory_allocated() / 1024 / 1024
            if torch.cuda.is_available()
            else 0.0
        )

        results.append(
            {
                "batch": b_idx + 1,
                "n_interactions": len(batch),
                "metrics": {k: float(v) for k, v in metrics.items()},
                "loss": loss,
                "timings": {
                    "rebuild_adj_s": round(rebuild_time, 2),
                    "train_s": round(train_time, 2),
                    "total_s": round(step_time, 2),
                },
                "peak_gpu_mb": round(peak_mb, 1),
            }
        )

        print(
            f"\n  Batch {b_idx + 1}/{len(batches)}  R@10={metrics.get('Recall@10', 0):.4f}  "
            f"train={train_time:.1f}s  total={step_time:.1f}s",
            flush=True,
        )

    return {
        "method": "PECL streaming (full graph per batch)",
        "warmup_epochs": warmup_epochs,
        "epochs_per_stream_batch": epochs_per_stream_batch,
        "warmup_time_s": round(warmup_time, 1),
        "stream_time_s": round(total_stream_time, 1),
        "total_time_s": round(warmup_time + total_stream_time, 1),
        "peak_gpu_mb": round(
            torch.cuda.max_memory_allocated() / 1024 / 1024
            if torch.cuda.is_available()
            else 0.0,
            1,
        ),
        "batches": results,
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data_root", default="../data", help="Parent of dataset folders (default ../data from PECL/)")
    ap.add_argument(
        "--dataset",
        default="ml-1m",
        choices=("ml-1m", "ml-10m", "ml-20m", "amazon23", "ciao", "gowala", "yelp"),
        help="Chronological streaming split on this dataset.",
    )
    ap.add_argument(
        "--amz23_category",
        default="",
        help="Amazon Reviews'23 category (required for --dataset amazon23).",
    )
    ap.add_argument(
        "--kcore",
        type=int,
        default=0,
        help="Symmetric k-core for Gowalla/Yelp/Amazon'23 (0 = dataset default).",
    )
    ap.add_argument(
        "--max_interactions",
        type=int,
        default=300000,
        help="Cap earliest interactions for large/sparse datasets (<=0 disables cap).",
    )
    ap.add_argument("--n_batches", type=int, default=10)
    ap.add_argument("--warmup_epochs", type=int, default=3,
                    help="3×3 paper protocol: 3 (override for longer runs if needed).")
    ap.add_argument("--epochs_per_stream_batch", type=int, default=3,
                    help="3×3 paper protocol: 3 fine-tune epochs per arrival batch.")
    ap.add_argument("--batch_size", type=int, default=2048)
    ap.add_argument("--device", default="cuda", choices=("cuda", "cpu"))
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument(
        "--output",
        default="",
        help="Write JSON (default path depends on --dataset; see PECL/results/)",
    )
    args = ap.parse_args()

    set_seed(args.seed)
    device = torch.device(args.device if torch.cuda.is_available() or args.device == "cpu" else "cpu")
    if args.device == "cuda" and not torch.cuda.is_available():
        device = torch.device("cpu")
        print("CUDA unavailable, using CPU.", flush=True)

    data_root = os.path.abspath(args.data_root)
    if args.dataset in ("ml-10m", "ml-20m", "amazon23"):
        from baselines.common import load_chronological

        if args.dataset == "amazon23" and not (args.amz23_category or "").strip():
            ap.error("--amz23_category is required when --dataset amazon23")
        init_data, batches, timestamps, n_users, n_items = load_chronological(
            args.dataset,
            data_root,
            args.max_interactions,
            kcore=args.kcore,
            amz23_category=args.amz23_category,
        )
    elif args.dataset == "ml-1m":
        init_data, batches, timestamps, n_users, n_items = load_ml1m_chronological_streaming(
            data_root, init_ratio=0.5, n_batches=args.n_batches,
        )
    elif args.dataset == "ciao":
        init_data, batches, timestamps, n_users, n_items = load_ciao_chronological_streaming(
            data_root, init_ratio=0.5, n_batches=args.n_batches,
        )
    elif args.dataset == "gowala":
        max_interactions = None if args.max_interactions <= 0 else args.max_interactions
        init_data, batches, timestamps, n_users, n_items = load_gowala_chronological_streaming(
            data_root,
            init_ratio=0.5,
            n_batches=args.n_batches,
            max_interactions=max_interactions,
        )
    else:
        max_interactions = None if args.max_interactions <= 0 else args.max_interactions
        init_data, batches, timestamps, n_users, n_items = load_yelp_chronological_streaming(
            data_root,
            init_ratio=0.5,
            n_batches=args.n_batches,
            max_interactions=max_interactions,
        )
    print(
        f"Users={n_users} Items={n_items}  init={len(init_data)}  batches={[len(b) for b in batches]}",
        flush=True,
    )

    out = run_streaming_pecl(
        init_data,
        batches,
        timestamps,
        n_users,
        n_items,
        device,
        warmup_epochs=args.warmup_epochs,
        batch_size=args.batch_size,
        epochs_per_stream_batch=args.epochs_per_stream_batch,
    )
    out["dataset"] = args.dataset
    out["seed"] = int(args.seed)
    out["device"] = str(device)
    if out.get("batches"):
        out["avg_recall10"] = float(
            np.mean([b["metrics"]["Recall@10"] for b in out["batches"]])
        )
        out["avg_ndcg10"] = float(
            np.mean([b["metrics"]["NDCG@10"] for b in out["batches"]])
        )

    if args.output:
        out_path = args.output
    elif args.dataset == "ciao":
        out_path = os.path.join(_PECL_ROOT, "results", "streaming_pecl_ciao.json")
    elif args.dataset == "gowala":
        out_path = os.path.join(_PECL_ROOT, "results", "streaming_pecl_gowala.json")
    elif args.dataset == "yelp":
        out_path = os.path.join(_PECL_ROOT, "results", "streaming_pecl_yelp.json")
    else:
        out_path = os.path.join(_PECL_ROOT, "results", "streaming_pecl.json")
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    with open(out_path, "w") as f:
        json.dump(out, f, indent=2)
    print(f"\nWrote {out_path}", flush=True)

    r10 = np.mean([x["metrics"]["Recall@10"] for x in out["batches"]])
    n10 = np.mean([x["metrics"]["NDCG@10"] for x in out["batches"]])
    print(f"Avg R@10={r10:.4f}  Avg NDCG@10={n10:.4f}", flush=True)


if __name__ == "__main__":
    main()
