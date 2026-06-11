
from __future__ import annotations

import argparse
from typing import Any, Dict, List, Optional, Tuple

import torch

from .src.config import config_from_dict
from .train import _build_driftla_model
from .src.streaming.data import (
    load_ciao_chronological,
    load_gowala_chronological,
    load_ml1m_chronological,
    load_yelp_chronological,
)
from .src.streaming.graph_manager import StreamingGraphManager
from .src.metrics import evaluate_on_batch
def _load_streaming_split(
    data_root: str,
    dataset: str,
    init_ratio: float,
    stream_batches: int,
    max_interactions: Optional[int],
) -> Tuple[
    List[Tuple[int, int]],
    List[List[Tuple[int, int]]],
    Dict[Tuple[int, int], float],
    int,
    int,
]:
    n_batches = 10
    if dataset == "ciao":
        init_data, batches, timestamps, n_users, n_items = load_ciao_chronological(
            data_root, init_ratio=init_ratio, n_batches=n_batches,
        )
    elif dataset == "gowala":
        init_data, batches, timestamps, n_users, n_items = load_gowala_chronological(
            data_root,
            init_ratio=init_ratio,
            n_batches=n_batches,
            max_interactions=max_interactions,
        )
    elif dataset == "yelp":
        init_data, batches, timestamps, n_users, n_items = load_yelp_chronological(
            data_root,
            init_ratio=init_ratio,
            n_batches=n_batches,
            max_interactions=max_interactions,
        )
    else:
        init_data, batches, timestamps, n_users, n_items = load_ml1m_chronological(
            data_root, init_ratio=init_ratio, n_batches=n_batches,
        )
    if stream_batches < len(batches):
        batches = batches[:stream_batches]
    return init_data, batches, timestamps, n_users, n_items


def main() -> None:
    parser = argparse.ArgumentParser(description="DriftLA checkpoint evaluation (last batch)")
    parser.add_argument("--data_root", type=str, default="data")
    parser.add_argument("--checkpoint", type=str, required=True)
    parser.add_argument("--device", type=str, default="cpu", choices=["cpu", "cuda"])
    parser.add_argument(
        "--dataset",
        type=str,
        default="",
        choices=["", "ml-1m", "ciao", "gowala", "yelp"],
        help="Override dataset (default: read from checkpoint).",
    )
    args = parser.parse_args()

    try:
        ckpt: Dict[str, Any] = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
    except TypeError:
        ckpt = torch.load(args.checkpoint, map_location="cpu")
    cfg = config_from_dict(ckpt["config"])
    n_users_ckpt = int(ckpt["n_users"])
    n_items_ckpt = int(ckpt["n_items"])
    seed = int(ckpt.get("seed", 42))
    meta = ckpt.get("dataloader_meta") or {}
    dataset = (args.dataset or ckpt.get("dataset") or meta.get("dataset") or "ml-1m").lower()
    if dataset in ("ml1m",):
        dataset = "ml-1m"
    init_ratio = float(meta.get("init_ratio", 0.5))
    stream_batches = int(meta.get("stream_batches", 10))
    max_inter = meta.get("max_interactions")
    max_interactions = None if max_inter in (None, "") else int(max_inter)

    init_data, batches, timestamps, n_users, n_items = _load_streaming_split(
        args.data_root, dataset, init_ratio, stream_batches, max_interactions,
    )
    if n_users != n_users_ckpt or n_items != n_items_ckpt:
        raise ValueError(
            f"Checkpoint user/item counts ({n_users_ckpt}, {n_items_ckpt}) do not match "
            f"reloaded data ({n_users}, {n_items}). Check --data_root, --dataset, and split metadata."
        )

    train_list = list(init_data)
    for b in batches[:-1]:
        train_list.extend(b)
    test_batch = batches[-1]

    graph_mgr = StreamingGraphManager(n_users, n_items, k_hop=2)
    all_triples = [
        (u, i, timestamps.get((u, i), 0.0))
        for u, i in init_data + [e for bb in batches for e in bb]
    ]
    graph_mgr.add_edges(all_triples)
    adj = graph_mgr.get_adjacency().to(args.device)

    all_interactions = init_data + [e for bb in batches for e in bb]

    model = _build_driftla_model(cfg, n_users, n_items, seed)
    model.set_path_sampler(all_interactions, timestamps)
    model.load_state_dict(ckpt["model_state_dict"], strict=True)
    model = model.to(args.device)
    model.eval()

    with torch.no_grad():
        u_emb, i_emb = model(adj)
        metrics = evaluate_on_batch(u_emb, i_emb, test_batch, train_list)

    print("Last-batch metrics (train mask = all prior interactions):")
    for k, v in sorted(metrics.items()):
        print(f"  {k}: {v:.4f}")


if __name__ == "__main__":
    main()
