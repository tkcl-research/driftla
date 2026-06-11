from __future__ import annotations

import argparse
import json
import os
import sys
from dataclasses import asdict, dataclass
from typing import Any, Dict, List, Tuple

import pandas as pd

_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)

from driftla.src.streaming.data import load_gowala_chronological, load_yelp_chronological


@dataclass(frozen=True)
class Meta:
    dataset: str
    data_root: str
    kcore: int
    max_interactions: int
    init_ratio: float
    n_batches: int
    min_stars: float
    n_users: int
    n_items: int
    n_total_interactions: int
    n_init: int
    n_stream: int
    stream_batch_sizes: List[int]


def _ensure_dir(path: str) -> None:
    os.makedirs(path, exist_ok=True)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", choices=("gowala", "yelp"), required=True)
    ap.add_argument("--data_root", default="data")
    ap.add_argument("--kcore", type=int, required=True)
    ap.add_argument("--max_interactions", type=int, default=300_000)
    ap.add_argument("--init_ratio", type=float, default=0.5)
    ap.add_argument("--n_batches", type=int, default=10)
    ap.add_argument("--min_stars", type=float, default=4.0)
    ap.add_argument("--out_dir", type=str, default="data/_materialized_dense")
    args = ap.parse_args()

    max_inter = None if args.max_interactions <= 0 else int(args.max_interactions)
    if args.dataset == "gowala":
        init, batches, ts, n_users, n_items = load_gowala_chronological(
            args.data_root,
            init_ratio=float(args.init_ratio),
            n_batches=int(args.n_batches),
            min_user_interactions=int(args.kcore),
            min_item_interactions=int(args.kcore),
            max_interactions=max_inter,
        )
        min_stars = 0.0
    else:
        init, batches, ts, n_users, n_items = load_yelp_chronological(
            args.data_root,
            init_ratio=float(args.init_ratio),
            n_batches=int(args.n_batches),
            min_user_interactions=int(args.kcore),
            min_item_interactions=int(args.kcore),
            max_interactions=max_inter,
            min_stars=float(args.min_stars),
        )
        min_stars = float(args.min_stars)

    all_edges: List[Tuple[int, int]] = list(init)
    for batch in batches:
        all_edges.extend(batch)

    rows: List[Tuple[int, int, float]] = []
    for user_id, item_id in all_edges:
        timestamp = float(ts.get((user_id, item_id), 0.0))
        rows.append((int(user_id), int(item_id), timestamp))

    df = pd.DataFrame(rows, columns=["user_id", "item_id", "timestamp"])
    df = df.sort_values("timestamp").reset_index(drop=True)

    tag = f"{args.dataset}_k{args.kcore}"
    out_dir = os.path.join(args.out_dir, tag)
    _ensure_dir(out_dir)

    out_csv = os.path.join(out_dir, "interactions.csv")
    df.to_csv(out_csv, index=False)

    meta = Meta(
        dataset=args.dataset,
        data_root=args.data_root,
        kcore=int(args.kcore),
        max_interactions=int(args.max_interactions),
        init_ratio=float(args.init_ratio),
        n_batches=int(args.n_batches),
        min_stars=min_stars,
        n_users=int(n_users),
        n_items=int(n_items),
        n_total_interactions=int(len(df)),
        n_init=int(len(init)),
        n_stream=int(len(df) - len(init)),
        stream_batch_sizes=[int(len(batch)) for batch in batches],
    )
    with open(os.path.join(out_dir, "meta.json"), "w", encoding="utf-8") as handle:
        json.dump(asdict(meta), handle, indent=2)

    print(f"Wrote {out_csv}")
    print(f"Users={meta.n_users} Items={meta.n_items} Interactions={meta.n_total_interactions}")


if __name__ == "__main__":
    main()
