
from __future__ import annotations

import argparse
import json
import os
import sys
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

from .src.config import preset_config
from .src.streaming.data import load_ml1m_chronological


def _load_pecl_cache(path: str) -> Optional[Dict[str, Any]]:
    if not os.path.isfile(path):
        return None
    with open(path) as f:
        return json.load(f)


def _summarize_run(d: Dict[str, Any]) -> Dict[str, float]:
    bs = d.get("batches", [])
    if not bs:
        return {}
    return {
        "avg_recall10": float(np.mean([x["metrics"]["Recall@10"] for x in bs])),
        "avg_ndcg10": float(np.mean([x["metrics"]["NDCG@10"] for x in bs])),
        "stream_time_s": float(d.get("stream_time_s", sum(x["timings"]["total_s"] for x in bs))),
        "total_time_s": float(d.get("total_time_s", 0)),
        "warmup_time_s": float(d.get("warmup_time_s", 0)),
    }


def main() -> None:
    ap = argparse.ArgumentParser(description="DriftLA streaming benchmark vs PECL cache")
    ap.add_argument("--data_root", default="data")
    ap.add_argument("--device", default="cuda", choices=("cuda", "cpu"))
    ap.add_argument(
        "--seeds",
        default="42,43,44",
        help="Comma-separated training seeds (split order fixed by data loader).",
    )
    ap.add_argument(
        "--pecl_cache",
        default="PECL/results/streaming_pecl.json",
        help="Cached PECL streaming JSON (full-graph per batch).",
    )
    ap.add_argument(
        "--out_json",
        default="driftla/streaming_paper_results.json",
        help="Write aggregate + per-seed DriftLA runs here.",
    )
    ap.add_argument(
        "--preset",
        default="final",
        choices=("final", "paper_100_batch", "paper_100_warmup", "paper_100_both"),
        help="DriftLA streaming preset (see driftla/README.md).",
    )
    args = ap.parse_args()

    seeds = [int(s.strip()) for s in args.seeds.split(",") if s.strip()]
    if not seeds:
        seeds = [42]


    from .train import run_experiment

    init_data, batches, timestamps, n_users, n_items = load_ml1m_chronological(
        args.data_root, init_ratio=0.5, n_batches=10,
    )

    import torch

    device = torch.device(
        "cuda" if args.device == "cuda" and torch.cuda.is_available() else "cpu",
    )
    if args.device == "cuda" and not torch.cuda.is_available():
        print("CUDA requested but not available; using CPU.", file=sys.stderr)

    cfg = preset_config(args.preset)
    driftla_runs: List[Dict[str, Any]] = []

    for seed in seeds:
        print(f"\n{'='*60}\nDriftLA seed={seed} device={device}\n{'='*60}")
        summary, _model = run_experiment(
            cfg, init_data, batches, timestamps, n_users, n_items, device, seed=seed,
        )
        driftla_runs.append({"seed": seed, **summary})

    r10 = np.array([r["avg_recall10"] for r in driftla_runs])
    n10 = np.array([r["avg_ndcg10"] for r in driftla_runs])
    st = np.array([r["stream_time_s"] for r in driftla_runs])

    pecl_raw = _load_pecl_cache(args.pecl_cache)
    pecl_sum = _summarize_run(pecl_raw) if pecl_raw else None

    out: Dict[str, Any] = {
        "protocol": {
            "dataset": "ml-1m implicit >=4",
            "init_ratio": 0.5,
            "n_streaming_batches": 10,
            "warmup_epochs": cfg.warmup_epochs,
            "streaming_passes_per_batch": cfg.streaming_passes,
            "batch_size": cfg.batch_size,
            "eval": "prequential (test-before-train per batch)",
            "driftla_preset": args.preset,
            "driftla_config": cfg.to_json_dict(),
        },
        "pecl_baseline": {
            "source": args.pecl_cache if pecl_raw else None,
            "note": "Single cached CUDA run (seed 42 training); full-graph retrain per batch.",
            "summary": pecl_sum,
        },
        "driftla": {
            "seeds": seeds,
            "per_seed": driftla_runs,
            "mean_std": {
                "avg_recall10_mean": float(r10.mean()),
                "avg_recall10_std": float(r10.std(ddof=1)) if len(r10) > 1 else 0.0,
                "avg_ndcg10_mean": float(n10.mean()),
                "avg_ndcg10_std": float(n10.std(ddof=1)) if len(n10) > 1 else 0.0,
                "stream_time_s_mean": float(st.mean()),
                "stream_time_s_std": float(st.std(ddof=1)) if len(st) > 1 else 0.0,
            },
        },
    }

    if pecl_sum and len(r10):
        pr = pecl_sum["avg_recall10"]
        out["comparison_vs_pecl_cache"] = {
            "recall10_mean_delta_vs_pecl": float(r10.mean() - pr),
            "recall10_pct_vs_pecl": float(100.0 * (r10.mean() - pr) / (pr + 1e-9)),
            "ndcg10_mean_delta_vs_pecl": float(n10.mean() - pecl_sum["avg_ndcg10"]),
        }

    os.makedirs(os.path.dirname(args.out_json) or ".", exist_ok=True)
    with open(args.out_json, "w") as f:
        json.dump(out, f, indent=2)

    print("\n" + "=" * 60)
    print("SUMMARY (streaming protocol)")
    print("=" * 60)
    if pecl_sum:
        print(
            f"PECL (cache)  R@10={pecl_sum['avg_recall10']:.4f}  NDCG@10={pecl_sum['avg_ndcg10']:.4f}  "
            f"stream_s={pecl_sum['stream_time_s']:.0f}"
        )
    print(
        f"DriftLA seeds {seeds}  R@10={r10.mean():.4f}±{r10.std(ddof=1) if len(r10)>1 else 0:.4f}  "
        f"NDCG@10={n10.mean():.4f}±{n10.std(ddof=1) if len(n10)>1 else 0:.4f}  "
        f"stream_s={st.mean():.0f}±{st.std(ddof=1) if len(st)>1 else 0:.0f}"
    )
    print(f"Wrote {args.out_json}")
    print("=" * 60)


if __name__ == "__main__":
    main()
