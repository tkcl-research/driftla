
from __future__ import annotations

from pathlib import Path as _Path
_REPO_ROOT = _Path(__file__).resolve().parents[2]
import argparse
import json
import os
import subprocess
import sys

import torch

from baselines.common import load_chronological
from baselines.lightgcn_ws import run_lightgcn_warmstart
from baselines.lightgcn_window_retrain import run_lightgcn_window_retrain
from baselines.graphsail_ws import run_graphsail_warmstart
from baselines.ergnn_ws import run_ergnn_warmstart
from baselines.simgcl_ws import run_simgcl_warmstart
from baselines.spmf import run_spmf_streaming


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--method",
        choices=(
            "spmf", "lightgcn_ws", "pecl", "simgcl_ws",
            "lightgcn_window", "graphsail_ws", "ergnn_ws",
        ),
        required=True,
    )
    ap.add_argument("--dataset", default="ml-1m",
                    choices=(
                        "ml-1m",
                        "ml-10m",
                        "ml-20m",
                        "amz23_digital_music",
                        "amz23_all_beauty",
                        "amazon23",
                        "ciao",
                        "gowala",
                        "gowala_dense",
                        "yelp",
                        "yelp_dense",
                    ))
    ap.add_argument(
        "--amz23_category",
        default="",
        help="Amazon Reviews'23 category (e.g. Kindle_Store). Required for --dataset amazon23.",
    )
    ap.add_argument("--data_root", default="data")
    ap.add_argument("--device", default="cuda", choices=("cpu", "cuda", "auto"))
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--max_interactions", type=int, default=300000)
    ap.add_argument(
        "--kcore",
        type=int,
        default=0,
        help="Gowalla/Yelp/Amazon'23: symmetric k-core min degree (0 = dataset default).",
    )
    ap.add_argument("--warmup_epochs", type=int, default=3,
                    help="3×3 protocol: 3.")
    ap.add_argument("--streaming_passes", type=int, default=3,
                    help="3×3 protocol: 3 SGD passes per batch (SPMF & LightGCN-WS).")
    ap.add_argument("--history_replay", type=int, default=4096)
    ap.add_argument("--out_json", default="")

    ap.add_argument(
        "--val_tail_frac",
        type=float,
        default=0.0,
        help="Fraction of init_data to hold out as val tail for hyperparameter selection. "
             "Requires --tune_on_val to activate the grid search.",
    )
    ap.add_argument(
        "--tune_on_val",
        action="store_true",
        help="For SimGCL-WS: grid-search (eps, tau, lambda_cl) on the val tail "
             "instead of using fixed defaults. Ensures equal tuning budget vs DriftLA.",
    )
    args = ap.parse_args()

    if args.dataset == "amazon23" and not (args.amz23_category or "").strip():
        print("error: --amz23_category is required for --dataset amazon23", file=sys.stderr)
        raise SystemExit(2)

    def _results_tag() -> str:
        if args.dataset == "ml-1m":
            return "ml1m"
        if args.dataset == "amazon23":
            return f"amazon23_{(args.amz23_category or 'unknown').replace('/', '_')}"
        return args.dataset

    if args.method == "pecl":
        pecl = os.path.join(
            os.path.dirname(__file__),
            "pecl",
            "streaming_pecl.py",
        )
        out = args.out_json
        if not out:
            tag = _results_tag()
            out = os.path.join(
                str(_REPO_ROOT),
                "results",
                "baselines",
                f"pecl_3x3_{tag}_seed{args.seed}.json",
            )
        os.makedirs(os.path.dirname(out) or ".", exist_ok=True)
        dev = args.device
        if dev == "auto":
            dev = "cuda" if torch.cuda.is_available() else "cpu"
        cmd = [
            sys.executable,
            pecl,
            "--data_root",
            os.path.abspath(args.data_root),
            "--dataset",
            args.dataset,
            "--device",
            dev,
            "--seed",
            str(args.seed),
            "--max_interactions",
            str(args.max_interactions),
            "--kcore",
            str(args.kcore),
            "--warmup_epochs",
            str(args.warmup_epochs),
            "--epochs_per_stream_batch",
            str(args.streaming_passes),
            "--output",
            out,
        ]
        if args.dataset == "amazon23":
            cmd.extend(["--amz23_category", (args.amz23_category or "").strip()])
        env = os.environ.copy()
        root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
        env["PYTHONPATH"] = root + os.pathsep + env.get("PYTHONPATH", "")
        print(" ".join(cmd), flush=True)
        raise SystemExit(subprocess.call(cmd, env=env))

    dev = args.device
    if dev == "auto":
        dev = "cuda" if torch.cuda.is_available() else "cpu"
    device = torch.device(
        dev if (dev == "cpu" or torch.cuda.is_available()) else "cpu"
    )
    if dev == "cuda" and not torch.cuda.is_available():
        device = torch.device("cpu")
        print("CUDA unavailable, using CPU.", flush=True)

    init_data, batches, timestamps, n_users, n_items = load_chronological(
        args.dataset,
        args.data_root,
        args.max_interactions,
        kcore=args.kcore,
        amz23_category=args.amz23_category,
    )


    val_tail = None
    val_tail_frac = float(getattr(args, "val_tail_frac", 0.0))
    if val_tail_frac > 0.0:
        split = int(len(init_data) * (1.0 - val_tail_frac))
        val_tail  = init_data[split:]
        init_data = init_data[:split]
        print(
            f"[val_tail] warmup={len(init_data)} val={len(val_tail)} "
            f"(frac={val_tail_frac:.2f})",
            flush=True,
        )

    print(
        f"Users={n_users} Items={n_items} init={len(init_data)} "
        f"batches={[len(b) for b in batches]}",
        flush=True,
    )

    if args.method == "spmf":
        out = run_spmf_streaming(
            init_data, batches, n_users, n_items, device,
            warmup_epochs=args.warmup_epochs,
            streaming_passes=args.streaming_passes,
            history_replay=args.history_replay,
            seed=args.seed,
        )
    elif args.method == "simgcl_ws":
        out = run_simgcl_warmstart(
            init_data, batches, timestamps, n_users, n_items, device,
            warmup_epochs=args.warmup_epochs,
            streaming_passes=args.streaming_passes,
            history_replay=args.history_replay,
            seed=args.seed,
            val_tail=val_tail if getattr(args, "tune_on_val", False) else None,
        )
    elif args.method == "lightgcn_window":
        out = run_lightgcn_window_retrain(
            init_data, batches, timestamps, n_users, n_items, device,
            warmup_epochs=args.warmup_epochs,
            streaming_passes=args.streaming_passes,
            seed=args.seed,
        )
    elif args.method == "graphsail_ws":
        out = run_graphsail_warmstart(
            init_data, batches, timestamps, n_users, n_items, device,
            warmup_epochs=args.warmup_epochs,
            streaming_passes=args.streaming_passes,
            history_replay=args.history_replay,
            seed=args.seed,
        )
    elif args.method == "ergnn_ws":
        out = run_ergnn_warmstart(
            init_data, batches, timestamps, n_users, n_items, device,
            warmup_epochs=args.warmup_epochs,
            streaming_passes=args.streaming_passes,
            history_replay=args.history_replay,
            seed=args.seed,
        )
    else:
        out = run_lightgcn_warmstart(
            init_data, batches, timestamps, n_users, n_items, device,
            warmup_epochs=args.warmup_epochs,
            streaming_passes=args.streaming_passes,
            history_replay=args.history_replay,
            seed=args.seed,
        )
    kcore_used = None
    if args.dataset in ("gowala", "gowala_dense"):
        kcore_used = int(args.kcore) if args.kcore > 0 else (
            20 if args.dataset == "gowala_dense" else 5
        )
    elif args.dataset in ("yelp", "yelp_dense"):
        kcore_used = int(args.kcore) if args.kcore > 0 else (
            10 if args.dataset == "yelp_dense" else 5
        )
    elif args.dataset in ("amz23_digital_music", "amz23_all_beauty", "amazon23"):
        kcore_used = int(args.kcore) if args.kcore > 0 else 5
    out["dataset"] = args.dataset
    out["kcore"] = kcore_used
    if args.dataset == "amazon23":
        out["amz23_category"] = (args.amz23_category or "").strip()

    print(
        f"\n{out['method']} on {args.dataset} (seed {args.seed}): "
        f"R@10={out['avg_recall10']:.4f} NDCG@10={out['avg_ndcg10']:.4f} "
        f"total={out['total_time_s']:.1f}s",
        flush=True,
    )

    if args.out_json:
        out_path = args.out_json
    else:
        tag = _results_tag()
        m = args.method if args.method in (
            "spmf", "simgcl_ws", "lightgcn_window", "graphsail_ws", "ergnn_ws",
        ) else "lightgcn_ws"
        out_path = os.path.join(
            str(_REPO_ROOT),
            "results",
            "baselines",
            f"{m}_3x3_{tag}_seed{args.seed}.json",
        )
    os.makedirs(os.path.dirname(out_path) or ".", exist_ok=True)
    with open(out_path, "w") as f:
        json.dump(out, f, indent=2)
    print(f"Wrote {out_path}")


if __name__ == "__main__":
    main()
