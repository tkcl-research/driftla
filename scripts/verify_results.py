#!/usr/bin/env python3
"""Summarize bundled experiment JSON logs for reproduction verification."""

from __future__ import annotations

import argparse
import json
import statistics
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(Path(__file__).resolve().parent))

from streaming_protocol import PAPER_STREAMING_PASSES, PAPER_WARMUP_EPOCHS, json_ok

SEEDS = [42, 43, 44, 45, 46]

DENSE_LABELS = {
    "ml1m": "ML-1M",
    "ciao": "Ciao",
    "ml10m_cap300k": "ML-10M-cap",
    "ml20m_cap300k": "ML-20M-cap",
}

METHOD_LABELS = {
    "driftla": "DriftLA (val-routed)",
    "lightgcn_ws": "LightGCN-WS",
    "pecl": "PECL",
    "spmf": "SPMF",
    "simgcl_ws": "SimGCL-WS (val-tuned)",
}

# Headline dense-benchmark rows (matches scripts/make_tables.py accuracy summary).
HEADLINE_ROWS: List[Tuple[str, str, str]] = [
    ("ml1m", "driftla", "results/driftla/driftla_valrouted_3x3_ml1m_seed{s}.json"),
    ("ml1m", "lightgcn_ws", "results/baselines/lightgcn_ws_3x3_ml1m_seed{s}.json"),
    ("ml1m", "spmf", "results/baselines/spmf_3x3_ml1m_seed{s}.json"),
    ("ml1m", "pecl", "results/baselines/pecl_3x3_ml1m_seed{s}.json"),
    ("ml1m", "simgcl_ws", "results/baselines/simgcl_ws_valtuned_3x3_ml1m_seed{s}.json"),
    ("ciao", "driftla", "results/driftla/driftla_valrouted_3x3_ciao_seed{s}.json"),
    ("ciao", "lightgcn_ws", "results/baselines/lightgcn_ws_3x3_ciao_seed{s}.json"),
    ("ciao", "spmf", "results/baselines/spmf_3x3_ciao_seed{s}.json"),
    ("ciao", "pecl", "results/baselines/pecl_3x3_ciao_seed{s}.json"),
    ("ciao", "simgcl_ws", "results/baselines/simgcl_ws_valtuned_3x3_ciao_seed{s}.json"),
    ("ml10m_cap300k", "driftla", "results/driftla/driftla_valrouted_3x3_ml10m_cap300k_seed{s}.json"),
    ("ml10m_cap300k", "lightgcn_ws", "results/baselines/lightgcn_ws_3x3_ml10m_cap300k_seed{s}.json"),
    ("ml10m_cap300k", "spmf", "results/baselines/spmf_3x3_ml10m_cap300k_seed{s}.json"),
    ("ml10m_cap300k", "pecl", "results/baselines/pecl_3x3_ml10m_cap300k_seed{s}.json"),
    ("ml10m_cap300k", "simgcl_ws", "results/baselines/simgcl_ws_valtuned_3x3_ml10m_cap300k_seed{s}.json"),
    ("ml20m_cap300k", "driftla", "results/driftla/driftla_valrouted_3x3_ml20m_cap300k_seed{s}.json"),
    ("ml20m_cap300k", "lightgcn_ws", "results/baselines/lightgcn_ws_3x3_ml20m_cap300k_seed{s}.json"),
    ("ml20m_cap300k", "spmf", "results/baselines/spmf_3x3_ml20m_cap300k_seed{s}.json"),
    ("ml20m_cap300k", "pecl", "results/baselines/pecl_3x3_ml20m_cap300k_seed{s}.json"),
    ("ml20m_cap300k", "simgcl_ws", "results/baselines/simgcl_ws_valtuned_3x3_ml20m_cap300k_seed{s}.json"),
]


def load_metrics(path: Path) -> Optional[Dict[str, float]]:
    if not json_ok(path):
        return None
    try:
        data = json.loads(path.read_text())
    except (json.JSONDecodeError, OSError):
        return None
    r10 = data.get("avg_recall10")
    n10 = data.get("avg_ndcg10")
    if r10 is None or n10 is None:
        batches = data.get("batches") or []
        r_vals, n_vals = [], []
        for batch in batches:
            metrics = batch.get("metrics") or {}
            if "Recall@10" in metrics:
                r_vals.append(float(metrics["Recall@10"]))
            if "NDCG@10" in metrics:
                n_vals.append(float(metrics["NDCG@10"]))
        if r_vals:
            r10 = sum(r_vals) / len(r_vals)
        if n_vals:
            n10 = sum(n_vals) / len(n_vals)
    if r10 is None or n10 is None:
        return None
    return {"r10": float(r10), "n10": float(n10)}


def fmt_mean_std(vals: List[float]) -> str:
    if not vals:
        return "---"
    if len(vals) == 1:
        return f"{vals[0]:.4f}"
    return f"{statistics.mean(vals):.4f}±{statistics.stdev(vals):.4f}"


def collect_row(dataset: str, method: str, tmpl: str) -> Tuple[int, List[float], List[float], List[int]]:
    r10_vals: List[float] = []
    n10_vals: List[float] = []
    present: List[int] = []
    for seed in SEEDS:
        path = ROOT / tmpl.format(s=seed)
        metrics = load_metrics(path)
        if metrics is None:
            continue
        present.append(seed)
        r10_vals.append(metrics["r10"])
        n10_vals.append(metrics["n10"])
    return len(present), r10_vals, n10_vals, present


def scan_results_tree(results_root: Path) -> Tuple[int, int, int]:
    valid = 0
    invalid = 0
    for path in sorted(results_root.rglob("*.json")):
        if json_ok(path):
            valid += 1
        else:
            invalid += 1
    return valid, invalid, valid + invalid


def compare_to_reference(new_path: Path, reference: Optional[Path] = None) -> int:
    if not new_path.exists():
        print(f"Missing rerun file: {new_path}")
        return 1
    new_metrics = load_metrics(new_path)
    if new_metrics is None:
        print(f"Could not read metrics from {new_path}")
        return 1

    ref_path = reference
    if ref_path is None:
        name = new_path.name
        for suffix in (".rerun.json", ".new.json"):
            if name.endswith(suffix):
                name = name[: -len(suffix)] + ".json"
                break
        ref_path = ROOT / "results" / new_path.parent.name / name
        if not ref_path.exists():
            ref_path = ROOT / new_path.parent.relative_to(ROOT) / name if new_path.is_relative_to(ROOT) else None

    if ref_path is None or not ref_path.exists():
        print(f"No bundled reference found for {new_path.name}")
        print(f"  R@10={new_metrics['r10']:.4f}  N@10={new_metrics['n10']:.4f}")
        return 0

    ref_metrics = load_metrics(ref_path)
    if ref_metrics is None:
        print(f"Could not read bundled reference: {ref_path}")
        return 1

    dr = new_metrics["r10"] - ref_metrics["r10"]
    dn = new_metrics["n10"] - ref_metrics["n10"]
    print(f"Compare: {new_path.name}")
    print(f"  bundled  R@10={ref_metrics['r10']:.4f}  N@10={ref_metrics['n10']:.4f}  ({ref_path})")
    print(f"  rerun    R@10={new_metrics['r10']:.4f}  N@10={new_metrics['n10']:.4f}  ({new_path})")
    print(f"  delta    R@10={dr:+.4f}  N@10={dn:+.4f}")
    print("  (Small differences are expected across hardware/driver stacks.)")
    return 0


def main() -> int:
    ap = argparse.ArgumentParser(
        description="Verify bundled experiment JSON logs (no datasets required).",
    )
    ap.add_argument(
        "--compare",
        type=Path,
        help="Compare a freshly rerun JSON to the bundled reference with the same name.",
    )
    ap.add_argument(
        "--reference",
        type=Path,
        help="Explicit bundled reference JSON for --compare.",
    )
    ap.add_argument(
        "--require-all-seeds",
        action="store_true",
        help="Exit with code 1 unless every headline row has all five seeds.",
    )
    args = ap.parse_args()

    if args.compare:
        return compare_to_reference(args.compare.resolve(), args.reference)

    driftla_dir = ROOT / "results" / "driftla"
    baseline_dir = ROOT / "results" / "baselines"
    d_valid, d_bad, d_total = scan_results_tree(driftla_dir)
    b_valid, b_bad, b_total = scan_results_tree(baseline_dir)

    print("DriftLA results verification")
    print(f"  Protocol: {PAPER_WARMUP_EPOCHS} warmup epochs + {PAPER_STREAMING_PASSES} streaming passes")
    print(f"  Seeds: {SEEDS}")
    print(f"  JSON logs: driftla {d_valid}/{d_total} valid, baselines {b_valid}/{b_total} valid")
    if d_bad or b_bad:
        print(f"  Warning: {d_bad + b_bad} invalid or incomplete JSON file(s) ignored.")
    print()

    header = f"{'Dataset':<14} {'Method':<24} {'Seeds':<7} {'R@10':<18} {'N@10':<18}"
    print(header)
    print("-" * len(header))

    incomplete = 0
    for dataset, method, tmpl in HEADLINE_ROWS:
        n, r10_vals, n10_vals, present = collect_row(dataset, method, tmpl)
        seeds_label = f"{n}/{len(SEEDS)}"
        if n < len(SEEDS):
            incomplete += 1
        ds_label = DENSE_LABELS.get(dataset, dataset)
        method_label = METHOD_LABELS.get(method, method)
        print(
            f"{ds_label:<14} {method_label:<24} {seeds_label:<7} "
            f"{fmt_mean_std(r10_vals):<18} {fmt_mean_std(n10_vals):<18}"
        )

    print()
    print("Optional: regenerate LaTeX table fragments from these JSONs:")
    print("  python3 scripts/make_tables.py")
    print()
    print("To rerun one dense headline experiment (GPU + data required):")
    print("  cp results/driftla/driftla_valrouted_3x3_ml1m_seed42.json /tmp/bundled.json")
    print("  python3 scripts/run_single.py valrouted ml-1m --seed 42 --force")
    print("  python3 scripts/verify_results.py --compare results/driftla/driftla_valrouted_3x3_ml1m_seed42.json --reference /tmp/bundled.json")

    if args.require_all_seeds and incomplete:
        print(f"\nFAIL: {incomplete} headline row(s) missing one or more seeds.")
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
