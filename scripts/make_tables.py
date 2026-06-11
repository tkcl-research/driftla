from __future__ import annotations
import argparse
import json
import math
import os
import statistics
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple
import numpy as np
from scipy import stats
ROOT = Path(__file__).resolve().parents[1]
DEFAULT_GEN_DIR = ROOT / "tables"
GEN_DIR = Path(os.environ.get("DRIFTLA_TABLES_DIR", DEFAULT_GEN_DIR))
SEEDS = [42, 43, 44, 45, 46]
RNG = np.random.default_rng(12345)
DENSE_DATASETS = ["ml1m", "ciao", "ml10m_cap300k", "ml20m_cap300k"]
DENSE_LABELS = {
    "ml1m":           "ML-1M",
    "ciao":           "Ciao",
    "ml10m_cap300k":  "ML-10M-cap",
    "ml20m_cap300k":  "ML-20M-cap",
}

ROWS: List[Tuple[str, str, str]] = [
    ("amazon23_Magazine_Subscriptions_k2", "driftla",      "results/driftla/driftla_v3_champion_3x3_amazon23_Magazine_Subscriptions_k2_seed{s}.json"),
    ("amazon23_Magazine_Subscriptions_k2", "lightgcn_ws", "results/baselines/lightgcn_ws_3x3_amazon23_Magazine_Subscriptions_k2_seed{s}.json"),
    ("amazon23_Magazine_Subscriptions_k2", "spmf",        "results/baselines/spmf_3x3_amazon23_Magazine_Subscriptions_k2_seed{s}.json"),
    ("amazon23_Magazine_Subscriptions_k2", "pecl",        "results/baselines/pecl_3x3_amazon23_Magazine_Subscriptions_k2_seed{s}.json"),
    ("ml1m",          "driftla",      "results/driftla/driftla_valrouted_3x3_ml1m_seed{s}.json"),
    ("ml1m",          "lightgcn_ws", "results/baselines/lightgcn_ws_3x3_ml1m_seed{s}.json"),
    ("ml1m",          "spmf",        "results/baselines/spmf_3x3_ml1m_seed{s}.json"),
    ("ml1m",          "pecl",        "results/baselines/pecl_3x3_ml1m_seed{s}.json"),
    ("ml1m",          "simgcl_ws",   "results/baselines/simgcl_ws_valtuned_3x3_ml1m_seed{s}.json"),
    ("ciao",          "driftla",      "results/driftla/driftla_valrouted_3x3_ciao_seed{s}.json"),
    ("ciao",          "lightgcn_ws", "results/baselines/lightgcn_ws_3x3_ciao_seed{s}.json"),
    ("ciao",          "spmf",        "results/baselines/spmf_3x3_ciao_seed{s}.json"),
    ("ciao",          "pecl",        "results/baselines/pecl_3x3_ciao_seed{s}.json"),
    ("ciao",          "simgcl_ws",   "results/baselines/simgcl_ws_valtuned_3x3_ciao_seed{s}.json"),
    ("ml10m_cap300k", "driftla",      "results/driftla/driftla_valrouted_3x3_ml10m_cap300k_seed{s}.json"),
    ("ml10m_cap300k", "lightgcn_ws", "results/baselines/lightgcn_ws_3x3_ml10m_cap300k_seed{s}.json"),
    ("ml10m_cap300k", "spmf",        "results/baselines/spmf_3x3_ml10m_cap300k_seed{s}.json"),
    ("ml10m_cap300k", "pecl",        "results/baselines/pecl_3x3_ml10m_cap300k_seed{s}.json"),
    ("ml10m_cap300k", "simgcl_ws",   "results/baselines/simgcl_ws_valtuned_3x3_ml10m_cap300k_seed{s}.json"),
    ("ml20m_cap300k", "driftla",      "results/driftla/driftla_valrouted_3x3_ml20m_cap300k_seed{s}.json"),
    ("ml20m_cap300k", "lightgcn_ws", "results/baselines/lightgcn_ws_3x3_ml20m_cap300k_seed{s}.json"),
    ("ml20m_cap300k", "spmf",        "results/baselines/spmf_3x3_ml20m_cap300k_seed{s}.json"),
    ("ml20m_cap300k", "pecl",        "results/baselines/pecl_3x3_ml20m_cap300k_seed{s}.json"),
    ("ml20m_cap300k", "simgcl_ws",   "results/baselines/simgcl_ws_valtuned_3x3_ml20m_cap300k_seed{s}.json"),
    ("gowalla",       "driftla",      "results/driftla/driftla_v3_champion_3x3_gowala_seed{s}.json"),
    ("gowalla",       "lightgcn_ws", "results/baselines/lightgcn_ws_3x3_gowala_seed{s}.json"),
    ("gowalla",       "simgcl_ws",   "results/baselines/simgcl_ws_3x3_gowala_seed{s}.json"),
    ("gowalla",       "pecl",        "results/baselines/pecl_3x3_gowala_seed{s}.json"),
    ("yelp",          "driftla",      "results/driftla/driftla_v3_champion_3x3_yelp_seed{s}.json"),
    ("yelp",          "lightgcn_ws", "results/baselines/lightgcn_ws_3x3_yelp_seed{s}.json"),
    ("yelp",          "simgcl_ws",   "results/baselines/simgcl_ws_3x3_yelp_seed{s}.json"),
    ("yelp",          "pecl",        "results/baselines/pecl_3x3_yelp_seed{s}.json"),
]

METHOD_LABELS = {
    "driftla":      "DriftLA",
    "lightgcn_ws": "LightGCN-WS",
    "pecl":        "PECL",
    "spmf":        "SPMF",
    "simgcl_ws":   "SimGCL-WS",
}
GATE_ROWS = [
    ("Drift-gated (fixed preset)", "drift"),
    ("Ungated adapter",       "ungated"),
    ("Fixed gate",            "fixed"),
    ("Random gate",           "random"),
    ("Plain LoRA residual",   "plain_lora"),
]
VALROUTED_GATE_ROWS = [
    ("Drift-gated (val-routed)", "drift"),
    ("Ungated adapter",          "ungated"),
    ("Fixed gate",               "fixed"),
    ("Plain LoRA residual",      "plain_lora"),
]
CONTINUAL_ROWS = [
    ("DriftLA",           None),
    ("LightGCN-WS",      "lightgcn_ws"),
    ("LightGCN-window",  "lightgcn_window"),
    ("GraphSAIL-WS",     "graphsail_ws"),
    ("ERGNN-WS",         "ergnn_ws"),
]
ABLATION_ROWS = [
    ("Full DriftLA",            None),
    ("w/o M1 TDA",             "no_m1_tda"),
    ("w/o M2 item bias",       "no_m2_item_bias"),
    ("w/o M3 recency replay",  "no_m3_rbr"),
    ("w/o M4 drift-gated width","no_m4_dtpc"),
    ("w/o M5 distillation",    "no_m5_distill"),
    ("w/o stream adapter",     "no_m7_adapter"),
    ("w/o path contrast",      "no_path_contrast"),
    ("w/o temporal smoothness","no_temporal_smooth"),
    ("uniform negatives",      "uniform_negatives"),
    ("w/o replay stability",   "no_replay_stability"),
]
DS_FILE = {
    "ml1m":           "ml1m",
    "ciao":           "ciao",
    "ml10m_cap300k":  "ml10m_cap300k",
    "ml20m_cap300k":  "ml20m_cap300k",
    "amazon23_Magazine_Subscriptions_k2": "amazon23_Magazine_Subscriptions_k2",
}
def load_json(rel: str) -> Optional[Dict[str, Any]]:
    path = ROOT / rel
    if not path.exists() or path.stat().st_size == 0:
        return None
    try:
        with open(path) as f:
            d = json.load(f)
    except (json.JSONDecodeError, OSError):
        return None
    batches = d.get("batches")
    if isinstance(batches, list) and batches:
        metrics = [b.get("metrics", {}) for b in batches]
        r10 = [float(m["Recall@10"]) for m in metrics if "Recall@10" in m]
        n10 = [float(m["NDCG@10"])   for m in metrics if "NDCG@10"   in m]
        r20 = [float(m["Recall@20"]) for m in metrics if "Recall@20" in m]
        n20 = [float(m["NDCG@20"])   for m in metrics if "NDCG@20"   in m]
        d = dict(d)
        if r10 and "avg_recall10" not in d: d["avg_recall10"] = sum(r10) / len(r10)
        if n10 and "avg_ndcg10"   not in d: d["avg_ndcg10"]   = sum(n10) / len(n10)
        if r20 and "avg_recall20" not in d: d["avg_recall20"] = sum(r20) / len(r20)
        if n20 and "avg_ndcg20"   not in d: d["avg_ndcg20"]   = sum(n20) / len(n20)
    return d if "avg_recall10" in d else None
def collect_jsons(dataset: str, method: str) -> Dict[int, Dict[str, Any]]:
    out: Dict[int, Dict[str, Any]] = {}
    for ds, mk, tmpl in ROWS:
        if ds != dataset or mk != method:
            continue
        for s in SEEDS:
            d = load_json(tmpl.format(s=s))
            if d is not None:
                out[s] = d
    return out
def collect_metric(dataset: str, method: str, metric: str) -> List[float]:
    key_map = {
        "recall10": "avg_recall10",
        "ndcg10":   "avg_ndcg10",
        "recall20": "avg_recall20",
        "ndcg20":   "avg_ndcg20",
    }
    key = key_map[metric]
    vals = []
    for d in collect_jsons(dataset, method).values():
        if key in d and not math.isnan(float(d[key])):
            vals.append(float(d[key]))
    return vals
def collect_metric_seed_map(dataset: str, method: str, metric: str) -> Dict[int, float]:
    key_map = {
        "recall10": "avg_recall10",
        "ndcg10":   "avg_ndcg10",
        "recall20": "avg_recall20",
        "ndcg20":   "avg_ndcg20",
    }
    key = key_map[metric]
    out: Dict[int, float] = {}
    for seed, d in collect_jsons(dataset, method).items():
        if key in d and not math.isnan(float(d[key])):
            out[seed] = float(d[key])
    return out
def fmt_mean_std(vals: List[float]) -> str:
    if not vals:
        return "---"
    if len(vals) == 1:
        return f"{vals[0]:.4f}"
    return f"{statistics.mean(vals):.4f}$\\pm${statistics.stdev(vals):.4f}"
def paired_vectors(dataset: str, metric: str, base: str) -> Tuple[np.ndarray, np.ndarray]:
    s = collect_metric_seed_map(dataset, "driftla", metric)
    b = collect_metric_seed_map(dataset, base, metric)
    common = sorted(set(s).intersection(b))
    return np.array([s[k] for k in common]), np.array([b[k] for k in common])
def bootstrap_ci_mean_diff(diffs: np.ndarray, n_boot: int = 10000) -> Tuple[float, float]:
    if len(diffs) == 0:
        return (float("nan"), float("nan"))
    boots = [float(np.mean(RNG.choice(diffs, size=len(diffs), replace=True))) for _ in range(n_boot)]
    lo, hi = np.percentile(boots, [2.5, 97.5])
    return float(lo), float(hi)
def effect_stats(dataset: str, base: str, metric: str = "recall10") -> Dict[str, float]:
    s, b = paired_vectors(dataset, metric, base)
    if len(s) < 2:
        return {"n": len(s), "mean_diff": float("nan"), "t_p": float("nan"), "w_p": float("nan"), "d": float("nan"), "ci_lo": float("nan"), "ci_hi": float("nan")}
    diffs = s - b
    t_res = stats.ttest_rel(s, b)
    try:
        w_res = stats.wilcoxon(diffs)
        w_p = float(w_res.pvalue)
    except Exception:
        w_p = float("nan")
    d = float(np.mean(diffs) / np.std(diffs, ddof=1)) if np.std(diffs, ddof=1) > 0 else float("nan")
    lo, hi = bootstrap_ci_mean_diff(diffs)
    return {
        "n":         len(s),
        "mean_diff": float(np.mean(diffs)),
        "t_p":       float(t_res.pvalue),
        "w_p":       w_p,
        "d":         d,
        "ci_lo":     lo,
        "ci_hi":     hi,
    }
def per_batch_series(dataset: str, method: str, metric_key: str) -> Optional[np.ndarray]:
    series = []
    for seed, d in sorted(collect_jsons(dataset, method).items()):
        vals = []
        for b in d.get("batches", []):
            if metric_key == "loss":
                if "loss" in b:
                    vals.append(float(b["loss"]))
            elif metric_key == "latency":
                t = b.get("timings", {})
                if "total_s" in t:
                    vals.append(float(t["total_s"]))
                elif "train_s" in t:
                    vals.append(float(t["train_s"]))
            else:
                m = b.get("metrics", {})
                if metric_key in m:
                    vals.append(float(m[metric_key]))
        if vals:
            series.append(vals)
        elif metric_key == "latency":
            batches = d.get("batches", [])
            stream_s = float(d.get("stream_time_s") or d.get("total_time_s") or 0.0)
            if batches and stream_s > 0:
                series.append([stream_s / len(batches)] * len(batches))
    if not series:
        return None
    n = min(len(x) for x in series)
    return np.array([x[:n] for x in series], dtype=float).T
def per_batch_latency(dataset: str, method: str) -> List[float]:
    arr = per_batch_series(dataset, method, "latency")
    return list(arr.flatten()) if arr is not None else []
def ablation_metric(dataset: str, ab_id: Optional[str], metric: str) -> Tuple[Optional[float], int, Optional[float]]:
    key = "avg_recall10" if metric == "recall10" else "avg_ndcg10"
    tag = DS_FILE[dataset]
    if ab_id is None:
        path = f"results/driftla/driftla_v3_champion_3x3_{tag}_seed{{s}}.json"
    else:
        path = f"results/driftla/driftla_champion_ablation_{ab_id}_3x3_{tag}_seed{{s}}.json"
    vals = []
    for s in SEEDS:
        d = load_json(path.format(s=s))
        if d and key in d:
            vals.append(float(d[key]))
    if not vals:
        return (None, 0, None)
    mean = statistics.mean(vals)
    std  = statistics.stdev(vals) if len(vals) > 1 else None
    return (mean, len(vals), std)
def gate_metric(dataset: str, mode: str, metric: str) -> Tuple[Optional[float], int, Optional[float]]:
    key = "avg_recall10" if metric == "recall10" else "avg_ndcg10"
    tag = DS_FILE[dataset]
    if mode == "drift":
        path = f"results/driftla/driftla_v3_champion_3x3_{tag}_seed{{s}}.json"
    else:
        path = f"results/driftla/driftla_champion_gate_{mode}_3x3_{tag}_seed{{s}}.json"
    vals = []
    for s in SEEDS:
        d = load_json(path.format(s=s))
        if d and key in d:
            vals.append(float(d[key]))
    if not vals:
        return (None, 0, None)
    mean = statistics.mean(vals)
    std = statistics.stdev(vals) if len(vals) > 1 else None
    return (mean, len(vals), std)
def valrouted_gate_metric(dataset: str, mode: str, metric: str) -> Tuple[Optional[float], int, Optional[float]]:
    key = "avg_recall10" if metric == "recall10" else "avg_ndcg10"
    tag = DS_FILE[dataset]
    if mode == "drift":
        path = f"results/driftla/driftla_valrouted_3x3_{tag}_seed{{s}}.json"
    else:
        path = f"results/driftla/driftla_valrouted_gate_{mode}_3x3_{tag}_seed{{s}}.json"
    vals = []
    for s in SEEDS:
        d = load_json(path.format(s=s))
        if d and key in d:
            vals.append(float(d[key]))
    if not vals:
        return (None, 0, None)
    mean = statistics.mean(vals)
    std = statistics.stdev(vals) if len(vals) > 1 else None
    return (mean, len(vals), std)
def _metric_from_template(tmpl: str, metric: str) -> Tuple[Optional[float], int, Optional[float]]:
    key = "avg_recall10" if metric == "recall10" else "avg_ndcg10"
    vals = []
    for s in SEEDS:
        d = load_json(tmpl.format(s=s))
        if d and key in d:
            vals.append(float(d[key]))
    if not vals:
        return (None, 0, None)
    mean = statistics.mean(vals)
    std = statistics.stdev(vals) if len(vals) > 1 else None
    return (mean, len(vals), std)
def matched_warmup_metric(dataset: str, method: str, metric: str) -> Tuple[Optional[float], int, Optional[float]]:
    tag = DS_FILE[dataset]
    if method == "driftla":
        tmpl = f"results/driftla/driftla_v3_champion_3x3_{tag}_seed{{s}}.json"
    elif method == "driftla_adapter":
        tmpl = f"results/driftla/driftla_adapter_3x3_{tag}_seed{{s}}.json"
    else:
        tmpl = f"results/baselines/{method}_3x3_{tag}_seed{{s}}.json"
    return _metric_from_template(tmpl, metric)
def continual_metric(dataset: str, method: Optional[str], metric: str) -> Tuple[Optional[float], int, Optional[float]]:
    key = "avg_recall10" if metric == "recall10" else "avg_ndcg10"
    tag = DS_FILE[dataset]
    if method is None:
        path = f"results/driftla/driftla_valrouted_3x3_{tag}_seed{{s}}.json"
    else:
        path = f"results/baselines/{method}_3x3_{tag}_seed{{s}}.json"
    vals = []
    for s in SEEDS:
        d = load_json(path.format(s=s))
        if d and key in d:
            vals.append(float(d[key]))
    if not vals:
        return (None, 0, None)
    mean = statistics.mean(vals)
    std = statistics.stdev(vals) if len(vals) > 1 else None
    return (mean, len(vals), std)
def adapter_per_batch_latency(dataset: str) -> List[float]:
    tag = DS_FILE[dataset]
    lats: List[float] = []
    for s in SEEDS:
        d = load_json(f"results/driftla/driftla_adapter_3x3_{tag}_seed{s}.json")
        if not d:
            continue
        for b in d.get("batches", []):
            t = b.get("timings", {})
            if "total_s" in t:
                lats.append(float(t["total_s"]))
    return lats
def write_macros() -> None:
    try:
        ml10_s, _, _ = matched_warmup_metric("ml10m_cap300k", "driftla_adapter", "recall10")
        ml10_l, _, _ = matched_warmup_metric("ml10m_cap300k", "lightgcn_ws", "recall10")
        ml20_s, _, _ = matched_warmup_metric("ml20m_cap300k", "driftla_adapter", "recall10")
        ml20_l, _, _ = matched_warmup_metric("ml20m_cap300k", "lightgcn_ws", "recall10")
        gain10 = 100.0 * (ml10_s - ml10_l) / ml10_l
        gain20 = 100.0 * (ml20_s - ml20_l) / ml20_l
    except Exception:
        gain10 = gain20 = float("nan")

    ratios = []
    for ds in DENSE_DATASETS:
        s = adapter_per_batch_latency(ds)
        l = per_batch_latency(ds, "lightgcn_ws")
        if s and l:
            ratios.append(statistics.median(s) / statistics.median(l))
    slow_min = min(ratios) if ratios else float("nan")
    slow_max = max(ratios) if ratios else float("nan")

    def _fmt(v: float, suffix: str = "") -> str:
        return "---" if math.isnan(v) else f"{v:.1f}{suffix}"

    pct = r"\%"
    text = [
        "% Auto-generated by make_tables.py\n",
        f"\\newcommand{{\\DriftLAGainMLTenCap}}{{{_fmt(gain10, pct)}}}\n",
        f"\\newcommand{{\\DriftLAGainMLTwentyCap}}{{{_fmt(gain20, pct)}}}\n",
        f"\\newcommand{{\\DriftLALatencySlowMin}}{{{_fmt(slow_min, 'x')}}}\n",
        f"\\newcommand{{\\DriftLALatencySlowMax}}{{{_fmt(slow_max, 'x')}}}\n",
    ]
    (GEN_DIR / "macros.tex").write_text("".join(text))
def write_significance_table() -> None:
    lines = [
        "% Auto-generated by make_tables.py\n",
        "\\begin{table*}[t]\n\\centering\\scriptsize\n",
        "\\caption{DriftLA vs baseline significance on Recall@10 (paired over shared seeds). "
        "We report mean difference, 95\\% bootstrap CI, paired $t$-test $p$, Wilcoxon signed-rank $p$, and paired Cohen's $d$.}\n",
        "\\label{tab:significance}\n",
        "\\begin{tabular}{llrrrrr}\n\\toprule\n",
        "Dataset & Baseline & $\\Delta$mean & 95\\% CI & $p_t$ & $p_W$ & $d$ \\\\\n\\midrule\n",
    ]
    for ds in DENSE_DATASETS:
        for base in ["lightgcn_ws", "simgcl_ws", "pecl"]:
            eff = effect_stats(ds, base, "recall10")
            if eff["n"] < 2:
                continue
            lines.append(
                f"{DENSE_LABELS[ds]} & {METHOD_LABELS[base]} & "
                f"{eff['mean_diff']:.4f} & [{eff['ci_lo']:.4f}, {eff['ci_hi']:.4f}] & "
                f"{eff['t_p']:.3g} & {eff['w_p']:.3g} & {eff['d']:.3f} \\\\\n"
            )
    lines += ["\\bottomrule\n\\end{tabular}\n\\end{table*}\n"]
    (GEN_DIR / "significance_table.tex").write_text("".join(lines))
def ablation_table(metric: str) -> str:
    metric_name = "Recall@10" if metric == "recall10" else "NDCG@10"
    label = "tab:ablation_recall" if metric == "recall10" else "tab:ablation_ndcg"
    out = [
        "% Auto-generated by make_tables.py\n",
        "\\begin{table*}[t]\n\\centering\\scriptsize\n",
        f"\\caption{{{metric_name} ablations (mean$\\pm$std over five seeds).}}\n\\label{{{label}}}\n",
        "\\resizebox{\\textwidth}{!}{\n\\begin{tabular}{lrrrrr}\n\\toprule\n",
        "Ablation & ML-1M & Ciao & ML-10M-cap & ML-20M-cap & Amazon23-Mag \\\\\n\\midrule\n",
    ]
    for name, ab_id in ABLATION_ROWS:
        row = [name]
        for ds in DENSE_DATASETS + ["amazon23_Magazine_Subscriptions_k2"]:
            m, n, sd = ablation_metric(ds, ab_id, metric)
            if m is None:
                row.append("---")
                continue
            if sd is None:
                row.append(f"{m:.4f}")
            else:
                row.append(f"{m:.4f}$\\pm${sd:.4f}")
        out.append(" & ".join(row) + " \\\\\n")
    out += [
        "\\bottomrule\n\\end{tabular}}\n",
        "\\end{table*}\n",
    ]
    return "".join(out)
def write_ablation_tables() -> None:
    (GEN_DIR / "ablation_recall.tex").write_text(ablation_table("recall10"))
    (GEN_DIR / "ablation_ndcg.tex").write_text(ablation_table("ndcg10"))
def gate_ablation_table(metric: str) -> str:
    metric_name = "Recall@10" if metric == "recall10" else "NDCG@10"
    label = "tab:gate_ablation_recall" if metric == "recall10" else "tab:gate_ablation_ndcg"
    out = [
        "% Auto-generated by make_tables.py\n",
        "\\begin{table*}[t]\n\\centering\\scriptsize\n",
        f"\\caption{{Drift-gate adapter controls ({metric_name}, mean$\\pm$std over five seeds). "
        "Controls match adapter capacity and optimizer settings; only the gate schedule differs "
        "(ungated, fixed mean gate, time-shuffled gate, or plain MLP residual).}\n",
        f"\\label{{{label}}}\n",
        "\\begin{tabular}{lrrrr}\n\\toprule\n",
        "Gate mode & ML-1M & Ciao & ML-10M-cap & ML-20M-cap \\\\\n\\midrule\n",
    ]
    for name, mode in GATE_ROWS:
        row = [name]
        for ds in DENSE_DATASETS:
            m, n, sd = gate_metric(ds, mode, metric)
            if m is None:
                row.append("---")
                continue
            if sd is None:
                row.append(f"{m:.4f}")
            else:
                row.append(f"{m:.4f}$\\pm${sd:.4f}")
        out.append(" & ".join(row) + " \\\\\n")
    out += ["\\bottomrule\n\\end{tabular}\n\\end{table*}\n"]
    return "".join(out)
def write_gate_ablation_tables() -> None:
    (GEN_DIR / "gate_ablation_recall.tex").write_text(gate_ablation_table("recall10"))
    (GEN_DIR / "gate_ablation_ndcg.tex").write_text(gate_ablation_table("ndcg10"))
def write_valrouted_gate_table() -> None:
    out = [
        "% Auto-generated by make_tables.py\n",
        "\\begin{table*}[t]\n\\centering\\scriptsize\n",
        "\\caption{Drift-gate adapter controls under the validation-only routing "
        "protocol (Recall@10, mean$\\pm$std over five seeds). "
        "Drift-gated row uses val-routed DriftLA; controls share the same routing "
        "and warmup holdout.}\n",
        "\\label{tab:valrouted_gate_recall}\n",
        "\\begin{tabular}{lrrrr}\n\\toprule\n",
        "Gate mode & ML-1M & Ciao & ML-10M-cap & ML-20M-cap \\\\\n\\midrule\n",
    ]
    for name, mode in VALROUTED_GATE_ROWS:
        row = [name]
        for ds in DENSE_DATASETS:
            m, n, sd = valrouted_gate_metric(ds, mode, "recall10")
            if m is None:
                row.append("---")
            elif sd is None:
                row.append(f"{m:.4f}")
            else:
                row.append(f"{m:.4f}$\\pm${sd:.4f}")
        out.append(" & ".join(row) + " \\\\\n")
    out += ["\\bottomrule\n\\end{tabular}\n\\end{table*}\n"]
    (GEN_DIR / "valrouted_gate_recall.tex").write_text("".join(out))
def write_matched_warmup_table() -> None:
    methods = [
        ("DriftLA-Adapter", "driftla_adapter"),
        ("DriftLA (full, fixed preset)", "driftla"),
        ("LightGCN-WS", "lightgcn_ws"),
        ("SimGCL-WS", "simgcl_ws"),
        ("PECL", "pecl"),
    ]
    out = [
        "% Auto-generated by make_tables.py\n",
        "\\begin{table*}[t]\n\\centering\\scriptsize\n",
        "\\caption{Matched-warmup comparison: all methods train on the full warmup "
        "prefix (no 20\\% holdout). DriftLA and DriftLA-Adapter use the fixed "
        "preset; baselines use default configurations. Cells show R@10 / N@10.}\n",
        "\\label{tab:matched_warmup}\n",
        "\\begin{tabularx}{\\textwidth}{lYYYY}\n\\toprule\n",
        "Method & ML-1M & Ciao & ML-10M-cap & ML-20M-cap \\\\\n\\midrule\n",
    ]
    for label, mk in methods:
        row = [label]
        for ds in DENSE_DATASETS:
            m_r, _, sd_r = matched_warmup_metric(ds, mk, "recall10")
            m_n, _, sd_n = matched_warmup_metric(ds, mk, "ndcg10")
            if m_r is None:
                row.append("---")
            else:
                sr = f"{m_r:.4f}$\\pm${sd_r:.4f}" if sd_r else f"{m_r:.4f}"
                sn = f"{m_n:.4f}$\\pm${sd_n:.4f}" if sd_n and m_n else f"{m_n:.4f}" if m_n else "---"
                row.append(f"{sr} / {sn}")
        out.append(" & ".join(row) + " \\\\\n")
    out += ["\\bottomrule\n\\end{tabularx}\n\\end{table*}\n"]
    (GEN_DIR / "matched_warmup.tex").write_text("".join(out))
def write_adapter_results_table() -> None:
    dense_tags = DS_FILE
    ds_order = DENSE_DATASETS + ["gowalla", "yelp_dense"]
    ds_labels = {**DENSE_LABELS, "gowalla": "Gowalla", "yelp_dense": "Yelp-dense"}
    n_cols = 2 + len(ds_order)
    col_spec = "ll" + "r" * len(ds_order)
    header_cols = " & ".join(["Metric", "Method"] + [ds_labels[d] for d in ds_order])
    out = [
        "% Auto-generated by make_tables.py\n",
        "\\begin{table*}[t]\n\\centering\\scriptsize\n",
        "\\caption{DriftLA-Adapter (headline method) vs.\\ full DriftLA and LightGCN-WS. "
        "R@10\\,/\\,N@10 = mean$\\pm$std over five seeds; latency = median [IQR] s/batch. "
        "Adapter uses node-embedding drift (no path contrast).}\n",
        "\\label{tab:adapter_results}\n",
        "\\resizebox{\\textwidth}{!}{\n",
        f"\\begin{{tabular}}{{{col_spec}}}\n\\toprule\n",
        header_cols + " \\\\\n\\midrule\n",
        f"\\multicolumn{{{n_cols}}}{{l}}{{\\textit{{Recall@10\\,/\\,NDCG@10}}}} \\\\\n",
    ]
    for method_key, label in (
        ("driftla_adapter", "DriftLA-Adapter"),
        ("driftla", "Full DriftLA"),
        ("lightgcn_ws", "LightGCN-WS"),
    ):
        row = ["", label]
        for ds in ds_order:
            _tag = DS_FILE.get(ds, "gowala" if ds == "gowalla" else ds)
            if method_key == "driftla_adapter":
                tmpl_r = f"results/driftla/driftla_adapter_3x3_{_tag}_seed{{s}}.json"
            elif method_key == "driftla":
                tmpl_r = f"results/driftla/driftla_v3_champion_3x3_{_tag}_seed{{s}}.json"
            else:
                tmpl_r = f"results/baselines/lightgcn_ws_3x3_{_tag}_seed{{s}}.json"
            m_r, _, sd_r = _metric_from_template(tmpl_r, "recall10")
            tmpl_n = tmpl_r.replace("recall", "ndcg")
            key_n = "avg_ndcg10"
            vals_n = []
            for s in SEEDS:
                d = load_json(tmpl_r.format(s=s))
                if d and key_n in d:
                    vals_n.append(float(d[key_n]))
            if m_r is None:
                row.append("---")
            else:
                sn = statistics.mean(vals_n) if vals_n else 0.0
                sdn = statistics.stdev(vals_n) if len(vals_n) > 1 else None
                sr = f"{m_r:.4f}$\\pm${sd_r:.4f}" if sd_r else f"{m_r:.4f}"
                sn_s = f"{sn:.4f}$\\pm${sdn:.4f}" if sdn else f"{sn:.4f}"
                row.append(f"{sr} / {sn_s}")
        out.append(" & ".join(row) + " \\\\\n")
    out += [f"\\midrule\n\\multicolumn{{{n_cols}}}{{l}}{{\\textit{{Latency (seconds/batch), median [IQR]}}}} \\\\\n"]
    for method_key, label in (
        ("driftla_adapter", "DriftLA-Adapter"),
        ("driftla", "Full DriftLA"),
        ("lightgcn_ws", "LightGCN-WS"),
    ):
        row = ["", label]
        for ds in ds_order:
            _tag = DS_FILE.get(ds, "gowala" if ds == "gowalla" else ds)
            if method_key == "driftla_adapter":
                tmpl = f"results/driftla/driftla_adapter_3x3_{_tag}_seed{{s}}.json"
            elif method_key == "driftla":
                tmpl = f"results/driftla/driftla_v3_champion_3x3_{_tag}_seed{{s}}.json"
            else:
                tmpl = f"results/baselines/lightgcn_ws_3x3_{_tag}_seed{{s}}.json"
            lats = []
            for s in SEEDS:
                d = load_json(tmpl.format(s=s))
                if not d:
                    continue
                for b in d.get("batches", []):
                    t = b.get("timings", {})
                    if "total_s" in t:
                        lats.append(float(t["total_s"]))
            if not lats:
                row.append("---")
            else:
                med = statistics.median(lats)
                q1, q3 = np.percentile(np.array(lats), [25, 75])
                row.append(f"{med:.0f} [{q1:.0f}, {q3:.0f}]")
        out.append(" & ".join(row) + " \\\\\n")
    out += ["\\bottomrule\n\\end{tabular}}\n\\end{table*}\n"]
    (GEN_DIR / "adapter_results.tex").write_text("".join(out))
def write_cap_sensitivity_table() -> None:
    caps = [
        ("ml10m_cap300k", "300k"),
        ("ml10m_cap1m", "1M"),
        ("ml10m_cap3m", "3M"),
    ]
    out = [
        "% Auto-generated by make_tables.py\n",
        "\\begin{table}[t]\n\\centering\\scriptsize\n",
        "\\caption{ML-10M cap-sensitivity: earliest $N$ interactions after chronological "
        "sort (Recall@10, mean over five seeds).}\n",
        "\\label{tab:cap_sensitivity}\n",
        "\\begin{tabular}{lrrr}\n\\toprule\n",
        "Method & 300k cap & 1M cap & 3M cap \\\\\n\\midrule\n",
    ]
    specs = [
        ("DriftLA (val-routed)", "results/driftla/driftla_valrouted_3x3_{tag}_seed{s}.json"),
        ("LightGCN-WS", "results/baselines/lightgcn_ws_3x3_{tag}_seed{s}.json"),
        ("SimGCL-WS (val-tuned)", "results/baselines/simgcl_ws_valtuned_3x3_{tag}_seed{s}.json"),
    ]
    for label, tmpl in specs:
        row = [label]
        for tag, _ in caps:
            vals = []
            for s in SEEDS:
                d = load_json(tmpl.format(tag=tag, s=s))
                if d and "avg_recall10" in d:
                    vals.append(float(d["avg_recall10"]))
            row.append(f"{statistics.mean(vals):.4f}" if vals else "---")
        out.append(" & ".join(row) + " \\\\\n")
    out += ["\\bottomrule\n\\end{tabular}\n\\end{table}\n"]
    (GEN_DIR / "cap_sensitivity.tex").write_text("".join(out))
def write_computematched_table() -> None:
    out = [
        "% Auto-generated by make_tables.py\n",
        "\\begin{table*}[t]\n\\centering\\scriptsize\n",
        "\\caption{Compute-matched baselines: LightGCN-WS and SimGCL-WS with "
        "warmup/streaming passes scaled to approximate DriftLA per-batch wall-clock "
        "(Recall@10, mean$\\pm$std).}\n",
        "\\label{tab:computematched}\n",
        "\\setlength{\\tabcolsep}{3.5pt}\n",
        "\\begin{tabular}{@{}lrrrr@{}}\n\\toprule\n",
        "Method & ML-1M & Ciao & ML-10M-cap & ML-20M-cap \\\\\n\\midrule\n",
    ]
    for label, prefix in (
        ("DriftLA (val-routed)", "results/driftla/driftla_valrouted_3x3_{tag}_seed{s}.json"),
        ("LGCN-WS (compute-matched)", "results/baselines/lightgcn_ws_computematched_3x3_{tag}_seed{s}.json"),
        ("SimGCL-WS (compute-matched)", "results/baselines/simgcl_ws_computematched_3x3_{tag}_seed{s}.json"),
        ("LGCN-WS (default)", "results/baselines/lightgcn_ws_3x3_{tag}_seed{s}.json"),
    ):
        row = [label]
        for ds in DENSE_DATASETS:
            tag = DS_FILE[ds]
            vals = []
            for s in SEEDS:
                d = load_json(prefix.format(tag=tag, s=s))
                if d and "avg_recall10" in d:
                    vals.append(float(d["avg_recall10"]))
            if not vals:
                row.append("---")
            elif len(vals) == 1:
                row.append(f"{vals[0]:.4f}")
            else:
                row.append(f"{statistics.mean(vals):.4f}$\\pm${statistics.stdev(vals):.4f}")
        out.append(" & ".join(row) + " \\\\\n")
    out += ["\\bottomrule\n\\end{tabular}\n\\end{table*}\n"]
    (GEN_DIR / "computematched.tex").write_text("".join(out))
def write_symmetric_holdout_table() -> None:
    out = [
        "% Auto-generated by make_tables.py\n",
        "\\begin{table*}[t]\n\\centering\\scriptsize\n",
        "\\caption{Symmetric 80\\% warmup protocol: all methods reserve the last 20\\% "
        "of the warmup prefix (Recall@10, mean$\\pm$std). SimGCL-WS uses validation tuning "
        "on the same tail.}\n",
        "\\label{tab:symmetric_holdout}\n",
        "\\setlength{\\tabcolsep}{3.5pt}\n",
        "\\begin{tabular}{@{}lrrrr@{}}\n\\toprule\n",
        "Method & ML-1M & Ciao & ML-10M-cap & ML-20M-cap \\\\\n\\midrule\n",
    ]
    specs = [
        ("DriftLA (val-routed)", "results/driftla/driftla_valrouted_3x3_{tag}_seed{s}.json"),
        ("LGCN-WS (80\\%)", "results/baselines/lightgcn_ws_valhold_3x3_{tag}_seed{s}.json"),
        ("SimGCL-WS (80\\%+val)", "results/baselines/simgcl_ws_valhold_3x3_{tag}_seed{s}.json"),
        ("LGCN-window (80\\%)", "results/baselines/lightgcn_window_valhold_3x3_{tag}_seed{s}.json"),
    ]
    for label, tmpl in specs:
        row = [label]
        for ds in DENSE_DATASETS:
            tag = DS_FILE[ds]
            vals = []
            for s in SEEDS:
                d = load_json(tmpl.format(tag=tag, s=s))
                if d and "avg_recall10" in d:
                    vals.append(float(d["avg_recall10"]))
            if not vals:
                row.append("---")
            elif len(vals) == 1:
                row.append(f"{vals[0]:.4f}")
            else:
                row.append(f"{statistics.mean(vals):.4f}$\\pm${statistics.stdev(vals):.4f}")
        out.append(" & ".join(row) + " \\\\\n")
    out += ["\\bottomrule\n\\end{tabular}\n\\end{table*}\n"]
    (GEN_DIR / "symmetric_holdout.tex").write_text("".join(out))
def write_continual_baselines_table() -> None:
    out = [
        "% Auto-generated by make_tables.py\n",
        "\\begin{table*}[t]\n\\centering\\scriptsize\n",
        "\\caption{Continual and windowed-retraining baselines vs.\\ val-routed DriftLA "
        "(Recall@10, mean$\\pm$std over five seeds on dense benchmarks). "
        "DriftLA uses 80\\% warmup; all other rows use 100\\% warmup. "
        "Same prequential $3{\\times}3$ protocol.}\n",
        "\\label{tab:continual_baselines}\n",
        "\\begin{tabular}{lrrrr}\n\\toprule\n",
        "Method & ML-1M & Ciao & ML-10M-cap & ML-20M-cap \\\\\n\\midrule\n",
    ]
    for name, method in CONTINUAL_ROWS:
        row = [name]
        for ds in DENSE_DATASETS:
            m, n, sd = continual_metric(ds, method, "recall10")
            if m is None:
                row.append("---")
                continue
            if sd is None:
                row.append(f"{m:.4f}")
            else:
                row.append(f"{m:.4f}$\\pm${sd:.4f}")
        out.append(" & ".join(row) + " \\\\\n")
    out += ["\\bottomrule\n\\end{tabular}\n\\end{table*}\n"]
    (GEN_DIR / "continual_baselines_recall.tex").write_text("".join(out))
def write_latex_tables() -> None:
    lines = ["% Auto-generated by make_tables.py\n"]

    def summary_table(metric_a: str, metric_b: str, cap: str, label: str) -> str:
        methods = ["driftla", "lightgcn_ws", "simgcl_ws", "pecl", "spmf"]
        best_a  = {ds: max(statistics.mean(collect_metric(ds, m, metric_a) or [0]) for m in methods) for ds in DENSE_DATASETS}
        best_b  = {ds: max(statistics.mean(collect_metric(ds, m, metric_b) or [0]) for m in methods) for ds in DENSE_DATASETS}
        t = [
            "\\begin{table*}[t]\n\\centering\\scriptsize\n",
            f"\\caption{{{cap}}}\n\\label{{{label}}}\n",
            "\\begin{tabularx}{\\textwidth}{lYYYY}\n\\toprule\n",
            "Method & ML-1M & Ciao & ML-10M-cap & ML-20M-cap \\\\\n\\midrule\n",
        ]
        for mk in methods:
            row = [METHOD_LABELS[mk]]
            for ds in DENSE_DATASETS:
                va = collect_metric(ds, mk, metric_a)
                vb = collect_metric(ds, mk, metric_b)
                sa = fmt_mean_std(va)
                sb = fmt_mean_std(vb)
                a_mean = statistics.mean(va) if va else float("-inf")
                b_mean = statistics.mean(vb) if vb else float("-inf")
                a = f"\\textbf{{{sa}}}" if a_mean >= best_a[ds] - 1e-12 else sa
                b = f"\\textbf{{{sb}}}" if b_mean >= best_b[ds] - 1e-12 else sb
                row.append(f"{a} / {b}")
            t.append(" & ".join(row) + " \\\\\n")
        t += ["\\bottomrule\n\\end{tabularx}\n\\end{table*}\n\n"]
        return "".join(t)

    lines.append(summary_table("recall10", "ndcg10",
                               "Dense benchmarks (mean$\\pm$std over seeds). Cells show R@10 / N@10.",
                               "tab:accuracy_summary"))
    lines.append(summary_table("recall20", "ndcg20",
                               "Dense benchmarks (mean$\\pm$std over seeds). Cells show R@20 / N@20.",
                               "tab:accuracy_summary_r20"))


    lines += [
        "\\begin{table}[t]\n\\centering\\scriptsize\n",
        "\\caption{Per-batch latency summary (seconds): median [IQR].}\n\\label{tab:per_batch_latency}\n",
        "\\resizebox{\\textwidth}{!}{\n\\begin{tabular}{lrrrr}\n\\toprule\n",
        "Method & ML-1M & Ciao & ML-10M-cap & ML-20M-cap \\\\\n\\midrule\n",
    ]
    latency_labels = {
        "driftla": "DriftLA (val-routed)",
    }
    for mk in ["driftla", "lightgcn_ws", "simgcl_ws", "pecl", "spmf"]:
        row = [latency_labels.get(mk, METHOD_LABELS[mk])]
        for ds in DENSE_DATASETS:
            l = per_batch_latency(ds, mk)
            if not l:
                row.append("---")
                continue
            med = statistics.median(l)
            q1, q3 = np.percentile(np.array(l), [25, 75])
            row.append(f"{med:.2f} [{q1:.2f}, {q3:.2f}]")
        lines.append(" & ".join(row) + " \\\\\n")
    lines += ["\\bottomrule\n\\end{tabular}}\n\\end{table}\n\n"]


    lines += [
        "\\begin{table}[t]\n\\centering\\scriptsize\n",
        "\\caption{Amazon23-Magazine negative result (mean$\\pm$std, 5 seeds). Cells show R@10 / N@10 and R@20 / N@20.}\n\\label{tab:amazon_negative}\n",
        "\\begin{tabular}{lcc}\n\\toprule\nMethod & @10 & @20 \\\\\n\\midrule\n",
    ]
    for mk in ["driftla", "lightgcn_ws", "pecl", "spmf"]:
        r10 = fmt_mean_std(collect_metric("amazon23_Magazine_Subscriptions_k2", mk, "recall10"))
        n10 = fmt_mean_std(collect_metric("amazon23_Magazine_Subscriptions_k2", mk, "ndcg10"))
        r20 = fmt_mean_std(collect_metric("amazon23_Magazine_Subscriptions_k2", mk, "recall20"))
        n20 = fmt_mean_std(collect_metric("amazon23_Magazine_Subscriptions_k2", mk, "ndcg20"))
        lines.append(f"{METHOD_LABELS[mk]} & {r10} / {n10} & {r20} / {n20} \\\\\n")
    lines += ["\\bottomrule\n\\end{tabular}\n\\end{table}\n"]

    full = "".join(lines)
    marker = "\\label{tab:amazon_negative}"
    idx = full.rfind("\\begin{table}[t]", 0, full.find(marker))
    (GEN_DIR / "latex_tables_main.tex").write_text(full[:idx])
    (GEN_DIR / "latex_tables_amazon.tex").write_text(full[idx:])
def write_negative_results_table() -> None:
    neg_datasets = [
        ("gowalla", "Gowalla 5-core"),
        ("yelp",    "Yelp 5-core"),
        ("amazon23_Magazine_Subscriptions_k2", "Amazon23-Mag 2-core"),
    ]
    methods = ["driftla", "lightgcn_ws", "simgcl_ws", "pecl"]

    lines = [
        "% Auto-generated by make_tables.py\n",
        "\\begin{table}[t]\n\\centering\\scriptsize\n",
        "\\caption{Negative results: datasets where DriftLA underperforms warm-start "
        "baselines (Recall@10, mean$\\pm$std over available seeds).  "
        "On Gowalla and Yelp, SimGCL-WS outperforms LightGCN-WS: "
        "embedding-space uniformity regularizes better than path contrast "
        "when graph density is very low.}\n"
        "\\label{tab:negative_results}\n",
        "\\begin{tabular}{llr}\n\\toprule\n",
        "Dataset & Method & Recall@10 \\\\\n\\midrule\n",
    ]
    for ds_key, ds_label in neg_datasets:
        for mk in methods:
            vals = collect_metric(ds_key, mk, "recall10")
            if not vals:
                continue
            cell = fmt_mean_std(vals)
            lines.append(f"{ds_label} & {METHOD_LABELS.get(mk, mk)} & {cell} \\\\\n")
        lines.append("\\midrule\n")
    lines[-1] = "\\bottomrule\n"
    lines += ["\\end{tabular}\n\\end{table}\n"]
    (GEN_DIR / "negative_results_table.tex").write_text("".join(lines))
def main() -> None:
    global GEN_DIR
    ap = argparse.ArgumentParser(
        description="Regenerate LaTeX table fragments from results/ JSON logs.",
    )
    ap.add_argument(
        "--output-dir",
        type=Path,
        default=GEN_DIR,
        help="Destination for .tex fragments (default: tables/ or DRIFTLA_TABLES_DIR)",
    )
    args = ap.parse_args()
    GEN_DIR = args.output_dir
    GEN_DIR.mkdir(parents=True, exist_ok=True)
    print(f"Generating LaTeX table fragments in {GEN_DIR} ...")
    write_macros()
    print("  macros.tex")
    write_significance_table()
    print("  significance_table.tex")
    write_ablation_tables()
    print("  ablation tables")
    write_gate_ablation_tables()
    print("  gate ablation tables")
    write_valrouted_gate_table()
    print("  valrouted_gate_recall.tex")
    write_matched_warmup_table()
    print("  matched_warmup.tex")
    write_adapter_results_table()
    print("  adapter_results.tex")
    write_cap_sensitivity_table()
    print("  cap_sensitivity.tex")
    write_computematched_table()
    print("  computematched.tex")
    write_symmetric_holdout_table()
    print("  symmetric_holdout.tex")
    write_continual_baselines_table()
    print("  continual_baselines_recall.tex")
    write_latex_tables()
    print("  latex_tables_main.tex, latex_tables_amazon.tex")
    write_negative_results_table()
    print("  negative_results_table.tex")
    print("Done.")

if __name__ == "__main__":
    main()
