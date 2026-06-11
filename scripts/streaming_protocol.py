
from __future__ import annotations

import json
from pathlib import Path

PAPER_WARMUP_EPOCHS = 3
PAPER_STREAMING_PASSES = 3


def config_matches_paper_protocol(config: dict | None) -> bool:
    if not config:
        return True
    w = config.get("warmup_epochs")
    s = config.get("streaming_passes")
    if w is None and s is None:
        return True
    return w == PAPER_WARMUP_EPOCHS and s == PAPER_STREAMING_PASSES


def path_expects_paper_3x3(path: Path | str) -> bool:
    return "_3x3_" in Path(path).name


def result_has_metrics(d: dict) -> bool:
    r = d.get("avg_recall10", d.get("recall10"))
    if r is None:
        batches = d.get("batches", [])
        if batches:
            vals = [
                b["metrics"]["Recall@10"]
                for b in batches
                if "metrics" in b and "Recall@10" in b["metrics"]
            ]
            r = sum(vals) / len(vals) if vals else None
    return r is not None and float(r) > 0


def json_ok(path: Path | str) -> bool:
    p = Path(path)
    if not p.exists() or p.stat().st_size < 50:
        return False
    try:
        d = json.loads(p.read_text())
    except Exception:
        return False
    if not result_has_metrics(d):
        return False
    if p.name.startswith("driftla_") and path_expects_paper_3x3(p):
        cfg = d.get("config")
        if isinstance(cfg, dict) and not config_matches_paper_protocol(cfg):
            return False
    return True


def prune_off_protocol_driftla_results(results_dir: Path) -> list[str]:
    removed: list[str] = []
    for p in sorted(results_dir.glob("driftla_*_3x3_*.json")):
        if not p.exists() or p.stat().st_size < 50:
            continue
        try:
            d = json.loads(p.read_text())
        except Exception:
            continue
        if not result_has_metrics(d):
            continue
        cfg = d.get("config")
        if isinstance(cfg, dict) and not config_matches_paper_protocol(cfg):
            p.unlink(missing_ok=True)
            removed.append(p.name)
    return removed


if __name__ == "__main__":
    import argparse

    ap = argparse.ArgumentParser(description="Prune driftla *_3x3_* JSONs not using paper 3×3 protocol")
    ap.add_argument(
        "--results-dir",
        type=Path,
        default=Path(__file__).resolve().parent.parent / "results" / "driftla",
    )
    args = ap.parse_args()
    removed = prune_off_protocol_driftla_results(args.results_dir)
    print(f"Removed {len(removed)} off-protocol file(s).")
    for name in removed:
        print(f"  {name}")
