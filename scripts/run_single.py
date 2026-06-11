
import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from experiment_utils import (
    ABLATION_FLAGS,
    DENSE_DATASETS,
    detect_device,
    run_ablation,
    run_adapter,
    run_baseline,
    run_champion,
    run_sparse,
    run_valrouted,
    run_v4,
)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Run one experiment with paper defaults (3×3 protocol, five seeds in full campaigns).",
    )
    parser.add_argument(
        "method",
        choices=[
            "valrouted", "adapter", "champion", "driftla",
            "lightgcn_ws", "simgcl_ws", "pecl", "spmf",
            "ablation", "sparse", "v4",
        ],
        help=(
            "valrouted = dense headline DriftLA (validation-only routing); "
            "champion = full-warmup DriftLA (sparse / negative benchmarks); "
            "adapter = node-drift adapter variant; "
            "driftla = alias for valrouted on dense datasets, champion otherwise"
        ),
    )
    parser.add_argument("dataset", help="Dataset name, e.g. ml-1m, ciao, ml-10m, gowala")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--ablation", choices=list(ABLATION_FLAGS.keys()),
                        help="Ablation ID (required when method=ablation)")
    parser.add_argument("--device", default=None, help="cuda or cpu (full runs require CUDA)")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument(
        "--force",
        action="store_true",
        help="Overwrite an existing valid JSON for this run (for reproduction checks).",
    )
    args = parser.parse_args()

    device = args.device or detect_device()
    method = args.method
    if method == "driftla":
        method = "valrouted" if args.dataset in DENSE_DATASETS else "champion"

    force = args.force
    if method == "valrouted":
        run_valrouted(args.dataset, args.seed, device=device, dry_run=args.dry_run, force=force)
    elif method == "adapter":
        run_adapter(args.dataset, args.seed, device=device, dry_run=args.dry_run, force=force)
    elif method == "champion":
        run_champion(args.dataset, args.seed, device=device, dry_run=args.dry_run, force=force)
    elif method == "sparse":
        run_sparse(args.dataset, args.seed, device=device, dry_run=args.dry_run)
    elif method == "v4":
        run_v4(args.dataset, args.seed, device=device, dry_run=args.dry_run)
    elif method == "ablation":
        if not args.ablation:
            parser.error("--ablation is required when method=ablation")
        run_ablation(args.dataset, args.seed, args.ablation,
                     device=device, dry_run=args.dry_run)
    else:
        run_baseline(method, args.dataset, args.seed,
                     device=device, dry_run=args.dry_run, force=force)


if __name__ == "__main__":
    main()
