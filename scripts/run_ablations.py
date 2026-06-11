
import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from experiment_utils import (
    ABLATION_FLAGS,
    DENSE_DATASETS,
    detect_device,
    run_ablation,
)

ALL_SEEDS = [42, 43, 44, 45, 46]


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Leave-one-out DriftLA ablation runner"
    )
    parser.add_argument("--datasets", nargs="+",
                        default=["ml-1m", "ciao", "ml-10m", "ml-20m"],
                        help="Datasets (default: all 4 dense benchmarks)")
    parser.add_argument("--ablations", nargs="+",
                        default=list(ABLATION_FLAGS.keys()),
                        choices=list(ABLATION_FLAGS.keys()),
                        help=f"Ablation IDs (default: all {len(ABLATION_FLAGS)})")
    parser.add_argument("--seeds", type=int, nargs="+", default=ALL_SEEDS)
    parser.add_argument("--device", default=None)
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    device = args.device or detect_device()
    print(f"Device: {device} | Seeds: {args.seeds}")
    print(f"Datasets: {args.datasets}")
    print(f"Ablations: {args.ablations}")
    if args.dry_run:
        print("(DRY RUN)")

    for seed in args.seeds:
        for ds in args.datasets:
            for ab in args.ablations:
                run_ablation(ds, seed, ab, device=device, dry_run=args.dry_run)

    print("\n=== Done ===")


if __name__ == "__main__":
    main()
