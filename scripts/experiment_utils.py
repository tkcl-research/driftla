
from __future__ import annotations

import gc
import json
import os
import shutil
import subprocess
import sys
import tempfile
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

from streaming_protocol import (
    PAPER_STREAMING_PASSES,
    PAPER_WARMUP_EPOCHS,
    json_ok,
    prune_off_protocol_driftla_results,
)

ROOT = Path(__file__).resolve().parent.parent
SRC = ROOT / "src"
RESULTS_DriftLA = ROOT / "results" / "driftla"
RESULTS_BASELINES = ROOT / "results" / "baselines"
DATASETS = ["ml-1m", "ciao", "ml-10m", "ml-20m", "gowala", "yelp",
            "amazon23_Magazine_Subscriptions",
            "amazon23_All_Beauty",
            "amazon23_Digital_Music"]

DENSE_DATASETS = ["ml-1m", "ciao", "ml-10m", "ml-20m"]

ABLATION_FLAGS: dict[str, list[str]] = {
    "no_m7_adapter":       ["--no_stream_adapter"],
    "no_path_contrast":    ["--no_path_contrast"],
    "no_temporal_smooth":  ["--no_temporal_smooth"],
    "no_m1_tda":           ["--no_time_decay"],
    "no_m5_distill":       ["--no_distill"],
    "no_m4_dtpc":          ["--no_drift_gate"],
    "no_m2_item_bias":     ["--no_item_bias"],
    "no_m3_rbr":           ["--no_recency_replay"],
    "uniform_negatives":   ["--uniform_negatives"],
    "no_replay_stability": ["--no_replay_stability"],
}


GATE_ABLATION_MODES = ["ungated", "fixed", "random", "plain_lora"]

CONTINUAL_BASELINES = ["lightgcn_window", "graphsail_ws", "ergnn_ws"]

DENSE_SEEDS = [42, 43, 44, 45, 46]

VAL_ROUTED_FLAGS = ["--routing", "auto_density_drift", "--val_tail_frac", "0.2"]
ADAPTER_FLAGS = ["--drift_source", "node"]


def env(*, require_gpu: bool = True) -> dict[str, str]:
    e = os.environ.copy()
    e["PYTHONPATH"] = str(SRC) + os.pathsep + e.get("PYTHONPATH", "")
    e["OMP_NUM_THREADS"] = "1"
    e["MKL_NUM_THREADS"] = "1"
    e["OPENBLAS_NUM_THREADS"] = "1"

    e.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")
    if require_gpu:
        e["DriftLA_REQUIRE_GPU"] = "1"

        e.setdefault("DriftLA_CUDA_SAFE", "1")
    return e


def detect_device() -> str:
    try:
        import torch
        return "cuda" if torch.cuda.is_available() else "cpu"
    except Exception:
        return "cpu"


def require_cuda() -> str:
    if detect_device() != "cuda":
        raise RuntimeError(
            "CUDA GPU required but not available. "
            "Check nvidia-smi and PyTorch CUDA build."
        )
    return "cuda"


def resolve_device(device: Optional[str] = None, *, require_gpu: bool = True) -> str:
    if require_gpu:
        return require_cuda()
    dev = device or detect_device()
    if dev == "cuda" and detect_device() != "cuda":
        raise RuntimeError("Requested cuda but torch.cuda.is_available() is False")
    return dev


def cleanup_gpu_cache() -> None:
    gc.collect()
    try:
        import torch
        if torch.cuda.is_available():
            torch.cuda.synchronize()
            torch.cuda.empty_cache()
            torch.cuda.ipc_collect()
    except Exception:
        pass


def reset_cuda_device() -> None:
    script = (
        "import gc\n"
        "import torch\n"
        "gc.collect()\n"
        "if torch.cuda.is_available():\n"
        "    for fn in (torch.cuda.synchronize, torch.cuda.empty_cache, torch.cuda.ipc_collect):\n"
        "        try:\n"
        "            fn()\n"
        "        except RuntimeError:\n"
        "            break\n"
        "        except Exception:\n"
        "            pass\n"
    )
    try:
        subprocess.run(
            [sys.executable, "-c", script],
            env=env(require_gpu=False),
            cwd=ROOT,
            capture_output=True,
            timeout=120,
        )
    except Exception as exc:
        print(f"  [WARN] reset_cuda_device: {exc}")
    cleanup_gpu_cache()


def recover_cuda_after_failure(wait_s: float = 45.0) -> None:
    print(f"  [CUDA RECOVER] waiting {wait_s:.0f}s before retry/next job...")
    time.sleep(wait_s)
    reset_cuda_device()


def prune_invalid_results(*roots: Path) -> int:
    removed = 0
    for root in roots:
        if not root.exists():
            continue
        for path in root.glob("*.json"):
            if path.exists() and not json_ok(path):
                try:
                    path.unlink()
                    removed += 1
                    print(f"  [PRUNE] removed invalid {path.name}")
                except OSError as exc:
                    print(f"  [PRUNE] failed {path.name}: {exc}")
    return removed


def prune_runtime_cache(max_age_hours: float = 24.0) -> int:
    import time
    removed = 0
    cutoff = time.time() - max_age_hours * 3600
    for cache_root in (ROOT / "src", ROOT / "scripts"):
        for pycache in cache_root.rglob("__pycache__"):
            try:
                if pycache.is_dir():
                    shutil.rmtree(pycache, ignore_errors=True)
                    removed += 1
            except OSError:
                pass
    tmp = Path(tempfile.gettempdir())
    for pattern in ("torchinductor_*", "triton_*"):
        for item in tmp.glob(pattern):
            try:
                if item.stat().st_mtime < cutoff:
                    if item.is_dir():
                        shutil.rmtree(item, ignore_errors=True)
                    else:
                        item.unlink(missing_ok=True)
                    removed += 1
            except OSError:
                pass
    return removed


def kill_stale_cpu_train_jobs() -> int:
    try:
        proc = subprocess.run(
            ["pgrep", "-f", r"driftla\.train.*--device cpu"],
            capture_output=True, text=True,
        )
    except FileNotFoundError:
        return 0
    pids = [p.strip() for p in proc.stdout.splitlines() if p.strip().isdigit()]
    killed = 0
    for pid in pids:
        r = subprocess.run(["kill", "-9", pid], capture_output=True)
        if r.returncode == 0:
            killed += 1
            print(f"  [KILL] CPU zombie PID {pid}")
    return killed


def prepare_gpu_run(*, prune_results: bool = True, prune_cache: bool = True) -> str:
    device = require_cuda()
    n_kill = kill_stale_cpu_train_jobs()
    if n_kill:
        print(f"Killed {n_kill} stale CPU train job(s)")
    if prune_results:
        n = prune_invalid_results(RESULTS_DriftLA, RESULTS_BASELINES)
        if n:
            print(f"Pruned {n} invalid result JSON(s)")
    if prune_cache:
        n = prune_runtime_cache()
        if n:
            print(f"Pruned {n} cache/temp item(s)")
    cleanup_gpu_cache()
    return device


def refresh_progress_md(quiet: bool = True) -> None:
    del quiet


def dataset_tag(dataset: str, *, uncapped: bool = False) -> str:
    mapping = {
        "ml-1m": "ml1m",
        "ml-10m": "ml10m_full" if uncapped else "ml10m_cap300k",
        "ml-20m": "ml20m_full" if uncapped else "ml20m_cap300k",
        "ciao": "ciao",
        "gowala": "gowala",
        "yelp": "yelp",
    }
    if dataset in mapping:
        return mapping[dataset]
    if dataset.startswith("amazon23_"):
        cat = dataset[len("amazon23_"):]
        return f"amazon23_{cat}_k2"
    return dataset.replace("-", "")


def dataset_flags(dataset: str, max_interactions: int = 300_000, *, uncapped: bool = False) -> list[str]:
    cap = 0 if uncapped else max_interactions
    if dataset == "ml-1m":
        return ["--dataset", "ml-1m"]
    if dataset == "ml-10m":
        return ["--dataset", "ml-10m", "--max_interactions", str(cap)]
    if dataset == "ml-20m":
        return ["--dataset", "ml-20m", "--max_interactions", str(cap)]
    if dataset == "ciao":
        return ["--dataset", "ciao"]
    if dataset == "gowala":
        return ["--dataset", "gowala"]
    if dataset == "yelp":
        return ["--dataset", "yelp"]
    if dataset.startswith("amazon23_"):
        cat = dataset[len("amazon23_"):]
        return ["--dataset", "amazon23", "--amz23_category", cat]
    return ["--dataset", dataset]


def run_driftla(
    dataset: str,
    seed: int,
    preset: str = "v3_champion",
    use_dataset_preset: bool = True,
    device: Optional[str] = None,
    extra_flags: Optional[list[str]] = None,
    ablation_id: str = "",
    gate_ablation_id: str = "",
    uncapped: bool = False,
    out_json: Optional[Path] = None,
    dry_run: bool = False,
    sparse_mode: bool = False,
    max_retries: int = 3,
    force: bool = False,
) -> tuple[Optional[Path], int]:
    device = resolve_device(device, require_gpu=True)
    tag = dataset_tag(dataset, uncapped=uncapped)
    if gate_ablation_id:
        out = out_json or RESULTS_DriftLA / f"driftla_champion_gate_{gate_ablation_id}_3x3_{tag}_seed{seed}.json"
    elif ablation_id:
        out = out_json or RESULTS_DriftLA / f"driftla_champion_ablation_{ablation_id}_3x3_{tag}_seed{seed}.json"
    elif preset == "v3_sparse_uniform":
        out = out_json or RESULTS_DriftLA / f"driftla_v3_sparse_uniform_3x3_{tag}_seed{seed}.json"
    elif preset == "v3_sparse" or sparse_mode:
        out = out_json or RESULTS_DriftLA / f"driftla_v3_sparse_3x3_{tag}_seed{seed}.json"
    elif preset == "v4_improved":
        out = out_json or RESULTS_DriftLA / f"driftla_v4_improved_3x3_{tag}_seed{seed}.json"
    else:
        out = out_json or RESULTS_DriftLA / f"driftla_v3_champion_3x3_{tag}_seed{seed}.json"

    if not force and json_ok(out):
        print(f"  [SKIP] {out.name}")
        return out, 0
    if out.exists() and (force or not json_ok(out)):
        out.unlink(missing_ok=True)
        if force:
            print(f"  [OVERWRITE] {out.name}")
        else:
            print(f"  [PRUNE] removed incomplete {out.name}")

    cmd = [
        sys.executable, "-m", "driftla.train",
        "--data_root", "data",
        "--preset", preset,
        "--device", device,
        "--seed", str(seed),
        "--out_json", str(out),
        *(["--use_dataset_preset"] if use_dataset_preset else []),
        *dataset_flags(dataset, uncapped=uncapped),
        *(extra_flags or []),
        *(["--ablation_id", ablation_id] if ablation_id else []),
        *(["--ablation_id", f"gate_{gate_ablation_id}"] if gate_ablation_id else []),
        *(["--sparse_mode"] if sparse_mode and preset != "v3_sparse" else []),
        "--warmup_epochs", str(PAPER_WARMUP_EPOCHS),
        "--streaming_passes", str(PAPER_STREAMING_PASSES),
    ]
    if preset == "v4_improved":
        cmd.extend(["--hard_neg_k", "8"])

    print(f"  --> DriftLA {preset} {dataset} seed {seed}"
            + (f" [{ablation_id}]" if ablation_id else "")
            + (f" [gate={gate_ablation_id}]" if gate_ablation_id else ""))
    if dry_run:
        print("     (dry-run, skipping)")
        return out, 0

    RESULTS_DriftLA.mkdir(parents=True, exist_ok=True)
    last_rc = 1
    for attempt in range(1, max_retries + 1):
        reset_cuda_device()
        if attempt > 1:
            if out.exists() and not json_ok(out):
                out.unlink(missing_ok=True)
                print(f"  [RETRY {attempt}/{max_retries}] {out.name}")

        result = subprocess.run(cmd, env=env(require_gpu=True), cwd=ROOT)
        if result.returncode == 2:
            recover_cuda_after_failure(wait_s=min(90.0, 30.0 * attempt))
        else:
            cleanup_gpu_cache()

        if result.returncode == 0 and json_ok(out):
            return out, 0
        if result.returncode == 0 and out.exists():
            print(f"  [WARN] exit 0 but invalid JSON: {out.name}")
        if attempt > 1 or result.returncode == 2:
            backoff = min(90, 15 * attempt) if result.returncode == 2 else min(15, 5 * attempt)
            time.sleep(backoff)
        last_rc = result.returncode

    print(f"  [FAIL] {out.name} after {max_retries} attempt(s)")
    if out.exists() and not json_ok(out):
        out.unlink(missing_ok=True)
    return None, last_rc


def run_baseline(
    method: str,
    dataset: str,
    seed: int,
    device: Optional[str] = None,
    out_json: Optional[Path] = None,
    dry_run: bool = False,
    uncapped: bool = False,
    force: bool = False,
) -> Optional[Path]:
    device = resolve_device(device, require_gpu=True)
    tag = dataset_tag(dataset, uncapped=uncapped)
    out = out_json or RESULTS_BASELINES / f"{method}_3x3_{tag}_seed{seed}.json"

    if not force and json_ok(out):
        print(f"  [SKIP] {out.name}")
        return out
    if out.exists() and (force or not json_ok(out)):
        out.unlink(missing_ok=True)
        if force:
            print(f"  [OVERWRITE] {out.name}")
        else:
            print(f"  [PRUNE] removed incomplete {out.name}")

    cmd = [
        sys.executable, "-m", "baselines",
        "--method", method,
        "--data_root", "data",
        "--device", device,
        "--seed", str(seed),
        "--out_json", str(out),
        "--warmup_epochs", str(PAPER_WARMUP_EPOCHS),
        "--streaming_passes", str(PAPER_STREAMING_PASSES),
        *dataset_flags(dataset, uncapped=uncapped),
    ]

    print(f"  --> {method} {dataset} seed {seed}")
    if dry_run:
        print("     (dry-run, skipping)")
        return out

    RESULTS_BASELINES.mkdir(parents=True, exist_ok=True)
    result = subprocess.run(cmd, env=env(require_gpu=True), cwd=ROOT)
    cleanup_gpu_cache()
    return out if result.returncode == 0 else None


def run_valrouted(
    dataset: str,
    seed: int,
    device: Optional[str] = None,
    dry_run: bool = False,
    uncapped: bool = False,
    out_json: Optional[Path] = None,
    force: bool = False,
) -> tuple[Optional[Path], int]:
    tag = dataset_tag(dataset, uncapped=uncapped)
    out = out_json or RESULTS_DriftLA / f"driftla_valrouted_3x3_{tag}_seed{seed}.json"
    return run_driftla(
        dataset=dataset,
        seed=seed,
        device=device,
        extra_flags=list(VAL_ROUTED_FLAGS),
        out_json=out,
        uncapped=uncapped,
        dry_run=dry_run,
        force=force,
    )


def run_adapter(
    dataset: str,
    seed: int,
    device: Optional[str] = None,
    dry_run: bool = False,
    uncapped: bool = False,
    out_json: Optional[Path] = None,
    force: bool = False,
) -> tuple[Optional[Path], int]:
    tag = dataset_tag(dataset, uncapped=uncapped)
    out = out_json or RESULTS_DriftLA / f"driftla_adapter_3x3_{tag}_seed{seed}.json"
    return run_driftla(
        dataset=dataset,
        seed=seed,
        device=device,
        extra_flags=list(ADAPTER_FLAGS),
        out_json=out,
        uncapped=uncapped,
        dry_run=dry_run,
        force=force,
    )


def run_champion(
    dataset: str,
    seed: int,
    device: Optional[str] = None,
    dry_run: bool = False,
    uncapped: bool = False,
    out_json: Optional[Path] = None,
    force: bool = False,
) -> tuple[Optional[Path], int]:
    tag = dataset_tag(dataset, uncapped=uncapped)
    out = out_json or RESULTS_DriftLA / f"driftla_v3_champion_3x3_{tag}_seed{seed}.json"
    return run_driftla(
        dataset=dataset,
        seed=seed,
        device=device,
        out_json=out,
        uncapped=uncapped,
        dry_run=dry_run,
        force=force,
    )


def run_v4(
    dataset: str,
    seed: int,
    device: Optional[str] = None,
    dry_run: bool = False,
) -> tuple[Optional[Path], int]:
    return run_driftla(
        dataset=dataset,
        seed=seed,
        preset="v4_improved",
        device=device,
        dry_run=dry_run,
    )


def run_sparse(
    dataset: str,
    seed: int,
    device: Optional[str] = None,
    dry_run: bool = False,
) -> tuple[Optional[Path], int]:
    return run_driftla(
        dataset=dataset,
        seed=seed,
        preset="v3_sparse",
        device=device,
        dry_run=dry_run,
    )


def run_sparse_uniform(
    dataset: str,
    seed: int,
    device: Optional[str] = None,
    dry_run: bool = False,
) -> tuple[Optional[Path], int]:
    return run_driftla(
        dataset=dataset,
        seed=seed,
        preset="v3_sparse_uniform",
        device=device,
        dry_run=dry_run,
    )


def run_sparse_sweep(
    dataset: str,
    seed: int,
    extra_flags: list[str],
    out_json: Path,
    device: Optional[str] = None,
    dry_run: bool = False,
) -> tuple[Optional[Path], int]:
    return run_driftla(
        dataset=dataset,
        seed=seed,
        preset="v3_sparse",
        use_dataset_preset=True,
        extra_flags=extra_flags,
        out_json=out_json,
        device=device,
        dry_run=dry_run,
    )


def run_gate_ablation(
    dataset: str,
    seed: int,
    gate_mode: str,
    device: Optional[str] = None,
    dry_run: bool = False,
    uncapped: bool = False,
) -> tuple[Optional[Path], int]:
    if gate_mode not in GATE_ABLATION_MODES:
        raise ValueError(f"Unknown gate_mode '{gate_mode}'. Choose from: {GATE_ABLATION_MODES}")
    flags = ["--adapter_gate_mode", gate_mode]
    if gate_mode == "fixed":
        flags.append("--adapter_fixed_gate=0.5")
    return run_driftla(
        dataset=dataset,
        seed=seed,
        device=device,
        extra_flags=flags,
        gate_ablation_id=gate_mode,
        uncapped=uncapped,
        dry_run=dry_run,
    )


def run_ablation(
    dataset: str,
    seed: int,
    ablation_id: str,
    device: Optional[str] = None,
    dry_run: bool = False,
    uncapped: bool = False,
) -> tuple[Optional[Path], int]:
    flags = ABLATION_FLAGS.get(ablation_id)
    if flags is None:
        raise ValueError(f"Unknown ablation_id '{ablation_id}'. "
                         f"Choose from: {list(ABLATION_FLAGS)}")
    return run_driftla(
        dataset=dataset,
        seed=seed,
        device=device,
        extra_flags=flags,
        ablation_id=ablation_id,
        uncapped=uncapped,
        dry_run=dry_run,
    )


@dataclass
class JobOutcome:
    path: Optional[Path]
    cuda_error: bool = False
    ok: bool = False


def query_gpu_stats() -> Optional[tuple[float, float, float]]:
    try:
        proc = subprocess.run(
            [
                "nvidia-smi",
                "--query-gpu=memory.used,memory.total,utilization.gpu",
                "--format=csv,noheader,nounits",
            ],
            capture_output=True,
            text=True,
            timeout=15,
        )
        if proc.returncode != 0:
            return None
        parts = [p.strip() for p in proc.stdout.strip().split(",")]
        if len(parts) < 3:
            return None
        return float(parts[0]), float(parts[1]), float(parts[2])
    except Exception:
        return None


def count_active_driftla_trains() -> int:
    try:
        proc = subprocess.run(
            ["pgrep", "-fc", r"driftla\.train"],
            capture_output=True,
            text=True,
            timeout=10,
        )
        if proc.returncode not in (0, 1):
            return 0
        return max(0, int(proc.stdout.strip() or "0"))
    except Exception:
        return 0


def suggest_gpu_workers(
    max_cap: int = 3,
    min_workers: int = 2,
    *,
    mib_per_job: float = 7000.0,
) -> int:
    max_cap = max(1, max_cap)
    min_workers = max(1, min(min_workers, max_cap))

    stats = query_gpu_stats()
    if stats is None:
        return min_workers

    used_mib, total_mib, util = stats
    free_mib = max(0.0, total_mib - used_mib)
    by_vram = max(1, int(free_mib // mib_per_job))
    if free_mib >= 12000 and util < 60:
        by_vram = max(by_vram, 2)
    if free_mib >= 16000 and util < 45:
        by_vram = max(by_vram, min(3, max_cap))

    n = min(max_cap, by_vram)
    if util >= 90:
        n = min(n, 1)
    elif util >= 80:
        n = min(n, 2)
    return max(min_workers, n)


def _job_worker(kwargs: dict) -> JobOutcome:
    kwargs = dict(kwargs)
    kind = kwargs.pop("_kind")
    cuda_error = False
    path: Optional[Path] = None
    try:
        if kind == "driftla":
            path, rc = run_driftla(**kwargs)
            cuda_error = rc == 2
        elif kind == "baseline":
            path = run_baseline(**kwargs)
        elif kind == "ablation":
            path, rc = run_ablation(**kwargs)
            cuda_error = rc == 2
        elif kind == "gate_ablation":
            path, rc = run_gate_ablation(**kwargs)
            cuda_error = rc == 2
        elif kind == "v4":
            path, rc = run_v4(**kwargs)
            cuda_error = rc == 2
        elif kind == "sparse":
            path, rc = run_sparse(**kwargs)
            cuda_error = rc == 2
        elif kind == "sparse_uniform":
            path, rc = run_sparse_uniform(**kwargs)
            cuda_error = rc == 2
        elif kind == "sparse_sweep":
            path, rc = run_sparse_sweep(**kwargs)
            cuda_error = rc == 2
        else:
            raise ValueError(f"Unknown job kind: {kind}")
        ok = path is not None and (path.exists() and json_ok(path))
        return JobOutcome(path=path, cuda_error=cuda_error, ok=ok)
    finally:
        cleanup_gpu_cache()


def run_parallel(
    jobs: list[dict],
    max_workers: int = 4,
    device: Optional[str] = None,
    require_gpu: bool = True,
) -> list[Optional[Path]]:
    from concurrent.futures import ProcessPoolExecutor, as_completed

    if require_gpu:
        device = prepare_gpu_run()
    else:
        device = resolve_device(device, require_gpu=False)
    if device != "cuda":
        raise RuntimeError(f"Refusing to run parallel campaign on device={device!r}")


    enriched = []
    for j in jobs:
        j = dict(j)
        j["device"] = "cuda"
        enriched.append(j)

    print(f"GPU campaign: device=cuda workers={max_workers} jobs={len(enriched)}")
    refresh_progress_md()

    results: list[Optional[Path]] = []
    with ProcessPoolExecutor(max_workers=max_workers) as pool:
        futures = {pool.submit(_job_worker, j): j for j in enriched}
        for fut in as_completed(futures):
            try:
                outcome = fut.result()
                results.append(outcome.path)
                if outcome.cuda_error:
                    recover_cuda_after_failure(wait_s=30.0)
                else:
                    reset_cuda_device()
            except Exception as exc:
                j = futures[fut]
                print(f"  [ERROR] job {j} raised: {exc}")
                results.append(None)
                recover_cuda_after_failure(wait_s=30.0)
            refresh_progress_md()
    refresh_progress_md()
    return results


def run_parallel_adaptive(
    jobs: list[dict],
    *,
    max_workers: int = 3,
    min_workers: int = 2,
    device: Optional[str] = None,
    require_gpu: bool = True,
) -> list[Optional[Path]]:
    from concurrent.futures import ProcessPoolExecutor, as_completed

    if require_gpu:
        prepare_gpu_run()
    else:
        resolve_device(device, require_gpu=False)
    max_workers = max(1, max_workers)
    min_workers = max(1, min(min_workers, max_workers))

    enriched = []
    for j in jobs:
        jj = dict(j)
        jj["device"] = "cuda"
        enriched.append(jj)

    pending = list(enriched)
    results: list[Optional[Path]] = []
    consecutive_ok_batches = 0
    target_workers = min_workers

    print(
        f"GPU campaign (adaptive): jobs={len(pending)} "
        f"min_workers={min_workers} max_workers={max_workers}"
    )
    refresh_progress_md()

    while pending:
        stats = query_gpu_stats()
        if stats:
            used, total, util = stats
            print(
                f"  [GPU] {used:.0f}/{total:.0f} MiB used, util={util:.0f}% "
                f"active_train={count_active_driftla_trains()}"
            )

        suggested = suggest_gpu_workers(max_workers, min_workers)
        if consecutive_ok_batches >= 1 and target_workers < max_workers:
            target_workers = min(max_workers, target_workers + 1)
        batch_workers = max(min_workers, min(target_workers, suggested, max_workers))
        batch_workers = min(batch_workers, len(pending))

        batch = pending[:batch_workers]
        pending = pending[batch_workers:]
        print(f"  [BATCH] workers={batch_workers} size={len(batch)} remaining={len(pending)}")

        batch_cuda = False
        batch_ok = 0
        with ProcessPoolExecutor(max_workers=batch_workers) as pool:
            futures = {pool.submit(_job_worker, j): j for j in batch}
            for fut in as_completed(futures):
                job = futures[fut]
                try:
                    outcome = fut.result()
                    results.append(outcome.path)
                    if outcome.cuda_error:
                        batch_cuda = True
                    if outcome.ok:
                        batch_ok += 1
                except Exception as exc:
                    print(f"  [ERROR] job {job} raised: {exc}")
                    results.append(None)
                    batch_cuda = True

        if batch_cuda:
            consecutive_ok_batches = 0
            target_workers = min_workers
            recover_cuda_after_failure(wait_s=45.0)
        elif batch_ok == len(batch):
            consecutive_ok_batches += 1
            target_workers = min(max_workers, batch_workers)
            reset_cuda_device()
        else:
            consecutive_ok_batches = 0
            target_workers = max(min_workers, batch_workers - 1)
            reset_cuda_device()

        refresh_progress_md()

    refresh_progress_md()
    return results
