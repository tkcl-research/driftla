
from __future__ import annotations

from pathlib import Path as _Path
_REPO_ROOT = _Path(__file__).resolve().parents[2]
import argparse
import copy
import dataclasses
import json
import os
import sys
import time
from collections import defaultdict
from dataclasses import asdict
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.nn.functional as F
import torch.optim as optim
from tqdm import tqdm

from .src.arch_variants import ARCH_VARIANT_CHOICES, apply_arch_variant
from .src.config import DriftLAConfig, preset_config
from .src.auto_controller import AutoController, AutoControllerConfig, DensityDriftRouter
from .src.streaming.data import (
    create_negative_samples,
    load_ciao_chronological,
    load_gowala_chronological,
    load_ml1m_chronological,
    load_ml10m_chronological,
    load_ml20m_chronological,
    load_amazon23_category_chronological,
    load_yelp_chronological,
)
from .src.streaming.graph_manager import StreamingGraphManager
from .src.metrics import evaluate_on_batch
from .src.model import DriftLAModel
from .src.utils import set_seed


def _sample_history_edges(
    history: List[Tuple[int, int]],
    k: int,
    rng: np.random.Generator,
    timestamps: Optional[Dict[Tuple[int, int], float]] = None,
    now: Optional[float] = None,
    ts_scale: float = 1.0,
    recency_lambda: float = 0.0,
    recency_floor: float = 0.0,
) -> List[Tuple[int, int]]:
    if k <= 0 or not history:
        return []
    k = min(k, len(history))
    if recency_lambda > 0.0 and timestamps is not None and now is not None:
        ts = np.fromiter(
            (timestamps.get(edge, 0.0) for edge in history),
            dtype=np.float64,
            count=len(history),
        )
        scale = float(max(ts_scale, 1e-9))
        ages = np.clip((float(now) - ts) / scale, 0.0, None)
        logw = -float(recency_lambda) * ages

        logw = logw - logw.max()
        p = np.exp(logw)
        floor = float(np.clip(recency_floor, 0.0, 0.999))
        if floor > 0.0:
            p = floor + (1.0 - floor) * p
        p = p / p.sum()
        idx = rng.choice(len(history), size=k, replace=False, p=p)
    else:
        idx = rng.choice(len(history), size=k, replace=False)
    return [history[i] for i in idx]


def _build_driftla_model(
    cfg: DriftLAConfig,
    n_users: int,
    n_items: int,
    seed: int,
) -> DriftLAModel:
    return DriftLAModel(
        n_users,
        n_items,
        embed_dim=cfg.embed_dim,
        n_layers=cfg.n_layers,
        alpha=2,
        beta=4,
        tau=0.05,
        num_center_paths=cfg.num_center_paths,
        cache_size=100,
        replay_size=cfg.replay_size,
        drift_threshold=cfg.drift_threshold,
        alpha_drift=0.5,
        momentum=0.999,
        propensity_gamma=cfg.propensity_gamma,
        lambda_s=cfg.lambda_s,
        lambda_stab=cfg.lambda_stab,
        propensity_eta=cfg.propensity_eta,
        propensity_min=cfg.min_propensity,
        propensity_mix_uniform_batches=cfg.mix_uniform_batches,
        k_hop=2,
        path_contrast_users=cfg.path_contrast_users,
        cache_ttl_batches=cfg.cache_ttl_batches,
        cache_min_path_weight=cfg.cache_min_path_weight,
        seed=seed,

        item_bias=cfg.item_bias,
        drift_gate=cfg.drift_gate,
        drift_gate_threshold=cfg.drift_gate_threshold,
        path_contrast_users_quiet=cfg.path_contrast_users_quiet,
        use_stream_adapter=cfg.use_stream_adapter,
        stream_adapter_rank=cfg.stream_adapter_rank,
        stream_adapter_gamma_floor=cfg.stream_adapter_gamma_floor,
        stream_adapter_ramp=cfg.stream_adapter_ramp,
        stream_adapter_min_gate=cfg.stream_adapter_min_gate,
        adapter_gate_mode=cfg.adapter_gate_mode,
        adapter_fixed_gate=cfg.adapter_fixed_gate,
        lightgcn_ego_skip_alpha=cfg.lightgcn_ego_skip_alpha,
        node_contrast_lambda=cfg.node_contrast_lambda,
        node_contrast_eps=cfg.node_contrast_eps,
        node_contrast_temp=cfg.node_contrast_temp,
        node_contrast_max_nodes=cfg.node_contrast_max_nodes,
    )


def _compute_ts_scale(timestamps: Dict[Tuple[int, int], float]) -> float:
    if not timestamps:
        return 1.0
    vals = np.fromiter(timestamps.values(), dtype=np.float64)
    vals = vals[vals > 0.0]
    if vals.size == 0:
        return 1.0
    span = float(vals.max() - vals.min())
    return max(span, 1.0)


def _apply_density_drift_routing(
    cfg: DriftLAConfig,
    init_data: List[Tuple[int, int]],
    val_tail: List[Tuple[int, int]],
    n_users: int,
    n_items: int,
    model: "DriftLAModel",
    adj: Any,
    graph_mgr: Any,
    timestamps_dict: Any,
) -> Tuple[DriftLAConfig, Dict[str, Any]]:
    router = DensityDriftRouter(len(init_data), n_users, n_items)


    if val_tail:
        val_triples = [(u, i, timestamps_dict.get((u, i), 0.0)) for u, i in val_tail]
        _, _ = graph_mgr.add_edges(val_triples)
        adj_val = graph_mgr.get_adjacency().to(adj.device)

        model.eval()
        model.score_and_refresh_cache(adj_val, timestamps_dict)
        model.train()

    mean_val_drift = float(model.drift_detector.mean_drift)
    routing = router.route(mean_val_drift)


    lambda_s_new = float(routing["lambda_s"])
    cfg = dataclasses.replace(cfg, lambda_s=lambda_s_new)
    model.smoothness_reg.lambda_s = lambda_s_new

    path_contrast_enabled: bool = bool(routing["path_contrast_enabled"])
    if not path_contrast_enabled:
        cfg = dataclasses.replace(cfg, lambda_path_intra=0.0, lambda_path_inter=0.0)

    cfg = dataclasses.replace(cfg, path_contrast_users=int(routing["path_contrast_users"]))
    model.path_contrast_users = int(routing["path_contrast_users"])

    print(
        f"[auto_density_drift] density={routing['density']:.5f} "
        f"bucket={routing['density_bucket']} "
        f"mean_val_drift={mean_val_drift:.4f} "
        f"→ lambda_s={lambda_s_new} "
        f"path_contrast={path_contrast_enabled} "
        f"K_hi={routing['path_contrast_users']}"
    )
    return cfg, routing


def run_experiment(
    cfg: DriftLAConfig,
    init_data: List[Tuple[int, int]],
    batches: List[List[Tuple[int, int]]],
    timestamps: Dict[Tuple[int, int], float],
    n_users: int,
    n_items: int,
    device: torch.device,
    seed: int = 42,
    val_tail: Optional[List[Tuple[int, int]]] = None,
    routing_mode: str = "",
) -> Tuple[Dict[str, Any], DriftLAModel]:
    rng = np.random.default_rng(seed)
    set_seed(seed)

    model = _build_driftla_model(cfg, n_users, n_items, seed)
    model.set_path_sampler(init_data, timestamps)
    model = model.to(device)

    if cfg.drift_source and cfg.drift_source != "path":
        model.set_drift_source(cfg.drift_source)
    optimizer = optim.Adam(model.parameters(), lr=cfg.lr)

    graph_mgr = StreamingGraphManager(n_users, n_items, k_hop=2)
    init_triples = [(u, i, timestamps.get((u, i), 0.0)) for u, i in init_data]
    _, _ = graph_mgr.add_edges(init_triples)


    ts_scale = _compute_ts_scale(timestamps)

    auto = AutoController(
        AutoControllerConfig(
            enabled=bool(cfg.auto_enabled),
            gamma_drift_threshold=float(cfg.auto_gamma_drift_threshold),
            path_scale_drift=float(cfg.auto_path_scale_drift),
            smooth_scale_drift=float(cfg.auto_smooth_scale_drift),
            stab_scale_drift=float(cfg.auto_stab_scale_drift),
            distill_scale_drift=float(cfg.auto_distill_scale_drift),
            time_decay_lambda_scale_drift=float(cfg.auto_time_decay_lambda_scale_drift),
            recency_replay_lambda_scale_drift=float(cfg.auto_recency_replay_lambda_scale_drift),
        )
    )

    def _now_of(interactions: List[Tuple[int, int]]) -> float:
        if not interactions:
            return 0.0
        m = 0.0
        for e in interactions:
            t = timestamps.get(e, 0.0)
            if t > m:
                m = t
        return float(m)

    def _adj_for(now_val: float, time_decay_lambda: float) -> torch.Tensor:
        if cfg.use_time_decay:
            return graph_mgr.get_time_decayed_adjacency(
                now=now_val,
                lambda_e=float(time_decay_lambda),
                ts_scale=ts_scale,
                floor=cfg.time_decay_floor,
            ).to(device)
        return graph_mgr.get_adjacency().to(device)

    now_cur = _now_of(init_data)
    adj = _adj_for(now_cur, cfg.time_decay_lambda)

    all_interactions = list(init_data)
    user_pos: Dict[int, set] = defaultdict(set)
    for u, i in all_interactions:
        user_pos[u].add(i)

    timestamps_dict = graph_mgr.get_timestamps_dict()
    batch_size = cfg.batch_size


    t0 = time.time()
    for ep in range(1, cfg.warmup_epochs + 1):
        model.train()
        indices = rng.permutation(len(all_interactions))
        n_mb = max(1, (len(indices) + batch_size - 1) // batch_size)
        total_loss = 0.0
        for b in tqdm(range(n_mb), desc=f"Warmup {ep}", leave=False):
            lo = b * batch_size
            hi = min(lo + batch_size, len(indices))
            batch_idx = indices[lo:hi]
            users_np = np.array([all_interactions[j][0] for j in batch_idx])
            items_np = np.array([all_interactions[j][1] for j in batch_idx])
            users = torch.tensor(users_np, dtype=torch.long, device=device)
            pos_items = torch.tensor(items_np, dtype=torch.long, device=device)
            neg_items_np = create_negative_samples(users_np, user_pos, n_items)
            neg_items = torch.tensor(neg_items_np, dtype=torch.long, device=device)
            optimizer.zero_grad()
            loss, _ = model.compute_streaming_loss(
                users,
                pos_items,
                neg_items,
                adj,
                cfg.lambda_path_intra,
                cfg.lambda_path_inter,
                cfg.lambda_l2,
                timestamps_dict,
            )
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
        print(f"  Warmup ep {ep} loss={total_loss / n_mb:.4f}")
    warmup_time = time.time() - t0

    model.init_momentum_encoder()


    routing_metadata: Optional[Dict[str, Any]] = None
    if routing_mode == "auto_density_drift":
        cfg, routing_metadata = _apply_density_drift_routing(
            cfg,
            init_data,
            val_tail or [],
            n_users,
            n_items,
            model,
            adj,
            graph_mgr,
            timestamps_dict,
        )

    teacher: Optional[DriftLAModel] = None
    if cfg.distill_lambda > 0:
        teacher = _build_driftla_model(cfg, n_users, n_items, seed)
        teacher.set_path_sampler(init_data, timestamps)
        teacher.load_state_dict(copy.deepcopy(model.state_dict()), strict=True)
        teacher = teacher.to(device).eval()
        for p in teacher.parameters():
            p.requires_grad = False

    results = []
    stream_time = 0.0

    for b_idx, batch in enumerate(batches):
        t_step = time.time()
        model.set_stream_batch_id(b_idx)

        model.eval()
        with torch.no_grad():
            u_emb, i_emb = model(adj)
            metrics = evaluate_on_batch(u_emb, i_emb, batch, all_interactions)

        batch_triples = [(u, i, timestamps.get((u, i), 0.0)) for u, i in batch]
        _, affected = graph_mgr.add_edges(batch_triples)
        now_cur = max(now_cur, _now_of(batch))


        scales_now = auto.compute_scales(model.drift_detector.gamma)
        adj = _adj_for(now_cur, cfg.time_decay_lambda * scales_now["time_decay_lambda"])
        model.update_path_sampler_edges(batch, timestamps)
        model.path_cache.invalidate_by_affected_nodes(
            affected, total_nodes=graph_mgr.total_nodes,
        )

        history_before = all_interactions[:]
        all_interactions.extend(batch)
        for u, i in batch:
            user_pos[u].add(i)
        timestamps_dict = graph_mgr.get_timestamps_dict()

        model.train()
        model.smoothness_reg.take_snapshot(torch.cat([u_emb, i_emb], dim=0))


        if cfg.auto_enabled:
            model.smoothness_reg.lambda_s = float(cfg.lambda_s) * float(scales_now["smooth"])
            model.lambda_stab = float(cfg.lambda_stab) * float(scales_now["stab"])

        total_loss = 0.0
        n_mb_total = 0
        train_t0 = time.time()

        for _pass in range(cfg.streaming_passes):
            indices = rng.permutation(len(batch))
            n_mb = max(1, (len(indices) + batch_size - 1) // batch_size)
            for mb in range(n_mb):
                lo = mb * batch_size
                hi = min(lo + batch_size, len(indices))
                bi = indices[lo:hi]
                users_np = np.array([batch[j][0] for j in bi])
                items_np = np.array([batch[j][1] for j in bi])
                n_primary = len(users_np)

                extra = _sample_history_edges(
                    history_before,
                    cfg.history_replay,
                    rng,
                    timestamps=timestamps if cfg.recency_replay else None,
                    now=now_cur if cfg.recency_replay else None,
                    ts_scale=ts_scale,
                    recency_lambda=(
                        (cfg.recency_replay_lambda * scales_now["recency_replay_lambda"])
                        if cfg.recency_replay
                        else 0.0
                    ),
                    recency_floor=(
                        cfg.recency_replay_floor if cfg.recency_replay else 0.0
                    ),
                )
                if extra:
                    eu = np.array([e[0] for e in extra], dtype=np.int64)
                    ei = np.array([e[1] for e in extra], dtype=np.int64)
                    users_np = np.concatenate([users_np, eu])
                    items_np = np.concatenate([items_np, ei])

                users = torch.tensor(users_np, dtype=torch.long, device=device)
                pos_items = torch.tensor(items_np, dtype=torch.long, device=device)

                hard_neg_k = getattr(cfg, "hard_neg_k", 1)
                if hard_neg_k > 1:


                    neg_items_np = model.exposure_sampler.sample_negatives(
                        users_np, user_pos, hard_neg_k,
                    )
                    neg_items_np = neg_items_np.reshape(len(users_np), hard_neg_k)
                    neg_items = torch.tensor(neg_items_np, dtype=torch.long, device=device)
                else:
                    neg_items_np = model.exposure_sampler.sample_negatives(
                        users_np, user_pos, 1,
                    )
                    neg_items = torch.tensor(neg_items_np, dtype=torch.long, device=device)

                optimizer.zero_grad()
                node_cl_n = n_primary if cfg.node_contrast_lambda > 0 else None
                loss, _ = model.compute_streaming_loss(
                    users,
                    pos_items,
                    neg_items,
                    adj,
                    cfg.lambda_path_intra * float(scales_now["path"]),
                    cfg.lambda_path_inter * float(scales_now["path"]),
                    cfg.lambda_l2,
                    timestamps_dict,
                    node_cl_primary_n=node_cl_n,
                )

                if cfg.bpr_pop_weight:
                    u_e, i_e = model(adj)
                    pos_s = (u_e[users] * i_e[pos_items]).sum(1)
                    neg_s = (u_e[users] * i_e[neg_items]).sum(1)
                    bpr = F.softplus(neg_s - pos_s)
                    pos_cpu = pos_items.detach().cpu().numpy()
                    p = np.clip(
                        model.exposure_sampler.propensity[pos_cpu],
                        cfg.min_propensity,
                        None,
                    )
                    w = torch.tensor(1.0 / np.sqrt(p), dtype=bpr.dtype, device=bpr.device)
                    w = w / (w.mean() + 1e-8)


                    w = torch.clamp(w, cfg.bpr_pop_clip_low, cfg.bpr_pop_clip_high)
                    loss = loss + cfg.bpr_pop_scalar * (bpr * w).mean()

                eff_distill = float(cfg.distill_lambda) * float(scales_now["distill"])
                if (
                    cfg.distill_gamma_coupling > 0.0
                    and cfg.drift_gate
                    and teacher is not None
                    and cfg.distill_lambda > 0
                ):
                    boost = max(
                        0.0,
                        float(model.drift_detector.gamma)
                        - float(cfg.drift_gate_threshold),
                    )
                    eff_distill = float(cfg.distill_lambda) * (
                        1.0 + float(cfg.distill_gamma_coupling) * boost
                    )

                if cfg.distill_lambda > 0 and teacher is not None:
                    if cfg.propagated_distill:


                        s_u_all, s_i_all = model(adj)
                        with torch.no_grad():
                            t_u_all, t_i_all = teacher(adj)
                        s_u = s_u_all[users]
                        s_i = s_i_all[pos_items]
                        t_u = t_u_all[users]
                        t_i = t_i_all[pos_items]
                    else:
                        s_u = model.lightgcn.user_embedding(users)
                        s_i = model.lightgcn.item_embedding(pos_items)
                        with torch.no_grad():
                            t_u = teacher.lightgcn.user_embedding(users)
                            t_i = teacher.lightgcn.item_embedding(pos_items)
                    loss = (
                        loss
                        + eff_distill * (F.mse_loss(s_u, t_u) + F.mse_loss(s_i, t_i))
                    )

                loss.backward()
                optimizer.step()

                if teacher is not None and cfg.distill_lambda > 0:


                    if cfg.drift_gate and model.drift_detector.gamma >= cfg.drift_gate_threshold:
                        tau = cfg.distill_ema_drift
                    else:
                        tau = cfg.distill_ema
                    with torch.no_grad():
                        for ps, pt in zip(model.parameters(), teacher.parameters()):
                            pt.data.mul_(tau).add_(ps.data, alpha=1.0 - tau)

                total_loss += loss.item()
                n_mb_total += 1

        train_time = time.time() - train_t0

        with torch.no_grad():
            _ = model(adj)
        model.post_update(adj, batch, graph_mgr, b_idx, timestamps_dict=timestamps_dict)

        step_time = time.time() - t_step
        stream_time += step_time


        results.append(
            {
                "batch": b_idx + 1,
                "metrics": {k: float(v) for k, v in metrics.items()},
                "loss": total_loss / max(1, n_mb_total),
                "gamma_t": float(model.drift_detector.gamma),
                "mean_drift": float(model.drift_detector.mean_drift),
                "timings": {
                    "train_s": round(train_time, 2),
                    "total_s": round(step_time, 2),
                },
            }
        )
        print(
            f"  batch {b_idx + 1} R@10={metrics.get('Recall@10', 0):.4f} "
            f"time={step_time:.1f}s"
        )

    r10 = float(np.mean([x["metrics"].get("Recall@10", 0) for x in results]))
    n10 = float(np.mean([x["metrics"].get("NDCG@10", 0) for x in results]))

    summary = {
        "config": cfg.to_json_dict(),
        "warmup_time_s": round(warmup_time, 1),
        "stream_time_s": round(stream_time, 1),
        "total_time_s": round(warmup_time + stream_time, 1),
        "avg_recall10": r10,
        "avg_ndcg10": n10,
        "device": str(device),
        "seed": seed,
        "n_users": n_users,
        "n_items": n_items,
        "batches": results,

        "drift_history": model.drift_detector.get_history(),

        "routing_metadata": routing_metadata,
    }
    return summary, model


def main() -> None:
    parser = argparse.ArgumentParser(description="DriftLA training (ML-1M/Ciao/Gowalla/Yelp streaming)")
    parser.add_argument(
        "--data_root",
        type=str,
        default="data",
        help="Contains ml-1m/, ciao/, gowala/, or yelp/ (see driftla/data.py)",
    )
    parser.add_argument(
        "--dataset",
        type=str,
        default="ml-1m",
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
        ),
        help="Dataset name (subfolder under data_root). "
             "*_dense variants apply a higher k-core filter (Gowalla 20-core, Yelp 10-core).",
    )
    parser.add_argument(
        "--max_interactions",
        type=int,
        default=300000,
        help="Gowalla/Yelp/ML-*M/Amazon'23: cap earliest interactions (<=0 disables cap).",
    )
    parser.add_argument(
        "--kcore",
        type=int,
        default=0,
        help="Gowalla/Yelp/Amazon'23: symmetric k-core (min user AND min item degree). "
             "0 = use dataset default (5 for standard Gowalla/Yelp/Amazon, 20/10 for *_dense). "
             "Higher k yields a denser subgraph (smaller n_users/n_items for the same 300k cap).",
    )
    parser.add_argument(
        "--amz23_category",
        type=str,
        default="",
        help="Amazon Reviews'23 category folder name, e.g. Kindle_Store, Software (required for "
             "--dataset amazon23; raw file data/amazon23/raw/review_categories/<name>.jsonl.gz).",
    )
    parser.add_argument("--device", type=str, default="cpu", choices=["cpu", "cuda"])
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--out_json", type=str, default="", help="Write metrics JSON here")
    parser.add_argument("--checkpoint", type=str, default="", help="Save model state_dict here")
    parser.add_argument("--smoke", action="store_true", help="2 batches, 1 warmup epoch (fast test)")
    parser.add_argument(
        "--max_stream_batches",
        type=int,
        default=0,
        help="If >0, truncate streaming to the first N batches after the split (fast ablations; "
        "metrics are not comparable to the full 10-batch protocol unless N=10).",
    )
    parser.add_argument(
        "--preset",
        type=str,
        default="final",
        choices=(
            "final",
            "bundle",
            "legacy_b2",
            "paper_100_batch",
            "paper_100_warmup",
            "paper_100_both",
            "v3",
            "v3_bundle_acc",
            "v3_auto",
            "v3_champion",
            "v4_improved",
            "v3_sparse",
            "v3_sparse_uniform",
        ),
        help="Streaming paper presets (driftla/README.md).",
    )
    parser.add_argument(
        "--use_dataset_preset",
        action="store_true",
        help="Layer per-dataset feedback presets (history_replay, distill_lambda) on top of --preset.",
    )
    parser.add_argument("--distill_lambda", type=float, default=None,
                        help="Override config.distill_lambda (e.g. 0.0 for high-drift streams).")
    parser.add_argument("--history_replay", type=int, default=None,
                        help="Override config.history_replay (e.g. 50000 for Gowalla/Yelp).")
    parser.add_argument("--batch_size", type=int, default=None,
                        help="Override config.batch_size (lower for VRAM-heavy datasets).")
    parser.add_argument("--warmup_epochs", type=int, default=None,
                        help="Override config.warmup_epochs (e.g. 3).")
    parser.add_argument("--streaming_passes", type=int, default=None,
                        help="Override config.streaming_passes (e.g. 3).")
    parser.add_argument("--num_center_paths", type=int, default=None,
                        help="Override config.num_center_paths (path supervision width per anchor).")
    parser.add_argument("--lambda_path_intra", type=float, default=None,
                        help="Override config.lambda_path_intra.")
    parser.add_argument("--lambda_path_inter", type=float, default=None,
                        help="Override config.lambda_path_inter.")
    parser.add_argument("--lambda_s", type=float, default=None,
                        help="Override config.lambda_s (temporal smoothness weight).")
    parser.add_argument("--lambda_stab", type=float, default=None,
                        help="Override config.lambda_stab (replay stability weight).")
    parser.add_argument("--no_bpr_pop_weight", action="store_true",
                        help="Disable the secondary BPR popularity-weighted term.")
    parser.add_argument(
        "--no_stream_adapter",
        action="store_true",
        help="Disable the M7 drift-gated low-rank adapter (ablation vs full v3).",
    )
    parser.add_argument(
        "--adapter_gate_mode",
        type=str,
        default="",
        choices=["", "drift", "ungated", "fixed", "random", "plain_lora"],
        help="Adapter gate ablation: drift (default), ungated, fixed, random, plain_lora.",
    )
    parser.add_argument(
        "--adapter_fixed_gate",
        type=float,
        default=None,
        help="Constant gate value when --adapter_gate_mode=fixed (default 0.5).",
    )
    parser.add_argument("--path_contrast_users", type=int, default=None,
                        help="Override config.path_contrast_users (default 256).")
    parser.add_argument(
        "--path_contrast_users_quiet",
        type=int,
        default=None,
        help="Override config.path_contrast_users_quiet (v3 drift-gated DTPC quiet width).",
    )
    parser.add_argument(
        "--time_decay_lambda",
        type=float,
        default=None,
        help="Override config.time_decay_lambda (v3 TDA strength).",
    )
    parser.add_argument(
        "--time_decay_floor",
        type=float,
        default=None,
        help="Override config.time_decay_floor (v3 TDA floor; anti-forgetting).",
    )
    parser.add_argument(
        "--recency_replay_lambda",
        type=float,
        default=None,
        help="Override config.recency_replay_lambda (v3 RBR strength).",
    )
    parser.add_argument(
        "--recency_replay_floor",
        type=float,
        default=None,
        help="Override config.recency_replay_floor (v3 RBR floor; anti-forgetting).",
    )
    parser.add_argument(
        "--mix_uniform_batches",
        type=int,
        default=None,
        help="Override config.mix_uniform_batches (how long to mix uniform negatives).",
    )
    parser.add_argument(
        "--distill_gamma_coupling",
        type=float,
        default=None,
        help="Override config.distill_gamma_coupling (M6 drift-scaled distill).",
    )
    parser.add_argument(
        "--arch_variant",
        type=str,
        default="",
        choices=["", *ARCH_VARIANT_CHOICES],
        help="Named architecture patch for paper ablations (see driftla/src/arch_variants.py).",
    )

    parser.add_argument(
        "--no_time_decay",
        action="store_true",
        help="Ablation: disable M1 time-decay adjacency (use static normalized adjacency).",
    )
    parser.add_argument(
        "--no_item_bias",
        action="store_true",
        help="Ablation: disable M2 learnable item bias in BPR scores.",
    )
    parser.add_argument(
        "--no_recency_replay",
        action="store_true",
        help="Ablation: disable M3 recency-biased history replay (uniform replay).",
    )
    parser.add_argument(
        "--no_drift_gate",
        action="store_true",
        help="Ablation: disable M4 drift-triggered path-contrast width (fixed wide cap).",
    )
    parser.add_argument(
        "--no_distill",
        action="store_true",
        help="Ablation: disable M5 teacher distillation (distill_lambda=0).",
    )
    parser.add_argument(
        "--no_path_contrast",
        action="store_true",
        help="Ablation: disable path intra/inter contrast (lambda_path_intra/inter=0).",
    )
    parser.add_argument(
        "--no_temporal_smooth",
        action="store_true",
        help="Ablation: disable temporal smoothness regularizer (lambda_s=0).",
    )
    parser.add_argument(
        "--uniform_negatives",
        action="store_true",
        help="Ablation: IPS exponent gamma=0 (uniform negative sampling weights vs propensity-shaped).",
    )
    parser.add_argument(
        "--no_replay_stability",
        action="store_true",
        help="Ablation: disable replay-buffer stability term (lambda_stab=0).",
    )
    parser.add_argument(
        "--ablation_id",
        type=str,
        default="",
        help="Optional tag stored in JSON (e.g. no_m1_tda) for harness scripts.",
    )

    parser.add_argument(
        "--val_tail_frac",
        type=float,
        default=0.0,
        help="Fraction of the warmup prefix to hold out as a validation tail "
             "(e.g. 0.2 = last 20%% of init_data). Used with --routing auto_density_drift "
             "so that config selection never sees test batches.",
    )
    parser.add_argument(
        "--routing",
        type=str,
        default="",
        choices=["", "auto_density_drift"],
        help="Pre-test routing rule.  'auto_density_drift' applies the frozen "
             "density/drift rule from §E1 on the val tail before the first test batch.",
    )

    parser.add_argument(
        "--drift_source",
        type=str,
        default="",
        choices=["", "path", "node"],
        help="Drift source for the drift detector.  'node' uses node-embedding drift "
             "(no path sampling) → defines the cheap DriftLA-Adapter variant.",
    )
    parser.add_argument(
        "--hard_neg_k",
        type=int,
        default=None,
        help="DriftLA v4: hard negative pool size. Sample K candidates per user via IPS, "
             "then select the hardest (highest dot-product score) for BPR. "
             "1 = disabled (v3 behavior). Recommended: 8.",
    )
    parser.add_argument(
        "--n_layers",
        type=int,
        default=None,
        help="Override config.n_layers (LightGCN propagation depth).",
    )
    parser.add_argument(
        "--node_contrast_lambda",
        type=float,
        default=None,
        help="Weight of SimGCL-style node uniformity loss (0 = disabled).",
    )
    parser.add_argument(
        "--node_contrast_eps",
        type=float,
        default=None,
        help="Uniform noise scale ε for node contrast views.",
    )
    parser.add_argument(
        "--node_contrast_temp",
        type=float,
        default=None,
        help="InfoNCE temperature τ for node contrast.",
    )
    parser.add_argument(
        "--sparse_mode",
        action="store_true",
        help="Sparse-graph adaptation: disables path contrast and EMA distillation, "
             "zeros anti-forgetting floors (time_decay_floor, recency_replay_floor). "
             "Maximises plasticity for datasets where continual anchoring hurts more than it helps. "
             "Applied AFTER --preset / --use_dataset_preset, BEFORE other --no_* flags.",
    )
    args = parser.parse_args()

    if args.dataset == "amazon23" and not (args.amz23_category or "").strip():
        parser.error("--amz23_category is required when --dataset amazon23")


    if os.getenv("DriftLA_FORCE_CPU", "0").strip() in ("1", "true", "True", "yes", "YES"):
        if os.getenv("DriftLA_REQUIRE_GPU", "0").strip() in ("1", "true", "True", "yes", "YES"):
            raise SystemExit(
                "DriftLA_REQUIRE_GPU=1 but DriftLA_FORCE_CPU=1; refusing CPU run."
            )
        args.device = "cpu"
    if (
        os.getenv("DriftLA_REQUIRE_GPU", "0").strip() in ("1", "true", "True", "yes", "YES")
        and args.device == "cpu"
    ):
        raise SystemExit(
            "DriftLA_REQUIRE_GPU=1; refusing --device cpu. Use --device cuda."
        )
    if os.getenv("DriftLA_CUDA_SAFE", "0").strip() in ("1", "true", "True", "yes", "YES"):

        try:
            torch.backends.cuda.matmul.allow_tf32 = False
        except Exception:
            pass
        try:
            torch.backends.cudnn.allow_tf32 = False
            torch.backends.cudnn.benchmark = False
            torch.backends.cudnn.deterministic = True
        except Exception:
            pass


    set_seed(args.seed)

    init_ratio = 0.5
    if args.smoke:
        cfg0 = preset_config(args.preset)
        init_ratio = 0.02
        cfg = dataclasses.replace(
            cfg0,
            batch_size=1024,
            warmup_epochs=1,
            streaming_passes=1,
            history_replay=min(512, cfg0.history_replay),
            replay_size=min(100, cfg0.replay_size),
        )
    else:
        cfg = preset_config(args.preset)
        if args.use_dataset_preset:
            if args.preset == "v3_bundle_acc":
                cfg = DriftLAConfig.for_dataset_v3_bundle_acc(args.dataset, cfg)
            elif args.preset == "v3_champion":
                cfg = DriftLAConfig.for_dataset_v3_champion(args.dataset, cfg)
            elif args.preset == "v4_improved":
                cfg = DriftLAConfig.for_dataset_v4_improved(args.dataset, cfg)
            elif args.preset == "v3_sparse":
                cfg = DriftLAConfig.for_dataset_v3_sparse(args.dataset, cfg)
            elif args.preset == "v3_sparse_uniform":
                cfg = DriftLAConfig.for_dataset_v3_sparse_uniform(args.dataset, cfg)
            else:
                cfg = DriftLAConfig.for_dataset(args.dataset, cfg)
        if args.distill_lambda is not None:
            cfg = dataclasses.replace(cfg, distill_lambda=float(args.distill_lambda))
        if args.history_replay is not None:
            cfg = dataclasses.replace(cfg, history_replay=int(args.history_replay))
        if args.batch_size is not None:
            cfg = dataclasses.replace(cfg, batch_size=int(args.batch_size))
        if args.warmup_epochs is not None:
            cfg = dataclasses.replace(cfg, warmup_epochs=int(args.warmup_epochs))
        if args.streaming_passes is not None:
            cfg = dataclasses.replace(cfg, streaming_passes=int(args.streaming_passes))
        if args.num_center_paths is not None:
            cfg = dataclasses.replace(cfg, num_center_paths=int(args.num_center_paths))
        if args.lambda_path_intra is not None:
            cfg = dataclasses.replace(cfg, lambda_path_intra=float(args.lambda_path_intra))
        if args.lambda_path_inter is not None:
            cfg = dataclasses.replace(cfg, lambda_path_inter=float(args.lambda_path_inter))
        if args.lambda_s is not None:
            cfg = dataclasses.replace(cfg, lambda_s=float(args.lambda_s))
        if args.lambda_stab is not None:
            cfg = dataclasses.replace(cfg, lambda_stab=float(args.lambda_stab))
        if args.path_contrast_users is not None:
            cfg = dataclasses.replace(cfg, path_contrast_users=int(args.path_contrast_users))
        if args.path_contrast_users_quiet is not None:
            cfg = dataclasses.replace(
                cfg, path_contrast_users_quiet=int(args.path_contrast_users_quiet),
            )
        if args.time_decay_lambda is not None:
            cfg = dataclasses.replace(cfg, time_decay_lambda=float(args.time_decay_lambda))
        if args.time_decay_floor is not None:
            cfg = dataclasses.replace(cfg, time_decay_floor=float(args.time_decay_floor))
        if args.recency_replay_lambda is not None:
            cfg = dataclasses.replace(
                cfg, recency_replay_lambda=float(args.recency_replay_lambda),
            )
        if args.recency_replay_floor is not None:
            cfg = dataclasses.replace(cfg, recency_replay_floor=float(args.recency_replay_floor))
        if args.distill_gamma_coupling is not None:
            cfg = dataclasses.replace(
                cfg, distill_gamma_coupling=float(args.distill_gamma_coupling),
            )
        if args.mix_uniform_batches is not None:
            cfg = dataclasses.replace(cfg, mix_uniform_batches=int(args.mix_uniform_batches))
        if args.no_bpr_pop_weight:
            cfg = dataclasses.replace(cfg, bpr_pop_weight=False)
        if args.hard_neg_k is not None:
            cfg = dataclasses.replace(cfg, hard_neg_k=int(args.hard_neg_k))
        if args.n_layers is not None:
            cfg = dataclasses.replace(cfg, n_layers=int(args.n_layers))
        if args.node_contrast_lambda is not None:
            cfg = dataclasses.replace(cfg, node_contrast_lambda=float(args.node_contrast_lambda))
        if args.node_contrast_eps is not None:
            cfg = dataclasses.replace(cfg, node_contrast_eps=float(args.node_contrast_eps))
        if args.node_contrast_temp is not None:
            cfg = dataclasses.replace(cfg, node_contrast_temp=float(args.node_contrast_temp))

    if args.no_stream_adapter:
        cfg = dataclasses.replace(cfg, use_stream_adapter=False)
    if args.adapter_gate_mode:
        cfg = dataclasses.replace(cfg, adapter_gate_mode=args.adapter_gate_mode)
    if args.adapter_fixed_gate is not None:
        cfg = dataclasses.replace(cfg, adapter_fixed_gate=float(args.adapter_fixed_gate))
    if args.no_time_decay:
        cfg = dataclasses.replace(cfg, use_time_decay=False)
    if args.no_item_bias:
        cfg = dataclasses.replace(cfg, item_bias=False)
    if args.no_recency_replay:
        cfg = dataclasses.replace(cfg, recency_replay=False)
    if args.no_drift_gate:
        cfg = dataclasses.replace(cfg, drift_gate=False)
    if args.no_distill:
        cfg = dataclasses.replace(cfg, distill_lambda=0.0)
    if args.no_path_contrast:
        cfg = dataclasses.replace(cfg, lambda_path_intra=0.0, lambda_path_inter=0.0)
    if args.no_temporal_smooth:
        cfg = dataclasses.replace(cfg, lambda_s=0.0)
    if args.uniform_negatives:
        cfg = dataclasses.replace(cfg, propensity_gamma=0.0)
    if args.no_replay_stability:
        cfg = dataclasses.replace(cfg, lambda_stab=0.0)


    if getattr(args, "drift_source", ""):
        cfg = dataclasses.replace(cfg, drift_source=str(args.drift_source))

    if args.sparse_mode:


        cfg = dataclasses.replace(
            cfg,
            lambda_path_intra=0.0,
            lambda_path_inter=0.0,
            distill_lambda=0.0,
            time_decay_floor=0.0,
            recency_replay_floor=0.0,
        )

    cfg = apply_arch_variant(cfg, args.dataset, getattr(args, "arch_variant", "") or "")

    if os.getenv("DriftLA_CUDA_SAFE", "0").strip() in ("1", "true", "True", "yes", "YES"):
        cfg = dataclasses.replace(
            cfg,
            batch_size=min(cfg.batch_size, 256),
            history_replay=min(cfg.history_replay, 4096),
            replay_size=min(cfg.replay_size, 200),
        )

    if args.dataset == "ciao":
        init_data, batches, timestamps, n_users, n_items = load_ciao_chronological(
            args.data_root, init_ratio=init_ratio, n_batches=10,
        )
    elif args.dataset in ("amz23_digital_music", "amz23_all_beauty", "amazon23"):

        k = int(args.kcore) if args.kcore > 0 else 5
        max_interactions = None if args.max_interactions <= 0 else args.max_interactions
        if args.dataset == "amz23_digital_music":
            cat = "Digital_Music"
        elif args.dataset == "amz23_all_beauty":
            cat = "All_Beauty"
        else:
            cat = (args.amz23_category or "").strip()
        init_data, batches, timestamps, n_users, n_items = load_amazon23_category_chronological(
            args.data_root,
            category=cat,
            init_ratio=init_ratio,
            n_batches=10,
            min_user_interactions=k,
            min_item_interactions=k,
            max_interactions=max_interactions,
            min_rating=4.0,
        )
    elif args.dataset == "ml-10m":
        max_interactions = None if args.max_interactions <= 0 else args.max_interactions
        init_data, batches, timestamps, n_users, n_items = load_ml10m_chronological(
            args.data_root,
            init_ratio=init_ratio,
            n_batches=10,
            max_interactions=max_interactions,
        )
    elif args.dataset == "ml-20m":
        max_interactions = None if args.max_interactions <= 0 else args.max_interactions
        init_data, batches, timestamps, n_users, n_items = load_ml20m_chronological(
            args.data_root,
            init_ratio=init_ratio,
            n_batches=10,
            max_interactions=max_interactions,
        )
    elif args.dataset in ("gowala", "gowala_dense"):
        max_interactions = None if args.max_interactions <= 0 else args.max_interactions
        if args.kcore > 0:
            k = args.kcore
        else:
            k = 20 if args.dataset == "gowala_dense" else 5
        init_data, batches, timestamps, n_users, n_items = load_gowala_chronological(
            args.data_root,
            init_ratio=init_ratio,
            n_batches=10,
            min_user_interactions=k,
            min_item_interactions=k,
            max_interactions=max_interactions,
        )
    elif args.dataset in ("yelp", "yelp_dense"):
        max_interactions = None if args.max_interactions <= 0 else args.max_interactions
        if args.kcore > 0:
            k = args.kcore
        else:
            k = 10 if args.dataset == "yelp_dense" else 5
        init_data, batches, timestamps, n_users, n_items = load_yelp_chronological(
            args.data_root,
            init_ratio=init_ratio,
            n_batches=10,
            min_user_interactions=k,
            min_item_interactions=k,
            max_interactions=max_interactions,
        )
    else:
        init_data, batches, timestamps, n_users, n_items = load_ml1m_chronological(
            args.data_root, init_ratio=init_ratio, n_batches=10,
        )
    if args.smoke:
        batches = batches[:2]
    elif args.max_stream_batches and args.max_stream_batches > 0:
        batches = batches[: int(args.max_stream_batches)]

    print(
        f"[data] dataset={args.dataset} users={n_users} items={n_items} "
        f"init={len(init_data)} batches={len(batches)} "
        f"batch_size={cfg.batch_size} history_replay={cfg.history_replay}",
        flush=True,
    )


    val_tail: Optional[List[Tuple[int, int]]] = None
    val_tail_frac = float(getattr(args, "val_tail_frac", 0.0))
    routing_mode  = str(getattr(args, "routing", "") or "")
    if val_tail_frac > 0.0 and not args.smoke:
        split = int(len(init_data) * (1.0 - val_tail_frac))
        val_tail  = init_data[split:]
        init_data = init_data[:split]
        print(
            f"[val_tail] warmup={len(init_data)} val={len(val_tail)} "
            f"(frac={val_tail_frac:.2f})"
        )

    device = torch.device(args.device)
    if device.type == "cuda":
        try:
            torch.cuda.empty_cache()
            torch.cuda.synchronize()
        except RuntimeError as exc:
            if _is_cuda_runtime_error(exc):
                print(f"\n[CUDA ERROR] {exc}", file=sys.stderr)
                _release_gpu(after_cuda_error=True)
                raise SystemExit(2) from exc
            raise

    trained = None
    try:
        out, trained = run_experiment(
            cfg,
            init_data,
            batches,
            timestamps,
            n_users,
            n_items,
            device,
            seed=args.seed,
            val_tail=val_tail,
            routing_mode=routing_mode,
        )
    except RuntimeError as exc:
        if _is_cuda_runtime_error(exc):
            print(f"\n[CUDA ERROR] {exc}", file=sys.stderr)
            _release_gpu(trained, after_cuda_error=True)
            raise SystemExit(2) from exc
        raise

    print(
        f"\nDone. avg R@10={out['avg_recall10']:.4f} avg NDCG@10={out['avg_ndcg10']:.4f} "
        f"stream_s={out['stream_time_s']:.1f} total_s={out['total_time_s']:.1f}"
    )

    out_json = args.out_json
    if not out_json:
        tag = args.dataset.replace("-", "")
        if args.dataset == "ml-1m":
            tag = "ml1m"
        out_json = str(_REPO_ROOT / "results" / "driftla" / f"driftla_run_{tag}_seed{args.seed}.json")
    if out_json:
        os.makedirs(os.path.dirname(out_json) or ".", exist_ok=True)
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
        payload = {
            **out,
            "streaming_preset": args.preset if not args.smoke else "smoke",
            "dataset": args.dataset,
            "amz23_category": (args.amz23_category or "").strip() or None,
            "kcore": kcore_used,
            "arch_variant": (getattr(args, "arch_variant", None) or "baseline"),
            "ablation_id": (getattr(args, "ablation_id", None) or "").strip() or None,
        }
        with open(out_json, "w") as f:
            json.dump(payload, f, indent=2)
        print(f"Wrote {out_json}")

    if args.checkpoint:
        os.makedirs(os.path.dirname(args.checkpoint) or ".", exist_ok=True)
        max_inter = None if args.max_interactions <= 0 else int(args.max_interactions)
        torch.save(
            {
                "model_state_dict": trained.state_dict(),
                "config": asdict(cfg),
                "n_users": n_users,
                "n_items": n_items,
                "seed": args.seed,
                "dataset": args.dataset,
                "dataloader_meta": {
                    "dataset": args.dataset,
                    "init_ratio": init_ratio,
                    "stream_batches": len(batches),
                    "max_interactions": max_inter,
                },
            },
            args.checkpoint,
        )
        print(f"Wrote checkpoint {args.checkpoint}")

    _release_gpu(trained)


def _release_gpu(*objs: object, after_cuda_error: bool = False) -> None:
    import gc
    for obj in objs:
        if obj is None:
            continue
        try:
            del obj
        except Exception:
            pass
    gc.collect()
    if not torch.cuda.is_available():
        return

    if after_cuda_error:
        return
    for fn in (torch.cuda.synchronize, torch.cuda.empty_cache, torch.cuda.ipc_collect):
        try:
            fn()
        except RuntimeError:
            break
        except Exception:
            pass


def _is_cuda_runtime_error(exc: BaseException) -> bool:
    msg = str(exc).lower()
    return any(
        token in msg
        for token in (
            "cuda",
            "cudnn",
            "cublas",
            "nccl",
            "device-side assert",
            "launch failure",
        )
    )


if __name__ == "__main__":
    main()
