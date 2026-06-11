
from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from typing import TYPE_CHECKING, Dict, List, Optional, Tuple

from .encoder.lightgcn import LightGCN
from .paths.sampling import PathSampler
from .paths.encoding import TemporalEncoder, encode_path_with_temporal
from .paths.contrastive import ContrastiveLoss, bpr_loss
from .paths.cache import DynamicPathCache
from .drift.detector import DriftDetector
from .continual.replay import TopologyPreservingReplayBuffer
from .continual.exposure import ExposureCalibratedSampler
from .continual.smoothness import TemporalSmoothnessRegularizer
from .continual.momentum import MomentumEncoder
from .continual.stream_adapter import DriftGatedStreamAdapter

if TYPE_CHECKING:
    from .streaming.graph_manager import StreamingGraphManager


class DriftLAModel(nn.Module):

    def __init__(
        self,
        n_users: int,
        n_items: int,
        embed_dim: int = 64,
        n_layers: int = 3,
        alpha: int = 2,
        beta: int = 4,
        tau: float = 0.05,
        num_center_paths: int = 5,
        num_positive_paths: int = 5,
        path_length: int = 5,
        cache_size: int = 100,
        replay_size: int = 500,
        momentum: float = 0.999,
        lambda_s: float = 0.01,
        lambda_stab: float = 0.001,
        drift_threshold: float = 0.3,
        alpha_drift: float = 0.5,
        propensity_gamma: float = 0.1,
        propensity_eta: float = 0.1,
        propensity_min: float = 1e-3,
        propensity_mix_uniform_batches: int = 3,
        k_hop: int = 2,
        path_contrast_users: int = 256,
        cache_ttl_batches: int = 2,
        cache_min_path_weight: float = 0.5,
        seed: int = 42,

        item_bias: bool = False,
        drift_gate: bool = False,
        drift_gate_threshold: float = 1.05,
        path_contrast_users_quiet: int = 32,
        use_stream_adapter: bool = False,
        stream_adapter_rank: int = 8,
        stream_adapter_gamma_floor: float = 1.0,
        stream_adapter_ramp: float = 0.18,
        stream_adapter_min_gate: float = 0.07,
        adapter_gate_mode: str = "drift",
        adapter_fixed_gate: float = 0.5,
        lightgcn_ego_skip_alpha: float = 0.0,
        node_contrast_lambda: float = 0.0,
        node_contrast_eps: float = 0.2,
        node_contrast_temp: float = 0.2,
        node_contrast_max_nodes: int = 2048,
    ):
        super().__init__()
        self.n_users = n_users
        self.n_items = n_items
        self.embed_dim = embed_dim
        self.alpha = alpha
        self.beta = beta
        self.tau = tau
        self.num_center_paths = num_center_paths
        self.num_positive_paths = num_positive_paths
        self.path_length = path_length

        self.lightgcn = LightGCN(
            n_users,
            n_items,
            embed_dim,
            n_layers,
            ego_skip_alpha=float(lightgcn_ego_skip_alpha),
        )
        self.temporal_encoder = TemporalEncoder(embed_dim)
        self.contrastive_loss = ContrastiveLoss(tau)


        self.item_bias_enabled = bool(item_bias)
        if self.item_bias_enabled:
            self.item_bias = nn.Embedding(n_items, 1)
            nn.init.zeros_(self.item_bias.weight)
        else:
            self.item_bias = None

        self.path_sampler: Optional[PathSampler] = None

        self.path_cache = DynamicPathCache(
            max_paths_per_node=cache_size,
            drift_threshold_high=drift_threshold + 0.2,
            drift_threshold_low=drift_threshold * 0.3,
        )
        self.drift_detector = DriftDetector(
            drift_threshold=drift_threshold,
            alpha_drift=alpha_drift,
        )
        self.replay_buffer = TopologyPreservingReplayBuffer(
            max_size=replay_size, k_hop=k_hop,
        )
        self.exposure_sampler = ExposureCalibratedSampler(
            n_items=n_items,
            gamma=propensity_gamma,
            eta_p=propensity_eta,
            min_propensity=propensity_min,
            mix_uniform_batches=propensity_mix_uniform_batches,
        )
        self.smoothness_reg = TemporalSmoothnessRegularizer(lambda_s=lambda_s)
        self.lambda_stab = lambda_stab

        self.momentum_encoder: Optional[MomentumEncoder] = None
        self._momentum_value = momentum


        self.path_contrast_users = path_contrast_users
        self.cache_ttl_batches = cache_ttl_batches
        self.cache_min_path_weight = cache_min_path_weight
        self._seed = seed


        self.drift_gate_enabled = bool(drift_gate)
        self.drift_gate_threshold = float(drift_gate_threshold)
        self.path_contrast_users_quiet = int(path_contrast_users_quiet)

        self.stream_adapter: Optional[DriftGatedStreamAdapter] = None
        self.stream_adapter_gamma_floor = float(stream_adapter_gamma_floor)
        self.stream_adapter_ramp = float(stream_adapter_ramp)
        self.stream_adapter_min_gate = float(stream_adapter_min_gate)
        self.adapter_gate_mode = str(adapter_gate_mode)
        if use_stream_adapter:
            self.stream_adapter = DriftGatedStreamAdapter(
                embed_dim,
                rank=stream_adapter_rank,
                gate_mode=adapter_gate_mode,
                fixed_gate=adapter_fixed_gate,
                gate_seed=seed,
            )

        self.node_contrast_lambda = float(node_contrast_lambda)
        self.node_contrast_eps = float(node_contrast_eps)
        self.node_contrast_temp = float(node_contrast_temp)
        self.node_contrast_max_nodes = int(node_contrast_max_nodes)


        self._drift_source: str = "path"
        self._prev_all_emb: Optional[torch.Tensor] = None

    def init_momentum_encoder(self) -> None:
        self.momentum_encoder = MomentumEncoder(self.lightgcn, self._momentum_value)

    def set_path_sampler(self, interactions, timestamps=None):
        self.path_sampler = PathSampler(
            self.n_users, self.n_items, interactions, timestamps,
            path_length=self.path_length, seed=self._seed,
        )

    def update_path_sampler_edges(self, new_interactions, timestamps=None):
        if self.path_sampler is None:
            self.set_path_sampler(new_interactions, timestamps)
            return
        timestamps = timestamps or {}
        for u, i in new_interactions:
            item_nid = i + self.n_users
            self.path_sampler.neighbors[u].append(item_nid)
            self.path_sampler.neighbors[item_nid].append(u)
            ts = timestamps.get((u, i), 0.0)
            self.path_sampler.edge_ts[(u, item_nid)] = ts
            self.path_sampler.edge_ts[(item_nid, u)] = ts

    def set_drift_source(self, source: str) -> None:
        if source not in ("path", "node"):
            raise ValueError(f"drift_source must be 'path' or 'node', got {source!r}")
        self._drift_source = source

    def set_stream_batch_id(self, batch_id: int) -> None:
        if self.stream_adapter is not None:
            self.stream_adapter.set_stream_batch_id(batch_id)

    def forward(self, adj_matrix: torch.Tensor):
        user_emb, item_emb = self.lightgcn(adj_matrix)
        if self.stream_adapter is not None:
            user_emb, item_emb = self.stream_adapter(
                user_emb,
                item_emb,
                self.drift_detector.gamma,
                self.stream_adapter_gamma_floor,
                self.stream_adapter_ramp,
                self.stream_adapter_min_gate,
            )
        return user_emb, item_emb

    def encode_path(
        self,
        path: List[int],
        all_emb: torch.Tensor,
        temporal_encoder=None,
        timestamps_dict: Optional[Dict] = None,
        n_users: int = 0,
    ) -> torch.Tensor:
        if temporal_encoder is None:
            temporal_encoder = self.temporal_encoder
        if n_users == 0:
            n_users = self.n_users

        node_embs = all_emb[path]
        edge_ts = []
        for i in range(len(path) - 1):
            a, b = path[i], path[i + 1]
            if a < n_users:
                key = (a, b - n_users)
            else:
                key = (b, a - n_users)
            ts = timestamps_dict.get(key, 0.0) if timestamps_dict else 0.0
            edge_ts.append(ts)

        edge_ts_t = torch.tensor(edge_ts, device=all_emb.device, dtype=torch.float32)
        return encode_path_with_temporal(node_embs, edge_ts_t, temporal_encoder)

    def compute_drift_weighted_intra_loss(
        self,
        target_nodes: np.ndarray,
        user_emb: torch.Tensor,
        item_emb: torch.Tensor,
        gamma: float = 1.0,
        timestamps_dict: Optional[Dict] = None,
    ) -> torch.Tensor:
        if self.path_sampler is None:
            return torch.tensor(0.0, device=user_emb.device)

        all_emb = torch.cat([user_emb, item_emb], dim=0)
        total_nodes = self.n_users + self.n_items
        losses = []

        for node in target_nodes:


            weighted_paths = self.path_cache.get_weighted_paths(
                node,
                ttl_batches=self.cache_ttl_batches,
                min_weight=self.cache_min_path_weight,
            )

            if not weighted_paths:
                center_paths, _, positive_nodes = self.path_sampler.sample_paths_for_node(
                    node, self.num_center_paths, self.alpha, self.beta, self.num_positive_paths,
                )
                positive_nodes = positive_nodes - {node}
                if not positive_nodes:
                    continue
                with torch.no_grad():
                    snapshots = [
                        self.encode_path(cp, all_emb, timestamps_dict=timestamps_dict).detach().clone()
                        for cp in center_paths
                    ]
                self.path_cache.put_paths(node, center_paths, embedding_snapshots=snapshots)
                path_weight = 1.0
            else:
                center_paths = [p for p, _ in weighted_paths]
                path_weights_list = [w for _, w in weighted_paths]
                path_weight = sum(path_weights_list) / len(path_weights_list)
                positive_nodes = PathSampler.get_positive_nodes(center_paths, self.alpha) - {node}
                if not positive_nodes:
                    continue

            target_emb = all_emb[node]
            pos_list = list(positive_nodes)
            pos_embs = all_emb[pos_list]

            n_neg = min(len(pos_list) * 10, 256)
            neg_candidates = np.random.randint(0, total_nodes, size=n_neg * 2)
            neg_list = [n for n in neg_candidates if n != node and n not in positive_nodes][:n_neg]
            if not neg_list:
                continue
            neg_embs = all_emb[neg_list]

            loss = self.contrastive_loss(target_emb, pos_embs, neg_embs)
            losses.append(path_weight * loss)

        if not losses:
            return torch.tensor(0.0, device=user_emb.device)
        return gamma * torch.stack(losses).mean()

    def compute_drift_weighted_inter_loss(
        self,
        target_nodes: np.ndarray,
        user_emb: torch.Tensor,
        item_emb: torch.Tensor,
        timestamps_dict: Optional[Dict] = None,
        gamma: float = 1.0,
    ) -> torch.Tensor:
        if self.path_sampler is None:
            return torch.tensor(0.0, device=user_emb.device)

        all_emb = torch.cat([user_emb, item_emb], dim=0)
        losses = []

        for node in target_nodes:
            weighted_paths = self.path_cache.get_weighted_paths(
                node,
                ttl_batches=self.cache_ttl_batches,
                min_weight=self.cache_min_path_weight,
            )

            if not weighted_paths:
                center_paths, positive_paths_dict, _ = self.path_sampler.sample_paths_for_node(
                    node, self.num_center_paths, self.alpha, self.beta, self.num_positive_paths,
                )
                path_weights = [1.0] * len(center_paths)
            else:
                center_paths = [p for p, _ in weighted_paths]
                path_weights = [w for _, w in weighted_paths]
                positive_paths_dict = {}
                for cp in center_paths:
                    pos_paths = self.path_sampler.target_guided_random_walk(
                        cp, self.beta, self.num_positive_paths,
                    )
                    positive_paths_dict[tuple(cp)] = pos_paths

            if len(center_paths) < 2:
                continue

            center_embs = []
            for cp in center_paths:
                center_embs.append(self.encode_path(cp, all_emb, timestamps_dict=timestamps_dict))
            center_embs_t = torch.stack(center_embs)

            for idx, cp in enumerate(center_paths):
                w = path_weights[idx] if idx < len(path_weights) else 1.0
                if w <= 0:
                    continue
                pos_paths = positive_paths_dict.get(tuple(cp), [])
                if not pos_paths:
                    continue
                pos_embs = torch.stack([
                    self.encode_path(pp, all_emb, timestamps_dict=timestamps_dict)
                    for pp in pos_paths
                ])
                neg_mask = torch.ones(len(center_paths), dtype=torch.bool, device=user_emb.device)
                neg_mask[idx] = False
                neg_embs = center_embs_t[neg_mask]
                if neg_embs.shape[0] == 0:
                    continue
                loss = self.contrastive_loss.inter_path_loss(
                    center_embs[idx], pos_embs, neg_embs,
                )
                losses.append(w * loss)

        if not losses:
            return torch.tensor(0.0, device=user_emb.device)
        return gamma * torch.stack(losses).mean()

    def compute_streaming_loss(
        self,
        users: torch.Tensor,
        pos_items: torch.Tensor,
        neg_items: torch.Tensor,
        adj_matrix: torch.Tensor,
        lambda1: float = 0.1,
        lambda2: float = 0.1,
        lambda3: float = 1e-4,
        timestamps_dict: Optional[Dict] = None,
        affected_indices: Optional[torch.Tensor] = None,
        node_cl_primary_n: Optional[int] = None,
    ) -> Tuple[torch.Tensor, Dict]:
        user_emb, item_emb = self.forward(adj_matrix)


        if neg_items.dim() == 2 and neg_items.shape[1] > 1:
            with torch.no_grad():
                u_e_hn = user_emb[users]
                i_cands = item_emb[neg_items]
                scores_cands = (u_e_hn.unsqueeze(1) * i_cands).sum(-1)
                best_k = scores_cands.argmax(dim=1)
            neg_items = neg_items[
                torch.arange(neg_items.shape[0], device=neg_items.device), best_k
            ]

        if self.item_bias_enabled and self.item_bias is not None:


            u_e = user_emb[users]
            i_pos = item_emb[pos_items]
            i_neg = item_emb[neg_items]
            b_pos = self.item_bias(pos_items).squeeze(-1)
            b_neg = self.item_bias(neg_items).squeeze(-1)
            s_pos = (u_e * i_pos).sum(-1) + b_pos
            s_neg = (u_e * i_neg).sum(-1) + b_neg
            l_bpr = torch.nn.functional.softplus(s_neg - s_pos).mean()
        else:
            l_bpr = bpr_loss(user_emb[users], item_emb[pos_items], item_emb[neg_items])

        gamma = self.drift_detector.gamma

        users_np_full = users.cpu().numpy()
        unique_users = np.unique(users_np_full)
        if self.drift_gate_enabled:


            if gamma >= self.drift_gate_threshold:
                target_k = self.path_contrast_users
            else:
                target_k = self.path_contrast_users_quiet
        else:
            target_k = self.path_contrast_users
        sample_size = min(target_k, len(unique_users))
        if sample_size > 0:
            sampled = np.random.choice(unique_users, size=sample_size, replace=False)
        else:
            sampled = unique_users[:0]

        l_intra = self.compute_drift_weighted_intra_loss(
            sampled, user_emb, item_emb, gamma, timestamps_dict,
        )
        l_inter = self.compute_drift_weighted_inter_loss(
            sampled, user_emb, item_emb, timestamps_dict, gamma,
        )

        all_emb = torch.cat([user_emb, item_emb], dim=0)
        l_smooth = self.smoothness_reg.compute_loss(all_emb, affected_indices)

        l_stability = torch.tensor(0.0, device=user_emb.device)
        if self.momentum_encoder is not None and len(self.replay_buffer) > 0:
            replay_nodes, replay_paths, _ = self.replay_buffer.get_replay_paths(
                n_replay=min(16, len(self.replay_buffer)),
                path_sampler=self.path_sampler,
                num_paths_per_node=3,
                alpha=self.alpha,
                beta=self.beta,
                num_positive_paths=2,
            )
            if replay_paths:
                current_embs = [
                    self.encode_path(rp, all_emb, timestamps_dict=timestamps_dict)
                    for rp in replay_paths
                ]
                current_embs_t = torch.stack(current_embs)

                with torch.no_grad():
                    ref_user_emb, ref_item_emb = self.momentum_encoder.get_ref_embeddings(adj_matrix)
                    ref_all_emb = torch.cat([ref_user_emb, ref_item_emb], dim=0)
                    ref_embs_t = self.momentum_encoder.encode_paths(
                        replay_paths, ref_all_emb, self.temporal_encoder,
                        timestamps_dict, self.n_users, self.encode_path,
                    )

                l_stability = self.momentum_encoder.compute_stability_loss(current_embs_t, ref_embs_t)

        ego_user = self.lightgcn.user_embedding(users)
        ego_pos = self.lightgcn.item_embedding(pos_items)
        ego_neg = self.lightgcn.item_embedding(neg_items)
        l2 = lambda3 * (
            ego_user.norm(2).pow(2) + ego_pos.norm(2).pow(2) + ego_neg.norm(2).pow(2)
        ) / (2.0 * len(users))

        l_node_cl = torch.tensor(0.0, device=user_emb.device)
        if self.node_contrast_lambda > 0:


            n_cl = users.size(0)
            if node_cl_primary_n is not None and node_cl_primary_n > 0:
                n_cl = min(n_cl, int(node_cl_primary_n))
            n_cl = min(n_cl, self.node_contrast_max_nodes)
            if n_cl > 0:
                u_idx = users[:n_cl]
                p_idx = pos_items[:n_cl]
                u_batch = user_emb[u_idx]
                i_batch = item_emb[p_idx]
                eps = self.node_contrast_eps
                tau = self.node_contrast_temp

                def _augment(h: torch.Tensor) -> torch.Tensor:
                    noise = torch.empty_like(h).uniform_(-eps, eps)
                    return h + noise

                def _infonce(v1: torch.Tensor, v2: torch.Tensor) -> torch.Tensor:
                    v1 = F.normalize(v1, dim=-1)
                    v2 = F.normalize(v2, dim=-1)
                    sim = torch.mm(v1, v2.t()) / tau
                    labels = torch.arange(v1.size(0), device=v1.device)
                    return F.cross_entropy(sim, labels)

                u_v1, u_v2 = _augment(u_batch), _augment(u_batch)
                i_v1, i_v2 = _augment(i_batch), _augment(i_batch)
                l_node_cl = 0.5 * (_infonce(u_v1, u_v2) + _infonce(i_v1, i_v2))

        total = (
            l_bpr
            + lambda1 * l_intra
            + lambda2 * l_inter
            + l_smooth
            + self.lambda_stab * l_stability
            + l2
            + self.node_contrast_lambda * l_node_cl
        )

        components = {
            "bpr": l_bpr.item(),
            "intra": l_intra.item(),
            "inter": l_inter.item(),
            "smooth": l_smooth.item(),
            "stability": l_stability.item() if isinstance(l_stability, torch.Tensor) else l_stability,
            "l2": l2.item(),
            "node_cl": l_node_cl.item()
            if isinstance(l_node_cl, torch.Tensor)
            else float(l_node_cl),
            "gamma": gamma,
            "total": total.item(),
        }
        return total, components

    def score_and_refresh_cache(
        self,
        adj_matrix: torch.Tensor,
        timestamps_dict: Optional[Dict] = None,
        max_nodes: int = 200,
    ) -> List[float]:
        cached_nodes = list(self.path_cache._cache.keys())
        if not cached_nodes:
            return []

        if len(cached_nodes) > max_nodes:
            cached_nodes = list(np.random.choice(cached_nodes, size=max_nodes, replace=False))

        with torch.no_grad():
            user_emb, item_emb = self.forward(adj_matrix)
            all_emb = torch.cat([user_emb, item_emb], dim=0)

        all_drift_scores: List[float] = []
        for node in cached_nodes:
            drifts = self.path_cache.score_drift(
                node, self.encode_path, all_emb,
                self.temporal_encoder, timestamps_dict, self.n_users,
            )
            all_drift_scores.extend(drifts)
            self.path_cache.invalidate_high_drift(node)

        if all_drift_scores:
            self.drift_detector.update(torch.tensor(all_drift_scores))

        return all_drift_scores

    def refresh_cache_snapshots(
        self,
        adj_matrix: torch.Tensor,
        timestamps_dict: Optional[Dict] = None,
        max_nodes: int = 200,
    ) -> None:
        cached_nodes = list(self.path_cache._cache.keys())
        if not cached_nodes:
            return

        if len(cached_nodes) > max_nodes:
            cached_nodes = list(np.random.choice(cached_nodes, size=max_nodes, replace=False))

        with torch.no_grad():
            user_emb, item_emb = self.forward(adj_matrix)
            all_emb = torch.cat([user_emb, item_emb], dim=0)

        for node in cached_nodes:
            entries = self.path_cache._cache.get(node, [])
            if not entries:
                continue
            new_snaps = [
                self.encode_path(
                    entry.path, all_emb,
                    timestamps_dict=timestamps_dict,
                ).detach().clone()
                for entry in entries
            ]
            self.path_cache.update_snapshots(node, new_snaps)

    def post_update(
        self,
        adj_matrix: torch.Tensor,
        new_interactions: List[Tuple[int, int]],
        graph_manager: StreamingGraphManager,
        batch_id: int = 0,
        timestamps_dict: Optional[Dict] = None,
    ) -> None:


        self.path_cache.current_batch = batch_id + 1

        if self.momentum_encoder is not None:
            self.momentum_encoder.update(self.lightgcn)

        batch_items = np.array([i for _, i in new_interactions])
        self.exposure_sampler.update_propensity(batch_items)

        self.replay_buffer.update(
            new_interactions,
            graph_manager.neighbors,
            graph_manager.edge_timestamps,
            batch_id=batch_id,
            n_users=self.n_users,
        )


        if self._drift_source == "node":


            with torch.no_grad():
                user_emb, item_emb = self.forward(adj_matrix)
                all_emb = torch.cat([user_emb, item_emb], dim=0)
            if self._prev_all_emb is not None:
                node_drift = self.drift_detector.compute_node_drift(
                    self._prev_all_emb, all_emb
                )
                self.drift_detector.update(node_drift.cpu())
            self._prev_all_emb = all_emb.detach().clone()
        else:


            if self.drift_gate_enabled:
                self.score_and_refresh_cache(adj_matrix, timestamps_dict)

        self.refresh_cache_snapshots(adj_matrix, timestamps_dict)

    def take_embedding_snapshot(self, adj_matrix: torch.Tensor) -> None:
        with torch.no_grad():
            user_emb, item_emb = self.forward(adj_matrix)
            all_emb = torch.cat([user_emb, item_emb], dim=0)
            self.smoothness_reg.take_snapshot(all_emb)
