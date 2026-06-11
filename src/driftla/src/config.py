
from __future__ import annotations

import dataclasses
from dataclasses import dataclass, asdict
from typing import Any, Dict, Literal, Optional

PaperStreamingPreset = Literal[
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
]


@dataclass
class DriftLAConfig:


    embed_dim: int = 64
    n_layers: int = 3

    lightgcn_ego_skip_alpha: float = 0.0
    replay_size: int = 500
    drift_threshold: float = 0.08
    lambda_s: float = 0.002
    lambda_stab: float = 0.0002


    lr: float = 0.001
    batch_size: int = 2048
    warmup_epochs: int = 3


    history_replay: int = 20000
    streaming_passes: int = 3
    distill_lambda: float = 0.05
    distill_ema: float = 0.99


    bpr_pop_weight: bool = True
    bpr_pop_scalar: float = 0.01
    bpr_pop_clip_low: float = 0.5
    bpr_pop_clip_high: float = 2.0


    lambda_path_intra: float = 0.1
    lambda_path_inter: float = 0.1
    lambda_l2: float = 1e-4


    path_contrast_users: int = 256
    num_center_paths: int = 5


    propensity_gamma: float = 0.1
    propensity_eta: float = 0.1
    min_propensity: float = 1e-3
    mix_uniform_batches: int = 3


    cache_ttl_batches: int = 2
    cache_min_path_weight: float = 0.5


    use_time_decay: bool = False
    time_decay_lambda: float = 1.0

    time_decay_floor: float = 0.0


    item_bias: bool = False


    recency_replay: bool = False
    recency_replay_lambda: float = 1.0

    recency_replay_floor: float = 0.0


    drift_gate: bool = False
    drift_gate_threshold: float = 1.05
    path_contrast_users_quiet: int = 32


    propagated_distill: bool = False
    distill_ema_drift: float = 0.9


    distill_gamma_coupling: float = 0.0


    use_stream_adapter: bool = False
    stream_adapter_rank: int = 8

    stream_adapter_gamma_floor: float = 1.0
    stream_adapter_ramp: float = 0.18
    stream_adapter_min_gate: float = 0.07

    adapter_gate_mode: str = "drift"
    adapter_fixed_gate: float = 0.5


    hard_neg_k: int = 1


    node_contrast_lambda: float = 0.0
    node_contrast_eps: float = 0.2
    node_contrast_temp: float = 0.2

    node_contrast_max_nodes: int = 2048


    drift_source: str = "path"


    auto_enabled: bool = False
    auto_gamma_drift_threshold: float = 1.05
    auto_path_scale_drift: float = 1.15
    auto_smooth_scale_drift: float = 0.35
    auto_stab_scale_drift: float = 0.35
    auto_distill_scale_drift: float = 0.35
    auto_time_decay_lambda_scale_drift: float = 1.25
    auto_recency_replay_lambda_scale_drift: float = 1.25

    def to_json_dict(self) -> Dict[str, Any]:
        return asdict(self)

    @staticmethod
    def final_default() -> "DriftLAConfig":
        return DriftLAConfig()

    @staticmethod
    def legacy_b2_replay2048_sweep() -> "DriftLAConfig":
        return dataclasses.replace(
            DriftLAConfig(),
            warmup_epochs=3,
            streaming_passes=1,
            history_replay=2048,
        )

    @staticmethod
    def paper_streaming_100_passes_per_batch() -> "DriftLAConfig":
        return dataclasses.replace(DriftLAConfig.final_default(), streaming_passes=100)

    @staticmethod
    def paper_streaming_100_warmup() -> "DriftLAConfig":
        return dataclasses.replace(DriftLAConfig.final_default(), warmup_epochs=100)

    @staticmethod
    def paper_streaming_100_both() -> "DriftLAConfig":
        return dataclasses.replace(
            DriftLAConfig.final_default(), warmup_epochs=100, streaming_passes=100,
        )

    @staticmethod
    def v3_default() -> "DriftLAConfig":
        return dataclasses.replace(
            DriftLAConfig.final_default(),

            bpr_pop_weight=False,

            use_time_decay=True,
            time_decay_lambda=1.0,
            time_decay_floor=0.15,
            item_bias=True,
            recency_replay=True,
            recency_replay_lambda=1.0,
            recency_replay_floor=0.10,
            drift_gate=True,
            drift_gate_threshold=1.05,
            path_contrast_users_quiet=32,

            num_center_paths=7,
            propagated_distill=True,
            distill_ema=0.99,
            distill_ema_drift=0.9,

            distill_lambda=0.05,

            use_stream_adapter=True,
            stream_adapter_rank=16,
        )

    @staticmethod
    def v3_auto_default() -> "DriftLAConfig":
        return dataclasses.replace(
            DriftLAConfig.v3_default(),
            auto_enabled=True,

            auto_gamma_drift_threshold=1.05,
        )

    @staticmethod
    def for_dataset(name: str, base: Optional["DriftLAConfig"] = None) -> "DriftLAConfig":
        cfg = base if base is not None else DriftLAConfig.final_default()
        name = name.lower().replace("gowalla", "gowala")
        if name in ("amazon23", "amz23_digital_music", "amz23_all_beauty"):
            name = "ml-1m"
        if name in ("gowala_dense", "yelp_dense"):


            if name == "gowala_dense":
                name = "gowala"
            else:
                name = "yelp"
        if name in ("ml-1m", "ml1m"):
            return dataclasses.replace(
                cfg,
                history_replay=20000,


                distill_lambda=0.055,
                time_decay_lambda=1.1,
                time_decay_floor=0.00,
                recency_replay_lambda=1.1,
                recency_replay_floor=0.00,
                path_contrast_users=288,
                path_contrast_users_quiet=40,
            )
        if name == "ciao":


            return dataclasses.replace(
                cfg,
                history_replay=50000,
                distill_lambda=0.055,
                time_decay_lambda=1.1,
                time_decay_floor=0.00,
                recency_replay_lambda=1.1,
                recency_replay_floor=0.00,
                path_contrast_users=288,
                path_contrast_users_quiet=40,
            )
        if name in ("gowala", "gowalla"):


            return dataclasses.replace(
                cfg,
                history_replay=50000,

                distill_lambda=0.01,
                lambda_path_intra=0.02,
                lambda_path_inter=0.02,

                lambda_s=0.0002,
                lambda_stab=0.00003,
                time_decay_lambda=0.35,
                time_decay_floor=0.25,
                recency_replay_lambda=0.35,
                recency_replay_floor=0.20,
                drift_gate_threshold=1.01,
                path_contrast_users=320,
                path_contrast_users_quiet=48,
                mix_uniform_batches=10,
                use_stream_adapter=True,
                stream_adapter_min_gate=0.02,
            )
        if name == "yelp":
            return dataclasses.replace(
                cfg,
                history_replay=50000,
                distill_lambda=0.01,
                lambda_path_intra=0.02,
                lambda_path_inter=0.02,
                lambda_s=0.0002,
                lambda_stab=0.00003,
                time_decay_lambda=0.35,
                time_decay_floor=0.25,
                recency_replay_lambda=0.35,
                recency_replay_floor=0.20,
                drift_gate_threshold=1.01,
                path_contrast_users=320,
                path_contrast_users_quiet=48,
                mix_uniform_batches=10,
                use_stream_adapter=True,
                stream_adapter_min_gate=0.02,
            )
        return cfg

    @staticmethod
    def for_dataset_v3_bundle_acc(
        name: str, base: Optional["DriftLAConfig"] = None
    ) -> "DriftLAConfig":
        cfg = base if base is not None else DriftLAConfig.v3_default()
        name = name.lower()
        if name in ("amazon23", "amz23_digital_music", "amz23_all_beauty"):
            name = "ml-1m"

        if name in ("ml-1m", "ml1m"):

            return DriftLAConfig.for_dataset(name, base=cfg)

        if name == "ciao":


            return dataclasses.replace(
                cfg,
                history_replay=50000,

                distill_lambda=0.035,

                lambda_s=0.0002,
                lambda_stab=0.00005,

                lambda_path_intra=0.06,
                lambda_path_inter=0.06,

                path_contrast_users=240,
                path_contrast_users_quiet=32,
            )

        if name in ("gowala", "gowalla"):


            return dataclasses.replace(
                cfg,
                history_replay=50000,

                distill_lambda=0.01,

                lambda_s=0.0002,
                lambda_stab=0.00003,

                lambda_path_intra=0.02,
                lambda_path_inter=0.02,

                time_decay_lambda=0.35,
                time_decay_floor=0.25,
                recency_replay_lambda=0.35,
                recency_replay_floor=0.20,
                drift_gate_threshold=1.01,
                path_contrast_users=320,
                path_contrast_users_quiet=48,
                mix_uniform_batches=10,

                use_stream_adapter=True,
                stream_adapter_min_gate=0.02,
            )

        if name == "yelp":

            return dataclasses.replace(
                cfg,
                history_replay=50000,
                distill_lambda=0.01,
                lambda_s=0.0002,
                lambda_stab=0.00003,
                lambda_path_intra=0.02,
                lambda_path_inter=0.02,
                time_decay_lambda=0.35,
                time_decay_floor=0.25,
                recency_replay_lambda=0.35,
                recency_replay_floor=0.20,
                drift_gate_threshold=1.01,
                path_contrast_users=320,
                path_contrast_users_quiet=48,
                mix_uniform_batches=10,
                use_stream_adapter=True,
                stream_adapter_min_gate=0.02,
            )

        return cfg

    @staticmethod
    def v3_champion_default() -> "DriftLAConfig":
        return DriftLAConfig.v3_auto_default()

    @staticmethod
    def for_dataset_v3_champion(
        name: str, base: Optional["DriftLAConfig"] = None
    ) -> "DriftLAConfig":
        cfg = base if base is not None else DriftLAConfig.v3_champion_default()
        name = name.lower().replace("gowalla", "gowala")
        if name in ("amazon23", "amz23_digital_music", "amz23_all_beauty"):
            name = "ml-1m"

        if name in ("ml-1m", "ml1m"):

            return DriftLAConfig.for_dataset(name, base=cfg)

        if name == "ciao":


            return dataclasses.replace(
                cfg,
                history_replay=50000,
                lambda_s=0.0,
                lambda_stab=0.0,
                distill_lambda=0.04,
                lambda_path_intra=0.08,
                lambda_path_inter=0.08,
                path_contrast_users=240,
                path_contrast_users_quiet=32,
                time_decay_lambda=1.1,
                time_decay_floor=0.0,
                recency_replay_lambda=1.1,
                recency_replay_floor=0.0,
            )

        if name in ("gowala_dense", "gowalla_dense", "yelp_dense"):
            canonical = "gowala" if "gowala" in name else "yelp"
            v3_base = DriftLAConfig.for_dataset(canonical, base=DriftLAConfig.v3_default())
            return dataclasses.replace(v3_base, lambda_s=0.0)

        if name in ("gowala", "gowalla"):
            v3_base = DriftLAConfig.for_dataset(name, base=DriftLAConfig.v3_default())
            return dataclasses.replace(v3_base, lambda_s=0.0)

        if name == "yelp":
            v3_base = DriftLAConfig.for_dataset("yelp", base=DriftLAConfig.v3_default())
            return dataclasses.replace(v3_base, lambda_s=0.0)

        return cfg


    @staticmethod
    def v4_improved_default() -> "DriftLAConfig":
        return dataclasses.replace(
            DriftLAConfig.v3_champion_default(),
            hard_neg_k=8,
        )

    @staticmethod
    def for_dataset_v4_improved(
        name: str, base: Optional["DriftLAConfig"] = None
    ) -> "DriftLAConfig":
        cfg = base if base is not None else DriftLAConfig.v4_improved_default()


        tuned = DriftLAConfig.for_dataset_v3_champion(name, base=cfg)
        return dataclasses.replace(tuned, hard_neg_k=cfg.hard_neg_k)

    @staticmethod
    def v3_sparse_default() -> "DriftLAConfig":
        return dataclasses.replace(
            DriftLAConfig.v3_champion_default(),
            hard_neg_k=8,
            lambda_s=0.0,
            lambda_path_intra=0.0,
            lambda_path_inter=0.0,
            distill_lambda=0.0,
            time_decay_floor=0.0,
            recency_replay_floor=0.0,
            use_stream_adapter=False,
        )

    @staticmethod
    def for_dataset_v3_sparse(
        name: str, base: Optional["DriftLAConfig"] = None
    ) -> "DriftLAConfig":
        cfg = base if base is not None else DriftLAConfig.v3_sparse_default()
        tuned = DriftLAConfig.for_dataset_v3_champion(name, base=cfg)
        return dataclasses.replace(
            tuned,
            hard_neg_k=cfg.hard_neg_k,
            lambda_s=0.0,
            lambda_path_intra=0.0,
            lambda_path_inter=0.0,
            distill_lambda=0.0,
            time_decay_floor=0.0,
            recency_replay_floor=0.0,
            use_stream_adapter=False,
        )

    @staticmethod
    def v3_sparse_uniform_default() -> "DriftLAConfig":
        return dataclasses.replace(
            DriftLAConfig.v3_sparse_default(),
            node_contrast_lambda=0.2,
            node_contrast_eps=0.2,
            node_contrast_temp=0.2,
        )

    @staticmethod
    def for_dataset_v3_sparse_uniform(
        name: str, base: Optional["DriftLAConfig"] = None
    ) -> "DriftLAConfig":
        cfg = base if base is not None else DriftLAConfig.v3_sparse_uniform_default()
        tuned = DriftLAConfig.for_dataset_v3_sparse(name, base=cfg)
        base_u = cfg if base is not None else DriftLAConfig.v3_sparse_uniform_default()
        return dataclasses.replace(
            tuned,
            node_contrast_lambda=base_u.node_contrast_lambda,
            node_contrast_eps=base_u.node_contrast_eps,
            node_contrast_temp=base_u.node_contrast_temp,
        )


def preset_config(name: PaperStreamingPreset) -> DriftLAConfig:
    if name == "final":
        return DriftLAConfig.final_default()
    if name == "bundle":

        return DriftLAConfig.final_default()
    if name == "legacy_b2":
        return DriftLAConfig.legacy_b2_replay2048_sweep()
    if name == "paper_100_batch":
        return DriftLAConfig.paper_streaming_100_passes_per_batch()
    if name == "paper_100_warmup":
        return DriftLAConfig.paper_streaming_100_warmup()
    if name == "paper_100_both":
        return DriftLAConfig.paper_streaming_100_both()
    if name == "v3":
        return DriftLAConfig.v3_default()
    if name == "v3_bundle_acc":
        return DriftLAConfig.v3_default()
    if name == "v3_auto":
        return DriftLAConfig.v3_auto_default()
    if name == "v3_champion":
        return DriftLAConfig.v3_champion_default()
    if name == "v4_improved":
        return DriftLAConfig.v4_improved_default()
    if name == "v3_sparse":
        return DriftLAConfig.v3_sparse_default()
    if name == "v3_sparse_uniform":
        return DriftLAConfig.v3_sparse_uniform_default()
    raise ValueError(f"Unknown preset: {name}")


def config_from_dict(d: Dict[str, Any]) -> DriftLAConfig:
    defaults = DriftLAConfig()
    kw = {}
    for f in dataclasses.fields(DriftLAConfig):
        if f.name in d:
            kw[f.name] = d[f.name]
        else:
            kw[f.name] = getattr(defaults, f.name)
    return DriftLAConfig(**kw)
