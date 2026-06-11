
from __future__ import annotations

import dataclasses

from .config import DriftLAConfig


ARCH_VARIANT_CHOICES = (
    "baseline",
    "asymmetric_m7",
    "adapter_r16",
    "ego_skip_backbone",
    "wide_path_heads",
    "distill_gamma_couple",
)


def apply_arch_variant(cfg: DriftLAConfig, dataset: str, variant: str) -> DriftLAConfig:
    v = (variant or "").strip().lower()
    if not v or v in ("baseline", "none"):
        return cfg
    ds_raw = dataset.lower().replace("gowalla", "gowala")
    is_dense = "_dense" in ds_raw
    ds = ds_raw.replace("_dense", "")

    if v == "asymmetric_m7":

        if ds in ("gowala", "yelp") and not is_dense:
            return dataclasses.replace(cfg, use_stream_adapter=False)
        return cfg

    if v == "adapter_r16":
        return dataclasses.replace(cfg, stream_adapter_rank=16)

    if v == "ego_skip_backbone":
        return dataclasses.replace(cfg, lightgcn_ego_skip_alpha=0.2)

    if v == "wide_path_heads":
        return dataclasses.replace(cfg, num_center_paths=7)

    if v == "distill_gamma_couple":
        return dataclasses.replace(cfg, distill_gamma_coupling=0.22)

    raise ValueError(
        f"Unknown arch variant {variant!r}. Choose one of: {', '.join(ARCH_VARIANT_CHOICES)}"
    )
