
from __future__ import annotations

from dataclasses import dataclass
from typing import Dict


@dataclass(frozen=True)
class AutoControllerConfig:
    enabled: bool = False

    gamma_drift_threshold: float = 1.05


    path_scale_quiet: float = 1.0
    path_scale_drift: float = 1.15


    smooth_scale_quiet: float = 1.0
    smooth_scale_drift: float = 0.35
    stab_scale_quiet: float = 1.0
    stab_scale_drift: float = 0.35


    distill_scale_quiet: float = 1.0
    distill_scale_drift: float = 0.35


    time_decay_lambda_scale_drift: float = 1.25
    recency_replay_lambda_scale_drift: float = 1.25

    def regime(self, gamma: float) -> str:
        return "drift" if float(gamma) >= float(self.gamma_drift_threshold) else "quiet"


class AutoController:
    def __init__(self, cfg: AutoControllerConfig):
        self.cfg = cfg

    def compute_scales(self, gamma: float) -> Dict[str, float]:
        if not self.cfg.enabled:
            return {
                "path": 1.0,
                "smooth": 1.0,
                "stab": 1.0,
                "distill": 1.0,
                "time_decay_lambda": 1.0,
                "recency_replay_lambda": 1.0,
            }

        reg = self.cfg.regime(gamma)
        if reg == "drift":
            return {
                "path": float(self.cfg.path_scale_drift),
                "smooth": float(self.cfg.smooth_scale_drift),
                "stab": float(self.cfg.stab_scale_drift),
                "distill": float(self.cfg.distill_scale_drift),
                "time_decay_lambda": float(self.cfg.time_decay_lambda_scale_drift),
                "recency_replay_lambda": float(self.cfg.recency_replay_lambda_scale_drift),
            }
        return {
            "path": float(self.cfg.path_scale_quiet),
            "smooth": float(self.cfg.smooth_scale_quiet),
            "stab": float(self.cfg.stab_scale_quiet),
            "distill": float(self.cfg.distill_scale_quiet),
            "time_decay_lambda": 1.0,
            "recency_replay_lambda": 1.0,
        }


class DensityDriftRouter:


    DENSITY_HIGH   = 0.03
    DENSITY_SPARSE = 0.0005
    DENSITY_MED_HI = 0.01
    DENSITY_MED_LO = 0.001


    K_HI_TABLE: Dict[str, int] = {
        "high":        128,
        "medium_high": 240,
        "medium_low":  160,
        "sparse":       64,
    }

    def __init__(self, n_edges: int, n_users: int, n_items: int) -> None:
        self.n_edges  = int(n_edges)
        self.n_users  = int(n_users)
        self.n_items  = int(n_items)
        self.density  = self.n_edges / max(1, self.n_users * self.n_items)


    def density_bucket(self) -> str:
        d = self.density
        if d >= self.DENSITY_HIGH:
            return "high"
        if d >= self.DENSITY_MED_HI:
            return "medium_high"
        if d > self.DENSITY_SPARSE:
            return "medium_low"
        return "sparse"


    def route(self, mean_val_drift: float = 0.0) -> Dict[str, object]:
        d      = self.density
        bucket = self.density_bucket()


        if bucket == "medium_high":
            lambda_s = 0.002
            path_contrast_enabled = True
            path_contrast_users = self.K_HI_TABLE[bucket]
        elif bucket == "medium_low":


            lambda_s = 0.0
            path_contrast_enabled = True
            path_contrast_users = 240
        else:
            lambda_s = 0.0
            path_contrast_enabled = False
            path_contrast_users = self.K_HI_TABLE.get(bucket, 64)

        return {

            "lambda_s":              lambda_s,
            "path_contrast_enabled": path_contrast_enabled,
            "path_contrast_users":   path_contrast_users,

            "density":               d,
            "density_bucket":        bucket,
            "mean_val_drift":        float(mean_val_drift),
        }
