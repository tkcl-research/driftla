
import torch
import numpy as np
from typing import Dict, List, Optional, Tuple


class DriftDetector:

    def __init__(
        self,
        drift_threshold: float = 0.3,
        alpha_drift: float = 0.5,
    ):
        self.drift_threshold = drift_threshold
        self.alpha_drift = alpha_drift

        self._prev_mean_drift: float = 0.0
        self._current_mean_drift: float = 0.0
        self._gamma: float = 1.0
        self._history: List[Dict] = []

    @staticmethod
    def cosine_drift(emb_old: torch.Tensor, emb_new: torch.Tensor) -> torch.Tensor:
        if emb_old.dim() == 1:
            emb_old = emb_old.unsqueeze(0)
            emb_new = emb_new.unsqueeze(0)
        cos = torch.nn.functional.cosine_similarity(emb_old, emb_new, dim=-1)
        return 1.0 - cos

    def compute_node_drift(
        self,
        emb_prev: torch.Tensor,
        emb_current: torch.Tensor,
        node_indices: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        if node_indices is not None:
            old = emb_prev[node_indices]
            new = emb_current[node_indices]
        else:
            old = emb_prev
            new = emb_current
        return self.cosine_drift(old, new)

    def compute_path_drift(
        self, path_emb_old: torch.Tensor, path_emb_new: torch.Tensor
    ) -> torch.Tensor:
        return self.cosine_drift(path_emb_old, path_emb_new)

    def update(self, drift_scores: torch.Tensor) -> float:
        self._prev_mean_drift = self._current_mean_drift
        self._current_mean_drift = float(drift_scores.mean()) if drift_scores.numel() > 0 else 0.0
        delta = self._current_mean_drift - self._prev_mean_drift


        self._gamma = max(1.0, 1.0 + self.alpha_drift * max(delta, 0.0))

        self._history.append({
            "mean_drift": self._current_mean_drift,
            "delta_drift": delta,
            "gamma": self._gamma,
            "max_drift": float(drift_scores.max()) if drift_scores.numel() > 0 else 0.0,
            "pct_above_threshold": float(
                (drift_scores > self.drift_threshold).float().mean()
            ) if drift_scores.numel() > 0 else 0.0,
        })
        return self._gamma

    @property
    def gamma(self) -> float:
        return self._gamma

    @property
    def mean_drift(self) -> float:
        return self._current_mean_drift

    def path_weights(self, drift_scores: torch.Tensor) -> torch.Tensor:
        weights = torch.clamp(1.0 - drift_scores / self.drift_threshold, min=0.0)
        return weights

    def get_history(self) -> List[Dict]:
        return self._history
