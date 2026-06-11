
import torch
from typing import Optional, Set


class TemporalSmoothnessRegularizer:

    def __init__(self, lambda_s: float = 0.01):
        self.lambda_s = lambda_s
        self._snapshot: Optional[torch.Tensor] = None

    def take_snapshot(self, embeddings: torch.Tensor) -> None:
        self._snapshot = embeddings.detach().clone()

    def compute_loss(
        self,
        current_embeddings: torch.Tensor,
        affected_indices: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        if self._snapshot is None:
            return torch.tensor(0.0, device=current_embeddings.device)

        snapshot = self._snapshot.to(current_embeddings.device)

        if affected_indices is not None:
            diff = current_embeddings[affected_indices] - snapshot[affected_indices]
        else:
            diff = current_embeddings - snapshot

        frobenius_sq = (diff ** 2).sum()
        return self.lambda_s * frobenius_sq

    @property
    def has_snapshot(self) -> bool:
        return self._snapshot is not None

    def clear(self) -> None:
        self._snapshot = None
