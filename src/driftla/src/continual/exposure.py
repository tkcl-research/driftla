
import torch
import numpy as np
from typing import Dict, Optional


class ExposureCalibratedSampler:

    def __init__(
        self,
        n_items: int,
        gamma: float = 0.1,
        eta_p: float = 0.1,
        min_propensity: float = 1e-3,
        mix_uniform_batches: int = 3,
    ):
        self.n_items = n_items
        self.gamma = gamma
        self.eta_p = eta_p
        self.min_propensity = min_propensity
        self.mix_uniform_batches = mix_uniform_batches

        self.propensity = np.ones(n_items, dtype=np.float64) / n_items
        self._batch_count = 0

    def update_propensity(self, batch_items: np.ndarray) -> None:
        freq = np.zeros(self.n_items, dtype=np.float64)
        unique, counts = np.unique(batch_items, return_counts=True)
        valid_mask = unique < self.n_items
        freq[unique[valid_mask]] = counts[valid_mask]
        batch_size = max(1, len(batch_items))
        batch_prop = freq / batch_size

        self.propensity = (
            (1.0 - self.eta_p) * self.propensity + self.eta_p * batch_prop
        )
        self.propensity = np.maximum(self.propensity, self.min_propensity)
        self._batch_count += 1

    def _sampling_probs_ips(self) -> np.ndarray:
        inv_prop = 1.0 / (self.propensity ** self.gamma)
        return inv_prop / inv_prop.sum()

    def _sampling_probs_popularity(self) -> np.ndarray:

        s = self.propensity.sum()
        return self.propensity / s if s > 0 else np.ones(self.n_items) / self.n_items

    def sample_negatives(
        self,
        users: np.ndarray,
        user_pos_items: Dict[int, set],
        num_negatives: int = 1,
    ) -> np.ndarray:
        use_mix = self._batch_count < self.mix_uniform_batches
        if use_mix:
            pop_probs = self._sampling_probs_popularity()
        ips_probs = self._sampling_probs_ips()

        negatives = []
        for u in users:
            pos = user_pos_items.get(int(u), set())
            negs = []
            attempts = 0
            while len(negs) < num_negatives and attempts < num_negatives * 20:
                if use_mix and np.random.random() < 0.5:
                    candidate = np.random.randint(0, self.n_items)
                elif use_mix:
                    candidate = np.random.choice(self.n_items, p=pop_probs)
                else:
                    candidate = np.random.choice(self.n_items, p=ips_probs)
                if candidate not in pos:
                    negs.append(candidate)
                attempts += 1
            while len(negs) < num_negatives:
                candidate = np.random.randint(0, self.n_items)
                if candidate not in pos:
                    negs.append(candidate)
            negatives.extend(negs)

        return np.array(negatives, dtype=np.int64)

    def get_weights(self, item_ids: np.ndarray) -> torch.Tensor:
        props = self.propensity[item_ids]
        weights = 1.0 / (props ** self.gamma)
        weights = weights / weights.mean()
        return torch.from_numpy(weights).float()

    def stats(self) -> Dict:
        return {
            "mean_propensity": float(self.propensity.mean()),
            "max_propensity": float(self.propensity.max()),
            "min_propensity": float(self.propensity.min()),
            "gini_propensity": float(self._gini(self.propensity)),
            "batches_seen": int(self._batch_count),
        }

    @staticmethod
    def _gini(values: np.ndarray) -> float:
        sorted_vals = np.sort(values)
        n = len(sorted_vals)
        index = np.arange(1, n + 1)
        return float((2.0 * np.sum(index * sorted_vals) / (n * np.sum(sorted_vals))) - (n + 1) / n)
