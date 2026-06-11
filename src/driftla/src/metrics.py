
from __future__ import annotations

from collections import defaultdict
from typing import Dict, List, Tuple

import numpy as np
import torch


def recall_at_k(predicted_items: np.ndarray, ground_truth: set, k: int = 10) -> float:
    if len(ground_truth) == 0:
        return 0.0
    predicted_set = set(predicted_items[:k])
    return len(predicted_set & ground_truth) / len(ground_truth)


def ndcg_at_k(predicted_items: np.ndarray, ground_truth: set, k: int = 10) -> float:
    if len(ground_truth) == 0:
        return 0.0
    dcg = sum(
        1.0 / np.log2(i + 2)
        for i, item in enumerate(predicted_items[:k])
        if item in ground_truth
    )
    idcg = sum(1.0 / np.log2(i + 2) for i in range(min(len(ground_truth), k)))
    return dcg / idcg if idcg > 0 else 0.0


def evaluate_on_batch(
    user_emb: torch.Tensor,
    item_emb: torch.Tensor,
    test_interactions: List[Tuple[int, int]],
    train_interactions: List[Tuple[int, int]],
    k_list: Tuple[int, ...] = (10, 20),
    batch_size: int = 256,
) -> Dict[str, float]:
    if not test_interactions:
        return {}

    train_dict: Dict[int, set] = defaultdict(set)
    for u, i in train_interactions:
        train_dict[u].add(i)
    test_dict: Dict[int, set] = defaultdict(set)
    for u, i in test_interactions:
        test_dict[u].add(i)

    test_users = sorted(test_dict.keys())
    if not test_users:
        return {f"Recall@{k}": 0.0 for k in k_list} | {f"NDCG@{k}": 0.0 for k in k_list}

    max_k = max(k_list)
    metrics = {f"Recall@{k}": 0.0 for k in k_list}
    metrics.update({f"NDCG@{k}": 0.0 for k in k_list})

    for start in range(0, len(test_users), batch_size):
        batch_users = test_users[start : start + batch_size]
        u_idx = torch.tensor(batch_users, device=user_emb.device, dtype=torch.long)
        scores = torch.matmul(user_emb[u_idx], item_emb.t())

        for local_idx, uid in enumerate(batch_users):
            seen = train_dict.get(uid)
            if seen:
                seen_t = torch.tensor(list(seen), device=scores.device, dtype=torch.long)
                scores[local_idx, seen_t] = -float("inf")

        _, topk = torch.topk(scores, max_k, dim=1)
        topk_np = topk.cpu().numpy()

        for local_idx, uid in enumerate(batch_users):
            gt = test_dict[uid]
            pred = topk_np[local_idx]
            for k in k_list:
                metrics[f"Recall@{k}"] += recall_at_k(pred, gt, k)
                metrics[f"NDCG@{k}"] += ndcg_at_k(pred, gt, k)

    n = len(test_users)
    for key in metrics:
        metrics[key] /= n
    return metrics
