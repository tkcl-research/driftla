
import numpy as np
import torch
from collections import defaultdict


def recall_at_k(predicted_items, ground_truth, k=10):
    if len(ground_truth) == 0:
        return 0.0
    predicted_set = set(predicted_items[:k])
    return len(predicted_set & ground_truth) / len(ground_truth)


def ndcg_at_k(predicted_items, ground_truth, k=10):
    if len(ground_truth) == 0:
        return 0.0
    dcg = sum(1.0 / np.log2(i + 2) for i, item in enumerate(predicted_items[:k])
              if item in ground_truth)
    idcg = sum(1.0 / np.log2(i + 2) for i in range(min(len(ground_truth), k)))
    return dcg / idcg if idcg > 0 else 0.0


def evaluate_model(model, train_interactions, test_interactions,
                   user_emb, item_emb, k_list=(10, 20), batch_size=256):

    train_dict = defaultdict(set)
    for u, i in train_interactions:
        train_dict[u].add(i)
    test_dict = defaultdict(set)
    for u, i in test_interactions:
        test_dict[u].add(i)

    test_users = sorted(test_dict.keys())
    if not test_users:
        return {f"Recall@{k}": 0.0 for k in k_list} | {f"NDCG@{k}": 0.0 for k in k_list}

    max_k = max(k_list)
    metrics = {f"Recall@{k}": 0.0 for k in k_list}
    metrics.update({f"NDCG@{k}": 0.0 for k in k_list})

    n_items = item_emb.shape[0]

    for start in range(0, len(test_users), batch_size):
        batch_users = test_users[start:start + batch_size]
        u_idx = torch.tensor(batch_users, device=user_emb.device, dtype=torch.long)
        scores = torch.matmul(user_emb[u_idx], item_emb.t())


        for local_idx, uid in enumerate(batch_users):
            seen = train_dict.get(uid)
            if seen:
                seen_t = torch.tensor(list(seen), device=scores.device, dtype=torch.long)
                scores[local_idx, seen_t] = -float("inf")

        _, topk = torch.topk(scores, max_k, dim=1)
        topk = topk.cpu().numpy()

        for local_idx, uid in enumerate(batch_users):
            gt = test_dict[uid]
            pred = topk[local_idx]
            for k in k_list:
                metrics[f"Recall@{k}"] += recall_at_k(pred, gt, k)
                metrics[f"NDCG@{k}"] += ndcg_at_k(pred, gt, k)

    n = len(test_users)
    for key in metrics:
        metrics[key] /= n
    return metrics


def set_seed(seed=42):
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
