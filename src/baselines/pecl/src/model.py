
import torch
import torch.nn as nn
import numpy as np

from .lightgcn import LightGCN
from .path_sampling import PathSampler
from .temporal_encoding import TemporalEncoder, encode_path_with_temporal
from .contrastive import ContrastiveLoss, bpr_loss


class PECL(nn.Module):

    def __init__(self, n_users, n_items, embed_dim=64, n_layers=3,
                 alpha=2, beta=4, tau=0.05, num_center_paths=10,
                 num_positive_paths=5, path_length=5):
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

        self.lightgcn = LightGCN(n_users, n_items, embed_dim, n_layers)
        self.temporal_encoder = TemporalEncoder(embed_dim)
        self.contrastive_loss = ContrastiveLoss(tau)

        self.path_sampler = None


    def set_path_sampler(self, interactions, timestamps=None):
        self.path_sampler = PathSampler(
            self.n_users, self.n_items, interactions, timestamps,
            path_length=self.path_length,
        )

    def forward(self, adj_matrix):
        return self.lightgcn(adj_matrix)


    def compute_intra_path_loss(self, target_nodes, user_emb, item_emb):
        if self.path_sampler is None:
            return torch.tensor(0.0, device=user_emb.device)

        all_emb = torch.cat([user_emb, item_emb], dim=0)
        total_nodes = self.n_users + self.n_items
        losses = []

        for node in target_nodes:
            center_paths, _, positive_nodes = self.path_sampler.sample_paths_for_node(
                node, self.num_center_paths, self.alpha, self.beta, self.num_positive_paths,
            )

            positive_nodes = positive_nodes - {node}
            if len(positive_nodes) == 0:
                continue

            target_emb = all_emb[node]
            pos_list = list(positive_nodes)
            pos_embs = all_emb[pos_list]


            n_neg = min(len(pos_list) * 10, 256)
            neg_candidates = np.random.randint(0, total_nodes, size=n_neg * 2)
            neg_list = [n for n in neg_candidates
                        if n != node and n not in positive_nodes][:n_neg]
            if len(neg_list) == 0:
                continue
            neg_embs = all_emb[neg_list]

            loss = self.contrastive_loss(target_emb, pos_embs, neg_embs)
            losses.append(loss)

        if not losses:
            return torch.tensor(0.0, device=user_emb.device)
        return torch.stack(losses).mean()


    def compute_inter_path_loss(self, target_nodes, user_emb, item_emb,
                                timestamps_dict=None):
        if self.path_sampler is None:
            return torch.tensor(0.0, device=user_emb.device)

        all_emb = torch.cat([user_emb, item_emb], dim=0)
        losses = []

        for node in target_nodes:
            center_paths, positive_paths_dict, _ = self.path_sampler.sample_paths_for_node(
                node, self.num_center_paths, self.alpha, self.beta, self.num_positive_paths,
            )
            if len(center_paths) < 2:
                continue


            center_embs = []
            for cp in center_paths:
                center_embs.append(self._encode_path(cp, all_emb, timestamps_dict))
            center_embs_t = torch.stack(center_embs)

            for idx, cp in enumerate(center_paths):
                pos_paths = positive_paths_dict.get(tuple(cp), [])
                if not pos_paths:
                    continue
                pos_embs = torch.stack([
                    self._encode_path(pp, all_emb, timestamps_dict)
                    for pp in pos_paths
                ])

                neg_mask = torch.ones(len(center_paths), dtype=torch.bool,
                                      device=user_emb.device)
                neg_mask[idx] = False
                neg_embs = center_embs_t[neg_mask]
                if neg_embs.shape[0] == 0:
                    continue
                loss = self.contrastive_loss.inter_path_loss(
                    center_embs[idx], pos_embs, neg_embs,
                )
                losses.append(loss)

        if not losses:
            return torch.tensor(0.0, device=user_emb.device)
        return torch.stack(losses).mean()


    def _encode_path(self, path, all_emb, timestamps_dict=None):
        node_embs = all_emb[path]


        edge_ts = []
        for i in range(len(path) - 1):
            a, b = path[i], path[i + 1]

            if a < self.n_users:
                key = (a, b - self.n_users)
            else:
                key = (b, a - self.n_users)
            ts = timestamps_dict.get(key, 0.0) if timestamps_dict else 0.0
            edge_ts.append(ts)

        edge_ts = torch.tensor(edge_ts, device=all_emb.device, dtype=torch.float32)
        return encode_path_with_temporal(node_embs, edge_ts, self.temporal_encoder)


    def compute_total_loss(self, users, pos_items, neg_items, adj_matrix,
                           lambda1=0.1, lambda2=0.1, lambda3=1e-4,
                           timestamps_dict=None):
        user_emb, item_emb = self.forward(adj_matrix)

        bpr = bpr_loss(user_emb[users], item_emb[pos_items], item_emb[neg_items])

        sample_size = min(32, len(users))
        sampled = np.random.choice(users.cpu().numpy(), size=sample_size, replace=False)

        intra = self.compute_intra_path_loss(sampled, user_emb, item_emb)
        inter = self.compute_inter_path_loss(sampled, user_emb, item_emb, timestamps_dict)

        ego_user = self.lightgcn.user_embedding(users)
        ego_pos = self.lightgcn.item_embedding(pos_items)
        ego_neg = self.lightgcn.item_embedding(neg_items)
        l2 = lambda3 * (ego_user.norm(2).pow(2) +
                        ego_pos.norm(2).pow(2) +
                        ego_neg.norm(2).pow(2)) / (2.0 * len(users))

        total = bpr + lambda1 * intra + lambda2 * inter + l2

        return total, {
            "bpr": bpr.item(),
            "intra": intra.item(),
            "inter": inter.item(),
            "l2": l2.item(),
        }
