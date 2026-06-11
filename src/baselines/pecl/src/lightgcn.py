
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from scipy.sparse import coo_matrix


class LightGCN(nn.Module):

    def __init__(self, n_users, n_items, embed_dim=64, n_layers=3, dropout=False, keep_prob=0.6):
        super(LightGCN, self).__init__()
        self.n_users = n_users
        self.n_items = n_items
        self.embed_dim = embed_dim
        self.n_layers = n_layers
        self.dropout = dropout
        self.keep_prob = keep_prob


        self.user_embedding = nn.Embedding(n_users, embed_dim)
        self.item_embedding = nn.Embedding(n_items, embed_dim)


        nn.init.normal_(self.user_embedding.weight, std=0.1)
        nn.init.normal_(self.item_embedding.weight, std=0.1)

    def __dropout_x(self, x, keep_prob):
        size = x.size()

        index = x.indices().t()
        values = x.values()


        random_index = torch.rand(len(values)) + keep_prob
        random_index = random_index.int().bool()


        index = index[random_index]

        values = values[random_index] / keep_prob


        g = torch.sparse.FloatTensor(index.t(), values, size)
        return g

    def __dropout(self, graph, keep_prob):
        if isinstance(graph, list):

            return [self.__dropout_x(g, keep_prob) for g in graph]
        else:
            return self.__dropout_x(graph, keep_prob)

    def computer(self, adj_matrix):


        users_emb = self.user_embedding.weight
        items_emb = self.item_embedding.weight


        all_emb = torch.cat([users_emb, items_emb], dim=0)


        embs = [all_emb]


        if self.dropout:
            if self.training:

                g_dropped = self.__dropout(adj_matrix, self.keep_prob)
            else:

                g_dropped = adj_matrix
        else:

            g_dropped = adj_matrix


        for layer in range(self.n_layers):


            all_emb = torch.sparse.mm(g_dropped, all_emb)


            embs.append(all_emb)


        embs = torch.stack(embs, dim=1)


        light_out = torch.mean(embs, dim=1)


        users, items = torch.split(light_out, [self.n_users, self.n_items])

        return users, items

    def forward(self, adj_matrix):
        return self.computer(adj_matrix)

    def get_embeddings(self, adj_matrix):
        return self.forward(adj_matrix)

    def getUsersRating(self, users, adj_matrix):

        all_users, all_items = self.computer(adj_matrix)


        users_emb = all_users[users.long()]
        items_emb = all_items


        rating = torch.matmul(users_emb, items_emb.t())


        rating = torch.sigmoid(rating)

        return rating

    def getEmbedding(self, users, pos_items, neg_items, adj_matrix):

        all_users, all_items = self.computer(adj_matrix)


        users_emb = all_users[users]
        pos_emb = all_items[pos_items]
        neg_emb = all_items[neg_items]


        users_emb_ego = self.user_embedding(users)
        pos_emb_ego = self.item_embedding(pos_items)
        neg_emb_ego = self.item_embedding(neg_items)

        return users_emb, pos_emb, neg_emb, users_emb_ego, pos_emb_ego, neg_emb_ego

    def bpr_loss(self, users, pos_items, neg_items, adj_matrix, reg=1e-4):

        (users_emb, pos_emb, neg_emb,
         userEmb0, posEmb0, negEmb0) = self.getEmbedding(
             users.long(), pos_items.long(), neg_items.long(), adj_matrix
         )


        pos_scores = torch.mul(users_emb, pos_emb)
        pos_scores = torch.sum(pos_scores, dim=1)

        neg_scores = torch.mul(users_emb, neg_emb)
        neg_scores = torch.sum(neg_scores, dim=1)


        loss = torch.mean(torch.nn.functional.softplus(neg_scores - pos_scores))


        reg_loss = (1.0 / 2.0) * (
            userEmb0.norm(2).pow(2) +
            posEmb0.norm(2).pow(2) +
            negEmb0.norm(2).pow(2)
        ) / float(len(users))


        reg_loss = reg_loss * reg

        return loss, reg_loss

    def predict(self, user_emb, item_emb, users, items):
        user_vecs = user_emb[users]
        item_vecs = item_emb[items]


        scores = torch.sum(user_vecs * item_vecs, dim=1)

        return scores


def create_adjacency_matrix(n_users, n_items, interactions):


    user_indices = [u for u, _ in interactions]
    item_indices = [i + n_users for _, i in interactions]


    row_indices = user_indices + item_indices
    col_indices = item_indices + user_indices


    values = np.ones(len(row_indices))
    adj = coo_matrix(
        (values, (row_indices, col_indices)),
        shape=(n_users + n_items, n_users + n_items)
    )


    rowsum = np.array(adj.sum(1)).flatten()
    d_inv_sqrt = np.zeros_like(rowsum, dtype=np.float64)
    positive = rowsum > 0
    d_inv_sqrt[positive] = np.power(rowsum[positive], -0.5)


    d_mat_inv_sqrt = coo_matrix(
        (d_inv_sqrt, (np.arange(len(d_inv_sqrt)), np.arange(len(d_inv_sqrt)))),
        shape=adj.shape
    )


    adj_normalized = d_mat_inv_sqrt.dot(adj).dot(d_mat_inv_sqrt)


    adj_normalized = adj_normalized.tocoo()


    indices = torch.from_numpy(
        np.vstack([adj_normalized.row, adj_normalized.col])
    ).long()
    values = torch.from_numpy(adj_normalized.data).float()
    shape = torch.Size(adj_normalized.shape)


    adj_tensor = torch.sparse_coo_tensor(indices, values, shape, dtype=torch.float32)

    return adj_tensor
