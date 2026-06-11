
import torch
import torch.nn as nn
import math


class TemporalEncoder(nn.Module):

    def __init__(self, embed_dim):
        super().__init__()
        if embed_dim % 2 != 0:
            raise ValueError("embed_dim must be even for temporal encoding")
        self.embed_dim = embed_dim
        half = embed_dim // 2
        inv_freq = 1.0 / (10000.0 ** (2.0 * torch.arange(half).float() / embed_dim))
        self.register_buffer("inv_freq", inv_freq)

    def forward(self, timestamps):
        squeeze = False
        if timestamps.dim() == 1:
            timestamps = timestamps.unsqueeze(0)
            squeeze = True


        angles = timestamps.unsqueeze(-1) * self.inv_freq.unsqueeze(0).unsqueeze(0)
        sin_enc = torch.sin(angles)
        cos_enc = torch.cos(angles)

        encoding = torch.stack([sin_enc, cos_enc], dim=-1)
        encoding = encoding.reshape(*timestamps.shape, self.embed_dim)

        if squeeze:
            encoding = encoding.squeeze(0)
        return encoding


def hermitian_inner_product(w, e):
    half = w.shape[-1] // 2
    w_real, w_imag = w[..., :half], w[..., half:]
    e_real, e_imag = e[..., :half], e[..., half:]


    mag = torch.sqrt(e_real ** 2 + e_imag ** 2 + 1e-8)
    e_real = e_real / mag
    e_imag = e_imag / mag


    wp_real = w_real * e_real + w_imag * e_imag
    wp_imag = -w_real * e_imag + w_imag * e_real

    return torch.cat([wp_real, wp_imag], dim=-1)


def encode_path_with_temporal(node_embeddings, edge_timestamps, temporal_encoder):
    L = node_embeddings.shape[0]


    time_emb = temporal_encoder(edge_timestamps)


    w = node_embeddings[0]

    for i in range(1, L):

        w_prime = hermitian_inner_product(w, time_emb[i - 1])

        w = node_embeddings[i] + w_prime


    path_emb = w / L
    return path_emb
