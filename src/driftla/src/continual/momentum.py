
import torch
import torch.nn as nn
import copy
from typing import Optional


class MomentumEncoder:

    def __init__(self, model: nn.Module, momentum: float = 0.999):
        self.momentum = momentum
        self.ref_model = copy.deepcopy(model)
        for param in self.ref_model.parameters():
            param.requires_grad = False

    @torch.no_grad()
    def update(self, model: nn.Module) -> None:
        for ref_param, param in zip(self.ref_model.parameters(), model.parameters()):
            ref_param.data.mul_(self.momentum).add_(param.data, alpha=1.0 - self.momentum)

    def forward(self, *args, **kwargs):
        self.ref_model.eval()
        with torch.no_grad():
            return self.ref_model(*args, **kwargs)

    def encode_paths(
        self,
        paths,
        all_emb_ref: torch.Tensor,
        temporal_encoder,
        timestamps_dict,
        n_users: int,
        encode_fn,
    ) -> torch.Tensor:
        embs = []
        for path in paths:
            emb = encode_fn(path, all_emb_ref, temporal_encoder, timestamps_dict, n_users)
            embs.append(emb)
        if not embs:
            return torch.zeros(0, device=all_emb_ref.device)
        return torch.stack(embs)

    def compute_stability_loss(
        self,
        current_path_embs: torch.Tensor,
        ref_path_embs: torch.Tensor,
    ) -> torch.Tensor:
        if current_path_embs.numel() == 0 or ref_path_embs.numel() == 0:
            return torch.tensor(0.0, device=current_path_embs.device)
        return ((current_path_embs - ref_path_embs) ** 2).mean()

    def get_ref_embeddings(self, adj_matrix: torch.Tensor):
        return self.forward(adj_matrix)

    @property
    def parameters(self):
        return self.ref_model.parameters()
