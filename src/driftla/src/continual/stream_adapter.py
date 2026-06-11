
from __future__ import annotations

from typing import Literal

import numpy as np
import torch
import torch.nn as nn

AdapterGateMode = Literal["drift", "ungated", "fixed", "random", "plain_lora"]

GATE_MODES: tuple[str, ...] = ("drift", "ungated", "fixed", "random", "plain_lora")


class DriftGatedStreamAdapter(nn.Module):

    def __init__(
        self,
        embed_dim: int,
        rank: int = 8,
        gate_mode: AdapterGateMode = "drift",
        fixed_gate: float = 0.5,
        gate_seed: int = 42,
    ):
        super().__init__()
        self.gate_mode: AdapterGateMode = gate_mode
        self.fixed_gate = float(fixed_gate)
        self.gate_seed = int(gate_seed)
        self._stream_batch_id = 0

        self.user_branch = nn.Sequential(
            nn.Linear(embed_dim, rank),
            nn.GELU(),
            nn.Linear(rank, embed_dim),
        )
        self.item_branch = nn.Sequential(
            nn.Linear(embed_dim, rank),
            nn.GELU(),
            nn.Linear(rank, embed_dim),
        )

        nn.init.zeros_(self.user_branch[-1].weight)
        nn.init.zeros_(self.user_branch[-1].bias)
        nn.init.zeros_(self.item_branch[-1].weight)
        nn.init.zeros_(self.item_branch[-1].bias)

    def set_stream_batch_id(self, batch_id: int) -> None:
        self._stream_batch_id = int(batch_id)

    def drift_gate(
        self,
        gamma: float,
        gamma_floor: float,
        ramp: float,
        min_gate: float,
    ) -> float:
        raw = (float(gamma) - float(gamma_floor)) / max(float(ramp), 1e-6)
        raw = float(max(0.0, min(1.0, raw)))
        mg = float(max(0.0, min(0.5, min_gate)))
        return mg + (1.0 - mg) * raw

    def resolve_gate(
        self,
        gamma: float,
        gamma_floor: float,
        ramp: float,
        min_gate: float,
    ) -> float:
        mode = self.gate_mode
        if mode in ("ungated", "plain_lora"):
            return 1.0
        if mode == "fixed":
            return float(max(0.0, min(1.0, self.fixed_gate)))
        if mode == "random":
            mg = float(max(0.0, min(0.5, min_gate)))
            rng = np.random.default_rng(self.gate_seed + self._stream_batch_id)
            return float(rng.uniform(mg, 1.0))
        return self.drift_gate(gamma, gamma_floor, ramp, min_gate)

    def gate(
        self,
        gamma: float,
        gamma_floor: float,
        ramp: float,
        min_gate: float,
    ) -> float:
        return self.resolve_gate(gamma, gamma_floor, ramp, min_gate)

    def forward(
        self,
        user_emb: torch.Tensor,
        item_emb: torch.Tensor,
        gamma: float,
        gamma_floor: float,
        ramp: float,
        min_gate: float,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        g = self.resolve_gate(gamma, gamma_floor, ramp, min_gate)
        if g <= 0.0:
            return user_emb, item_emb
        gt = user_emb.new_tensor(g)
        u_out = user_emb + gt * self.user_branch(user_emb)
        i_out = item_emb + gt * self.item_branch(item_emb)
        return u_out, i_out
