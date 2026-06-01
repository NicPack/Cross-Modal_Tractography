"""Two-stage Transformer encoder for streamline bundles.

Stage 1 — per-streamline self-attention with positional encoding over P.
Stage 2 — set-level self-attention over the K streamline embeddings,
          NO positional encoding so it is permutation-equivariant in K.

Mean-pool at the end of each stage and an MLP head produce (μ, log σ).
"""

from __future__ import annotations

import math

import torch
from torch import nn


def _sinusoidal_pe(length: int, d_model: int) -> torch.Tensor:
    """Standard sinusoidal positional encoding, shape (length, d_model)."""
    pe = torch.zeros(length, d_model)
    position = torch.arange(length, dtype=torch.float32).unsqueeze(1)
    div = torch.exp(
        torch.arange(0, d_model, 2, dtype=torch.float32) * (-math.log(10000.0) / d_model)
    )
    pe[:, 0::2] = torch.sin(position * div)
    pe[:, 1::2] = torch.cos(position * div)
    return pe


class BundleEncoder(nn.Module):
    def __init__(
        self,
        n_points: int,
        d_model: int = 128,
        latent_dim: int = 256,
        n_heads: int = 8,
        n_layers_streamline: int = 4,
        n_layers_set: int = 2,
        dim_feedforward: int | None = None,
        dropout: float = 0.0,
    ) -> None:
        super().__init__()
        if dim_feedforward is None:
            dim_feedforward = 4 * d_model

        self.n_points = n_points
        self.d_model = d_model
        self.latent_dim = latent_dim

        self.input_proj = nn.Linear(3, d_model)
        self.register_buffer("pos_enc", _sinusoidal_pe(n_points, d_model))

        streamline_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=n_heads,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            batch_first=True,
            norm_first=True,
        )
        self.streamline_tx = nn.TransformerEncoder(
            streamline_layer, num_layers=n_layers_streamline, enable_nested_tensor=False
        )

        set_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=n_heads,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            batch_first=True,
            norm_first=True,
        )
        self.set_tx = nn.TransformerEncoder(
            set_layer, num_layers=n_layers_set, enable_nested_tensor=False
        )

        self.head_mu = nn.Sequential(
            nn.Linear(d_model, d_model),
            nn.GELU(),
            nn.Linear(d_model, latent_dim),
        )
        self.head_log_sigma = nn.Sequential(
            nn.Linear(d_model, d_model),
            nn.GELU(),
            nn.Linear(d_model, latent_dim),
        )

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """x: (B, K, P, 3) -> (mu, log_sigma) each (B, latent_dim)."""
        B, K, P, _ = x.shape
        h = self.input_proj(x)                          # (B, K, P, d)
        h = h + self.pos_enc.unsqueeze(0).unsqueeze(0)  # add PE over P

        # per-streamline transformer: flatten K into batch
        h = h.reshape(B * K, P, self.d_model)
        h = self.streamline_tx(h)
        h = h.mean(dim=1)                               # (B*K, d)
        h = h.reshape(B, K, self.d_model)

        # set-level transformer over K (no PE -> permutation-equivariant)
        h = self.set_tx(h)
        h = h.mean(dim=1)                               # (B, d)

        return self.head_mu(h), self.head_log_sigma(h)
