"""Bundle decoder.

z (B, latent_dim) -> K learnable seed queries -> Transformer decoder
cross-attending to a z-derived memory token -> per-query MLP producing
a (P, 3) streamline -> (B, K, P, 3).
"""

from __future__ import annotations

import torch
from torch import nn


class BundleDecoder(nn.Module):
    def __init__(
        self,
        K: int,
        n_points: int,
        d_model: int = 128,
        latent_dim: int = 256,
        n_heads: int = 8,
        n_layers: int = 4,
        dim_feedforward: int | None = None,
        dropout: float = 0.0,
    ) -> None:
        super().__init__()
        if dim_feedforward is None:
            dim_feedforward = 4 * d_model

        self.K = K
        self.n_points = n_points
        self.d_model = d_model

        # Project z into the decoder's memory, expanded into a few tokens so
        # cross-attention has somewhere to look up structure.
        self.memory_tokens = 4
        self.z_to_memory = nn.Sequential(
            nn.Linear(latent_dim, d_model * self.memory_tokens),
            nn.GELU(),
            nn.Linear(d_model * self.memory_tokens, d_model * self.memory_tokens),
        )

        self.seed_queries = nn.Parameter(torch.randn(K, d_model) * 0.02)

        layer = nn.TransformerDecoderLayer(
            d_model=d_model,
            nhead=n_heads,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            batch_first=True,
            norm_first=True,
        )
        self.decoder_tx = nn.TransformerDecoder(layer, num_layers=n_layers)

        self.point_head = nn.Sequential(
            nn.Linear(d_model, d_model),
            nn.GELU(),
            nn.Linear(d_model, n_points * 3),
        )

    def forward(self, z: torch.Tensor) -> torch.Tensor:
        """z: (B, latent_dim) -> (B, K, P, 3)."""
        B = z.shape[0]
        memory = self.z_to_memory(z).view(B, self.memory_tokens, self.d_model)
        queries = self.seed_queries.unsqueeze(0).expand(B, -1, -1)  # (B, K, d)
        refined = self.decoder_tx(tgt=queries, memory=memory)        # (B, K, d)
        points = self.point_head(refined)                            # (B, K, P*3)
        return points.view(B, self.K, self.n_points, 3)
