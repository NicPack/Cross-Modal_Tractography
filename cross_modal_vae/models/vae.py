"""Cross-modal VAE: two encoders + one shared decoder.

The MRI and PLI encoders are independent (different parameters); they
emit posteriors q_mri(z|x) and q_pli(z|x) over the same latent space
R^latent_dim. A single decoder reconstructs streamline geometry from z,
which is the mechanism the cross-modal Chamfer + alignment losses
exploit at training time.

In eval mode the latent is the posterior mean (deterministic). In train
mode we reparameterise with Gaussian noise.
"""

from __future__ import annotations

import torch
from torch import nn

from .decoder import BundleDecoder
from .transformer_encoder import BundleEncoder


class CrossModalVAE(nn.Module):
    def __init__(
        self,
        K: int,
        n_points: int,
        d_model: int = 128,
        latent_dim: int = 256,
        n_heads: int = 8,
        n_layers_streamline: int = 4,
        n_layers_set: int = 2,
        n_layers_decoder: int = 4,
        dropout: float = 0.0,
    ) -> None:
        super().__init__()
        enc_kwargs = dict(
            n_points=n_points,
            d_model=d_model,
            latent_dim=latent_dim,
            n_heads=n_heads,
            n_layers_streamline=n_layers_streamline,
            n_layers_set=n_layers_set,
            dropout=dropout,
        )
        self.encoder_mri = BundleEncoder(**enc_kwargs)
        self.encoder_pli = BundleEncoder(**enc_kwargs)
        self.decoder = BundleDecoder(
            K=K,
            n_points=n_points,
            d_model=d_model,
            latent_dim=latent_dim,
            n_heads=n_heads,
            n_layers=n_layers_decoder,
            dropout=dropout,
        )
        self.latent_dim = latent_dim

    def _reparameterise(self, mu: torch.Tensor, log_sigma: torch.Tensor) -> torch.Tensor:
        if self.training:
            eps = torch.randn_like(mu)
            return mu + torch.exp(log_sigma) * eps
        return mu

    def _forward_with(
        self, encoder: BundleEncoder, x: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        mu, log_sigma = encoder(x)
        z = self._reparameterise(mu, log_sigma)
        recon = self.decoder(z)
        return recon, mu, log_sigma

    def forward_mri(self, x: torch.Tensor):
        return self._forward_with(self.encoder_mri, x)

    def forward_pli(self, x: torch.Tensor):
        return self._forward_with(self.encoder_pli, x)

    def cross_forward(self, x: torch.Tensor, *, source: str) -> torch.Tensor:
        """Encode with one modality's encoder, decode with the shared decoder."""
        if source == "mri":
            encoder = self.encoder_mri
        elif source == "pli":
            encoder = self.encoder_pli
        else:
            raise ValueError(f"unknown source: {source!r}")
        mu, log_sigma = encoder(x)
        z = self._reparameterise(mu, log_sigma)
        return self.decoder(z)
