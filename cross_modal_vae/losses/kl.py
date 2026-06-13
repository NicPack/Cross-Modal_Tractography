"""KL divergence from a diagonal-Gaussian posterior to N(0, I)."""

from __future__ import annotations

import torch


def kl_to_standard_normal(
    mu: torch.Tensor, log_sigma: torch.Tensor, free_bits: float = 0.0
) -> torch.Tensor:
    """KL(N(mu, sigma^2) || N(0, 1)) summed over latent dims, averaged over batch.

    mu, log_sigma: (B, D)
    free_bits: per-dimension KL floor in nats. When > 0, each latent dim's KL is
        clamped up to this value *before* summing, so the optimiser feels no
        pressure to push a dim's KL below the floor. This is Kingma's "free
        bits" trick — it reserves a guaranteed amount of information capacity in
        the latent and is the standard antidote to posterior collapse (where KL
        is driven to ~0 and the decoder ignores z). free_bits=0.0 is a no-op and
        recovers the plain KL.
    returns: scalar
    """
    var = torch.exp(2.0 * log_sigma)
    per_dim = 0.5 * (var + mu.pow(2) - 1.0 - 2.0 * log_sigma)
    if free_bits > 0.0:
        per_dim = per_dim.clamp(min=free_bits)
    return per_dim.sum(dim=-1).mean()
