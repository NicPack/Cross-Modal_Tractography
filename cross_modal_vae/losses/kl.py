"""KL divergence from a diagonal-Gaussian posterior to N(0, I)."""

from __future__ import annotations

import torch


def kl_to_standard_normal(mu: torch.Tensor, log_sigma: torch.Tensor) -> torch.Tensor:
    """KL(N(mu, sigma^2) || N(0, 1)) summed over latent dims, averaged over batch.

    mu, log_sigma: (B, D)
    returns: scalar
    """
    var = torch.exp(2.0 * log_sigma)
    per_dim = 0.5 * (var + mu.pow(2) - 1.0 - 2.0 * log_sigma)
    return per_dim.sum(dim=-1).mean()
