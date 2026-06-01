"""Composite cross-modal VAE loss.

L = L_recon + β·L_kl + λ_cross·L_cross + λ_align·L_align

- L_recon = Chamfer(mri, recon_mri) + Chamfer(pli, recon_pli)
- L_kl    = KL(q_mri || N(0,I))      + KL(q_pli || N(0,I))
- L_cross = Chamfer(cross_mri_to_pli, pli) + Chamfer(cross_pli_to_mri, mri)
- L_align = MSE(μ_mri, μ_pli)

The cross term is the alignment mechanism the supervisor specified: it
forces the shared decoder to translate an MRI-encoded z into PLI-shaped
geometry (and vice versa), independent of streamline correspondence.
"""

from __future__ import annotations

from dataclasses import dataclass

import torch

from cross_modal_vae.losses.chamfer import chamfer_distance
from cross_modal_vae.losses.kl import kl_to_standard_normal
from cross_modal_vae.models.vae import CrossModalVAE


@dataclass(frozen=True)
class LossWeights:
    beta: float = 1.0
    lambda_cross: float = 1.0
    lambda_align: float = 0.1


def compute_losses(
    vae: CrossModalVAE,
    mri: torch.Tensor,
    pli: torch.Tensor,
    weights: LossWeights,
) -> dict[str, torch.Tensor]:
    recon_mri, mu_mri, log_sigma_mri = vae.forward_mri(mri)
    recon_pli, mu_pli, log_sigma_pli = vae.forward_pli(pli)

    recon = chamfer_distance(mri, recon_mri) + chamfer_distance(pli, recon_pli)
    kl = kl_to_standard_normal(mu_mri, log_sigma_mri) + kl_to_standard_normal(
        mu_pli, log_sigma_pli
    )

    cross_mri_to_pli = vae.cross_forward(mri, source="mri")
    cross_pli_to_mri = vae.cross_forward(pli, source="pli")
    cross = chamfer_distance(cross_mri_to_pli, pli) + chamfer_distance(
        cross_pli_to_mri, mri
    )

    align = ((mu_mri - mu_pli) ** 2).mean()

    total = (
        recon
        + weights.beta * kl
        + weights.lambda_cross * cross
        + weights.lambda_align * align
    )

    return {
        "total": total,
        "recon": recon,
        "kl": kl,
        "cross": cross,
        "align": align,
    }
