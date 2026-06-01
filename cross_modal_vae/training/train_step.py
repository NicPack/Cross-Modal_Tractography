"""Single optimisation step + checkpoint I/O."""

from __future__ import annotations

from pathlib import Path

import torch

from .losses import LossWeights, compute_losses
from cross_modal_vae.models.vae import CrossModalVAE


def run_train_step(
    vae: CrossModalVAE,
    optimizer: torch.optim.Optimizer,
    mri: torch.Tensor,
    pli: torch.Tensor,
    weights: LossWeights,
) -> dict[str, float]:
    vae.train()
    optimizer.zero_grad(set_to_none=True)
    out = compute_losses(vae, mri, pli, weights)
    out["total"].backward()
    optimizer.step()
    return {k: v.detach().item() for k, v in out.items()}


def save_checkpoint(vae: CrossModalVAE, path: Path | str) -> None:
    torch.save({"model_state": vae.state_dict()}, path)


def load_checkpoint(vae: CrossModalVAE, path: Path | str) -> None:
    payload = torch.load(path, map_location="cpu", weights_only=True)
    vae.load_state_dict(payload["model_state"])
