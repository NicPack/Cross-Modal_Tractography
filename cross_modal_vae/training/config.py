"""Hyperparameters for cross-modal VAE training."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class TrainConfig:
    # Data / bundle shape
    K: int = 64
    n_points: int = 256

    # Model
    d_model: int = 128
    latent_dim: int = 256
    n_heads: int = 8
    n_layers_streamline: int = 4
    n_layers_set: int = 2
    n_layers_decoder: int = 4
    dropout: float = 0.0

    # Training schedule
    epochs: int = 200
    bundles_per_epoch: int = 256
    batch_size: int = 8
    lr: float = 3e-4

    # Loss weights (supervisor's starting values)
    beta_target: float = 1.0
    beta_warmup_epochs: int = 50
    beta_ramp_epochs: int = 50
    lambda_cross: float = 1.0
    lambda_align: float = 0.1
    # Per-dim KL floor (nats) to prevent posterior collapse. 0.0 = off (the
    # brief's behaviour); set via --free-bits at the CLI for experiments.
    free_bits: float = 0.0

    # Misc
    seed: int = 0
    device: str = "cpu"
    out_dir: str = "cross_modal_vae/runs/default"
