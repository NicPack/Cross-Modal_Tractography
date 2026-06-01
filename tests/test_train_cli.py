"""Integration test for the assemble-and-train entry point.

Uses tiny synthetic arrays so it runs in a few seconds and exercises the
full pipeline: dataset, model build, train loop, logging dict, checkpoint.
"""
from __future__ import annotations

import numpy as np
import torch

from cross_modal_vae.training.config import TrainConfig
from cross_modal_vae.training.train import train_from_arrays


def _make_synth(n: int, seed: int) -> np.ndarray:
    rng = np.random.default_rng(seed)
    return rng.standard_normal((n, 32, 3)).astype(np.float32)


def test_train_from_arrays_runs_one_epoch_and_logs_components(tmp_path):
    mri = _make_synth(50, 0)
    pli = _make_synth(60, 1)
    cfg = TrainConfig(
        K=4,
        n_points=32,
        d_model=16,
        latent_dim=8,
        n_heads=4,
        n_layers_streamline=1,
        n_layers_set=1,
        n_layers_decoder=1,
        epochs=2,
        bundles_per_epoch=4,
        batch_size=2,
        lr=1e-3,
        beta_target=1.0,
        beta_warmup_epochs=1,
        beta_ramp_epochs=1,
        lambda_cross=1.0,
        lambda_align=0.1,
        seed=0,
        device="cpu",
        out_dir=str(tmp_path / "run"),
    )
    history = train_from_arrays(mri=mri, pli=pli, cfg=cfg)
    assert len(history) == cfg.epochs
    for entry in history:
        for key in ("epoch", "total", "recon", "kl", "cross", "align", "beta"):
            assert key in entry
            assert np.isfinite(entry[key])
    # Beta schedule: epoch 0 < warmup (1) -> 0; epoch 1 inside ramp.
    assert history[0]["beta"] == 0.0
    # Checkpoint written.
    assert (tmp_path / "run" / "final.pt").exists()
    # Normalization stats persisted for each modality.
    assert (tmp_path / "run" / "stats_mri.npz").exists()
    assert (tmp_path / "run" / "stats_pli.npz").exists()


def test_train_from_arrays_is_deterministic_given_seed(tmp_path):
    mri = _make_synth(40, 0)
    pli = _make_synth(40, 1)
    cfg = TrainConfig(
        K=4, n_points=32, d_model=16, latent_dim=8, n_heads=4,
        n_layers_streamline=1, n_layers_set=1, n_layers_decoder=1,
        epochs=1, bundles_per_epoch=4, batch_size=2, lr=1e-3,
        beta_target=1.0, beta_warmup_epochs=0, beta_ramp_epochs=1,
        lambda_cross=1.0, lambda_align=0.1, seed=42,
        device="cpu", out_dir=str(tmp_path / "a"),
    )
    h_a = train_from_arrays(mri=mri, pli=pli, cfg=cfg)
    cfg_b = TrainConfig(**{**cfg.__dict__, "out_dir": str(tmp_path / "b")})
    h_b = train_from_arrays(mri=mri, pli=pli, cfg=cfg_b)
    assert h_a[0]["total"] == h_b[0]["total"]
