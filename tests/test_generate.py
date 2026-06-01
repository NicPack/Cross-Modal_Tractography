"""Tests for cross_modal_vae.evaluation.generate — inference / .trk export.

The contract pinned here: MRI (raw) → normalize with stats_mri → encode with
MRI encoder → decode with shared decoder (cross_forward, source='mri') →
denormalize with stats_pli → .trk file in PLI's RASMM frame. Eval mode is
deterministic (posterior = μ).
"""
from __future__ import annotations

import subprocess
import sys

import numpy as np
import nibabel as nib
import torch

from cross_modal_vae.data.normalize import (
    NormalizationStats,
    fit_stats,
    save_stats,
)
from cross_modal_vae.evaluation.generate import (
    generate_pli_from_mri,
    save_streamlines_trk,
)
from cross_modal_vae.models.vae import CrossModalVAE
from cross_modal_vae.training.config import TrainConfig
from cross_modal_vae.training.train import train_from_arrays
from cross_modal_vae.training.train_step import save_checkpoint


def _make_model(K: int = 4, n_points: int = 8, latent_dim: int = 8) -> CrossModalVAE:
    torch.manual_seed(0)
    model = CrossModalVAE(
        K=K,
        n_points=n_points,
        d_model=16,
        latent_dim=latent_dim,
        n_heads=4,
        n_layers_streamline=1,
        n_layers_set=1,
        n_layers_decoder=1,
    )
    model.eval()
    return model


def _make_streamlines(n: int, p: int, seed: int) -> np.ndarray:
    rng = np.random.default_rng(seed)
    return rng.standard_normal((n, p, 3)).astype(np.float32)


def test_generate_pli_returns_correct_shape():
    K, P = 4, 8
    model = _make_model(K=K, n_points=P)
    mri = _make_streamlines(20, P, seed=0)
    pli = _make_streamlines(30, P, seed=1)
    stats_mri = fit_stats(mri)
    stats_pli = fit_stats(pli)

    out = generate_pli_from_mri(model, mri, stats_mri, stats_pli, K=K, device="cpu")

    assert out.shape == (20, P, 3)
    assert out.dtype == np.float32


def test_generate_pli_drops_remainder():
    K, P = 4, 8
    model = _make_model(K=K, n_points=P)
    mri = _make_streamlines(23, P, seed=0)  # 23 = 5*4 + 3 → 3 dropped
    pli = _make_streamlines(30, P, seed=1)
    stats_mri = fit_stats(mri)
    stats_pli = fit_stats(pli)

    out = generate_pli_from_mri(model, mri, stats_mri, stats_pli, K=K, device="cpu")

    assert out.shape == (20, P, 3)


def test_generate_pli_is_deterministic_in_eval():
    K, P = 4, 8
    model = _make_model(K=K, n_points=P)
    mri = _make_streamlines(16, P, seed=0)
    pli = _make_streamlines(16, P, seed=1)
    stats_mri = fit_stats(mri)
    stats_pli = fit_stats(pli)

    a = generate_pli_from_mri(model, mri, stats_mri, stats_pli, K=K, device="cpu")
    b = generate_pli_from_mri(model, mri, stats_mri, stats_pli, K=K, device="cpu")
    np.testing.assert_array_equal(a, b)


def test_generate_pli_uses_pli_stats_for_denormalization():
    K, P = 4, 8
    model = _make_model(K=K, n_points=P)
    mri = _make_streamlines(16, P, seed=0)
    pli = _make_streamlines(16, P, seed=1)
    stats_mri = fit_stats(mri)
    # Distinctive PLI stats so we can verify they (not stats_mri) are applied.
    stats_pli = NormalizationStats(
        centroid=np.array([10.0, 20.0, 30.0], dtype=np.float32),
        scale=np.array([2.0, 3.0, 4.0], dtype=np.float32),
    )
    # Patch the model so cross_forward yields a known constant in normalised space.
    const = torch.full((1, K, P, 3), 0.5)

    def fake_cross_forward(x, *, source):
        assert source == "mri"
        B = x.shape[0]
        return const.expand(B, -1, -1, -1).clone()

    model.cross_forward = fake_cross_forward  # type: ignore[assignment]

    out = generate_pli_from_mri(model, mri, stats_mri, stats_pli, K=K, device="cpu")

    expected_point = 0.5 * stats_pli.scale + stats_pli.centroid  # (3,)
    expected = np.broadcast_to(expected_point, (16, P, 3)).astype(np.float32)
    np.testing.assert_allclose(out, expected, atol=1e-5)


def test_generate_pli_uses_mri_encoder_and_mri_stats():
    K, P = 4, 8
    model = _make_model(K=K, n_points=P)
    mri = _make_streamlines(16, P, seed=0)
    pli = _make_streamlines(16, P, seed=1)
    stats_mri = fit_stats(mri)
    stats_pli = fit_stats(pli)

    captured: dict = {}
    original = model.cross_forward

    def spy(x, *, source):
        captured["source"] = source
        captured["input"] = x.detach().cpu().numpy().copy()
        return original(x, source=source)

    model.cross_forward = spy  # type: ignore[assignment]

    _ = generate_pli_from_mri(model, mri, stats_mri, stats_pli, K=K, device="cpu")

    assert captured["source"] == "mri"
    # First bundle of K streamlines, normalized with stats_mri (not stats_pli).
    expected_first = (mri[:K] - stats_mri.centroid) / stats_mri.scale
    np.testing.assert_allclose(captured["input"][0], expected_first, atol=1e-5)


def test_generate_cli_writes_trk(tmp_path):
    # Train a tiny model so we have a real checkpoint + saved stats.
    mri = _make_streamlines(32, 8, seed=0)
    pli = _make_streamlines(32, 8, seed=1)
    cfg = TrainConfig(
        K=4, n_points=8, d_model=16, latent_dim=8, n_heads=4,
        n_layers_streamline=1, n_layers_set=1, n_layers_decoder=1,
        epochs=1, bundles_per_epoch=4, batch_size=2, lr=1e-3,
        beta_target=1.0, beta_warmup_epochs=0, beta_ramp_epochs=1,
        lambda_cross=1.0, lambda_align=0.1, seed=0,
        device="cpu", out_dir=str(tmp_path / "run"),
    )
    train_from_arrays(mri=mri, pli=pli, cfg=cfg)

    mri_npz = tmp_path / "mri.npz"
    np.savez(str(mri_npz), mri)
    out_trk = tmp_path / "gen.trk"

    result = subprocess.run(
        [
            sys.executable,
            "-m",
            "cross_modal_vae.evaluation.generate",
            "--checkpoint",
            str(tmp_path / "run" / "final.pt"),
            "--mri",
            str(mri_npz),
            "--stats-dir",
            str(tmp_path / "run"),
            "--out",
            str(out_trk),
            "--K",
            "4",
            "--n-points",
            "8",
            "--latent-dim",
            "8",
            "--d-model",
            "16",
            "--n-heads",
            "4",
            "--n-layers-streamline",
            "1",
            "--n-layers-set",
            "1",
            "--n-layers-decoder",
            "1",
            "--device",
            "cpu",
        ],
        capture_output=True,
        text=True,
    )
    assert result.returncode == 0, result.stderr
    assert out_trk.exists()
    loaded = nib.streamlines.load(str(out_trk))
    # 32 MRI streamlines, K=4 → 8 bundles → 32 streamlines out.
    assert len(loaded.streamlines) == 32
    for s in loaded.streamlines:
        assert s.shape == (8, 3)


def test_save_streamlines_trk_roundtrip(tmp_path):
    rng = np.random.default_rng(0)
    streamlines = rng.standard_normal((5, 16, 3)).astype(np.float32)
    out = tmp_path / "out.trk"
    save_streamlines_trk(streamlines, out)
    assert out.exists()
    loaded = nib.streamlines.load(str(out))
    assert len(loaded.streamlines) == 5
    for i, s in enumerate(loaded.streamlines):
        np.testing.assert_allclose(s, streamlines[i], atol=1e-5)
