import torch

from cross_modal_vae.models.vae import CrossModalVAE
from cross_modal_vae.training.losses import LossWeights, compute_losses


def _vae(K=4, P=16, latent_dim=16):
    torch.manual_seed(0)
    return CrossModalVAE(
        K=K, n_points=P, d_model=24, latent_dim=latent_dim,
        n_heads=4, n_layers_streamline=2, n_layers_set=1, n_layers_decoder=2,
    )


def _batch(K=4, P=16, B=2):
    return torch.randn(B, K, P, 3), torch.randn(B, K, P, 3)


def test_loss_returns_named_components_and_total():
    vae = _vae()
    vae.eval()
    mri, pli = _batch()
    w = LossWeights(beta=1.0, lambda_cross=1.0, lambda_align=0.1)
    out = compute_losses(vae, mri, pli, w)
    assert {"total", "recon", "kl", "cross", "align"} <= set(out.keys())
    for key in ("recon", "kl", "cross", "align", "total"):
        assert torch.isfinite(out[key]).all()
    expected_total = (
        out["recon"]
        + w.beta * out["kl"]
        + w.lambda_cross * out["cross"]
        + w.lambda_align * out["align"]
    )
    assert torch.allclose(out["total"], expected_total, atol=1e-6)


def test_beta_zero_zeros_out_kl_contribution_to_total():
    vae = _vae()
    vae.eval()
    mri, pli = _batch()
    out_beta1 = compute_losses(vae, mri, pli, LossWeights(beta=1.0))
    out_beta0 = compute_losses(vae, mri, pli, LossWeights(beta=0.0))
    # KL component itself is unchanged; only its contribution to total disappears.
    assert torch.allclose(out_beta1["kl"], out_beta0["kl"], atol=1e-6)
    diff = out_beta1["total"] - out_beta0["total"]
    assert torch.allclose(diff, out_beta1["kl"], atol=1e-6)


def test_align_is_mse_between_modality_means():
    vae = _vae()
    vae.eval()
    mri, pli = _batch()
    out = compute_losses(vae, mri, pli, LossWeights())
    # Recompute the alignment loss independently from the means.
    with torch.no_grad():
        mu_mri, _ = vae.encoder_mri(mri)
        mu_pli, _ = vae.encoder_pli(pli)
    expected_align = ((mu_mri - mu_pli) ** 2).mean()
    assert torch.allclose(out["align"], expected_align, atol=1e-5)


def test_kl_is_nonnegative_and_chamfer_is_nonnegative():
    vae = _vae()
    vae.eval()
    mri, pli = _batch()
    out = compute_losses(vae, mri, pli, LossWeights())
    assert out["kl"].item() >= -1e-6
    assert out["recon"].item() >= -1e-6
    assert out["cross"].item() >= -1e-6
    assert out["align"].item() >= -1e-6


def test_total_loss_is_differentiable_through_all_components():
    vae = _vae()
    vae.train()
    mri, pli = _batch()
    out = compute_losses(vae, mri, pli, LossWeights(beta=0.5, lambda_cross=2.0, lambda_align=0.25))
    out["total"].backward()
    # All trainable parameters should have non-None grads after backward.
    for name, p in vae.named_parameters():
        if p.requires_grad:
            assert p.grad is not None, f"no grad on {name}"
