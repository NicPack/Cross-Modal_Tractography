import torch

from cross_modal_vae.models.transformer_encoder import BundleEncoder


def _small_encoder(latent_dim: int = 32) -> BundleEncoder:
    torch.manual_seed(0)
    return BundleEncoder(
        n_points=16,
        d_model=24,
        latent_dim=latent_dim,
        n_heads=4,
        n_layers_streamline=2,
        n_layers_set=2,
    )


def test_encoder_output_shapes():
    enc = _small_encoder(latent_dim=32)
    enc.eval()
    x = torch.randn(2, 8, 16, 3)  # (B, K, P, 3)
    mu, log_sigma = enc(x)
    assert mu.shape == (2, 32)
    assert log_sigma.shape == (2, 32)
    assert torch.isfinite(mu).all()
    assert torch.isfinite(log_sigma).all()


def test_encoder_is_permutation_invariant_over_K():
    enc = _small_encoder()
    enc.eval()
    x = torch.randn(1, 6, 16, 3)
    perm = torch.randperm(6)
    x_perm = x[:, perm, :, :]
    with torch.no_grad():
        mu_a, ls_a = enc(x)
        mu_b, ls_b = enc(x_perm)
    assert torch.allclose(mu_a, mu_b, atol=1e-5)
    assert torch.allclose(ls_a, ls_b, atol=1e-5)


def test_encoder_is_NOT_permutation_invariant_over_P():
    # Points along a streamline are ordered (a curve). The encoder must
    # distinguish a streamline from a reversed/shuffled copy.
    enc = _small_encoder()
    enc.eval()
    x = torch.randn(1, 4, 16, 3)
    perm = torch.randperm(16)
    x_perm = x[:, :, perm, :]
    with torch.no_grad():
        mu_a, _ = enc(x)
        mu_b, _ = enc(x_perm)
    assert not torch.allclose(mu_a, mu_b, atol=1e-3)


def test_encoder_gradient_flows():
    enc = _small_encoder()
    x = torch.randn(1, 4, 16, 3, requires_grad=True)
    mu, log_sigma = enc(x)
    (mu.sum() + log_sigma.sum()).backward()
    assert x.grad is not None
    assert torch.isfinite(x.grad).all()
    assert (x.grad.abs().sum() > 0).item()
