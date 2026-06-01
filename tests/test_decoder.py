import torch

from cross_modal_vae.models.decoder import BundleDecoder


def _small_decoder(K: int = 8, P: int = 16, latent_dim: int = 32) -> BundleDecoder:
    torch.manual_seed(0)
    return BundleDecoder(
        K=K, n_points=P, d_model=24, latent_dim=latent_dim, n_heads=4, n_layers=2
    )


def test_decoder_output_shape():
    dec = _small_decoder(K=8, P=16, latent_dim=32)
    dec.eval()
    z = torch.randn(2, 32)
    out = dec(z)
    assert out.shape == (2, 8, 16, 3)
    assert torch.isfinite(out).all()


def test_decoder_responds_to_z():
    # Untrained decoder must still produce different outputs for different z;
    # otherwise the model has already collapsed.
    dec = _small_decoder()
    dec.eval()
    z1 = torch.randn(1, 32)
    z2 = torch.randn(1, 32)
    with torch.no_grad():
        out1 = dec(z1)
        out2 = dec(z2)
    assert not torch.allclose(out1, out2, atol=1e-3)


def test_decoder_gradient_flows_to_z():
    dec = _small_decoder()
    z = torch.randn(1, 32, requires_grad=True)
    out = dec(z)
    out.sum().backward()
    assert z.grad is not None
    assert torch.isfinite(z.grad).all()
    assert (z.grad.abs().sum() > 0).item()
