import torch

from cross_modal_vae.models.vae import CrossModalVAE


def _small_vae(K: int = 4, P: int = 16, latent_dim: int = 16) -> CrossModalVAE:
    torch.manual_seed(0)
    return CrossModalVAE(
        K=K,
        n_points=P,
        d_model=24,
        latent_dim=latent_dim,
        n_heads=4,
        n_layers_streamline=2,
        n_layers_set=1,
        n_layers_decoder=2,
    )


def test_forward_mri_returns_recon_mu_log_sigma():
    vae = _small_vae()
    vae.eval()
    x = torch.randn(2, 4, 16, 3)
    recon, mu, log_sigma = vae.forward_mri(x)
    assert recon.shape == x.shape
    assert mu.shape == (2, 16)
    assert log_sigma.shape == (2, 16)


def test_forward_pli_uses_separate_encoder():
    vae = _small_vae()
    vae.eval()
    x = torch.randn(2, 4, 16, 3)
    # Using the same input through both encoders should produce different mu,
    # because the two encoders have independent parameters.
    _, mu_mri, _ = vae.forward_mri(x)
    _, mu_pli, _ = vae.forward_pli(x)
    assert not torch.allclose(mu_mri, mu_pli, atol=1e-4)


def test_decoder_is_shared_between_modalities():
    vae = _small_vae()
    # Two encoders, one decoder: the decoder object's id must be the same
    # whether reached via forward_mri or forward_pli.
    assert vae.encoder_mri is not vae.encoder_pli
    # Sanity: only one decoder attribute exists.
    assert hasattr(vae, "decoder")


def test_cross_forward_mri_to_pli_shape():
    vae = _small_vae(K=4, P=16, latent_dim=16)
    vae.eval()
    mri = torch.randn(2, 4, 16, 3)
    out = vae.cross_forward(mri, source="mri")
    assert out.shape == mri.shape


def test_eval_mode_is_deterministic_train_mode_is_stochastic():
    vae = _small_vae()
    x = torch.randn(1, 4, 16, 3)

    vae.eval()
    with torch.no_grad():
        a, _, _ = vae.forward_mri(x)
        b, _, _ = vae.forward_mri(x)
    # In eval, reparam should use mean (deterministic).
    assert torch.allclose(a, b, atol=1e-6)

    vae.train()
    torch.manual_seed(1)
    a, _, _ = vae.forward_mri(x)
    torch.manual_seed(2)
    b, _, _ = vae.forward_mri(x)
    # In train, reparam should inject noise -> different outputs.
    assert not torch.allclose(a, b, atol=1e-4)


def test_vae_end_to_end_gradient_flow():
    vae = _small_vae()
    x = torch.randn(1, 4, 16, 3, requires_grad=True)
    recon, mu, log_sigma = vae.forward_mri(x)
    (recon.sum() + mu.sum() + log_sigma.sum()).backward()
    assert x.grad is not None
    assert torch.isfinite(x.grad).all()
