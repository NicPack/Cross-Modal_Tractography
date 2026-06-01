import torch

from cross_modal_vae.models.vae import CrossModalVAE
from cross_modal_vae.training.losses import LossWeights
from cross_modal_vae.training.train_step import (
    load_checkpoint,
    run_train_step,
    save_checkpoint,
)


def _vae():
    torch.manual_seed(0)
    return CrossModalVAE(
        K=4, n_points=16, d_model=24, latent_dim=16,
        n_heads=4, n_layers_streamline=2, n_layers_set=1, n_layers_decoder=2,
    )


def test_single_train_step_returns_loss_components():
    vae = _vae()
    opt = torch.optim.Adam(vae.parameters(), lr=1e-3)
    mri = torch.randn(2, 4, 16, 3)
    pli = torch.randn(2, 4, 16, 3)
    out = run_train_step(vae, opt, mri, pli, LossWeights(beta=0.0))
    assert {"total", "recon", "kl", "cross", "align"} <= out.keys()
    for v in out.values():
        assert isinstance(v, float)


def test_multiple_train_steps_reduce_loss_on_a_fixed_batch():
    # Overfit on a tiny fixed batch -> loss must decrease across steps.
    torch.manual_seed(0)
    vae = _vae()
    opt = torch.optim.Adam(vae.parameters(), lr=3e-3)
    mri = torch.randn(2, 4, 16, 3)
    pli = torch.randn(2, 4, 16, 3)

    losses = []
    for _ in range(20):
        out = run_train_step(vae, opt, mri, pli, LossWeights(beta=0.0))
        losses.append(out["total"])

    # First 3 epochs vs last 3 epochs -> last should be clearly lower.
    early = sum(losses[:3]) / 3
    late = sum(losses[-3:]) / 3
    assert late < early, f"loss did not decrease: early={early}, late={late}"


def test_checkpoint_roundtrip_preserves_outputs(tmp_path):
    vae = _vae()
    vae.eval()
    x = torch.randn(1, 4, 16, 3)
    with torch.no_grad():
        before, _, _ = vae.forward_mri(x)

    path = tmp_path / "ckpt.pt"
    save_checkpoint(vae, path)

    vae2 = _vae()  # fresh init, different random params
    # Sanity: fresh instance is different.
    with torch.no_grad():
        diff, _, _ = vae2.forward_mri(x)
    assert not torch.allclose(before, diff, atol=1e-4)

    load_checkpoint(vae2, path)
    vae2.eval()
    with torch.no_grad():
        after, _, _ = vae2.forward_mri(x)
    assert torch.allclose(before, after, atol=1e-6)
