import torch

from cross_modal_vae.losses.chamfer import chamfer_distance


def test_chamfer_is_zero_for_identical_bundles():
    bundle = torch.randn(1, 4, 16, 3)  # (B, K, P, 3)
    d = chamfer_distance(bundle, bundle.clone())
    assert torch.allclose(d, torch.zeros_like(d), atol=1e-6)


def test_chamfer_known_toy_case():
    # A is a single point at origin, B is a single point at (1, 0, 0).
    # Squared-L2 Chamfer (symmetric, mean over points):
    #   forward: ||[0,0,0] - [1,0,0]||^2 = 1
    #   backward: ||[1,0,0] - [0,0,0]||^2 = 1
    # total = 1 + 1 = 2
    a = torch.tensor([[[[0.0, 0.0, 0.0]]]])  # (1, 1, 1, 3)
    b = torch.tensor([[[[1.0, 0.0, 0.0]]]])
    d = chamfer_distance(a, b)
    assert torch.allclose(d, torch.tensor(2.0), atol=1e-6)


def test_chamfer_is_symmetric():
    torch.manual_seed(0)
    a = torch.randn(2, 5, 8, 3)
    b = torch.randn(2, 5, 8, 3)
    d_ab = chamfer_distance(a, b)
    d_ba = chamfer_distance(b, a)
    assert torch.allclose(d_ab, d_ba, atol=1e-6)


def test_chamfer_is_differentiable():
    torch.manual_seed(1)
    a = torch.randn(1, 3, 4, 3, requires_grad=True)
    b = torch.randn(1, 3, 4, 3)
    d = chamfer_distance(a, b)
    d.backward()
    assert a.grad is not None
    assert torch.isfinite(a.grad).all()
    assert (a.grad.abs().sum() > 0).item()


def test_chamfer_batched_returns_scalar_mean_across_batch():
    # By convention we return a scalar (mean over batch) so it can be added
    # directly to other scalar loss terms.
    a = torch.randn(4, 2, 8, 3)
    d = chamfer_distance(a, a.clone())
    assert d.ndim == 0
