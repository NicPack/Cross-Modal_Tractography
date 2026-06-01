import pytest

from cross_modal_vae.losses.schedules import beta_anneal


def test_beta_is_zero_during_warmup():
    assert beta_anneal(epoch=0, warmup=50, ramp=50, target=1.0) == 0.0
    assert beta_anneal(epoch=49, warmup=50, ramp=50, target=1.0) == 0.0


def test_beta_reaches_target_at_end_of_ramp():
    # warmup=50, ramp=50 -> target reached at epoch 100
    assert beta_anneal(epoch=100, warmup=50, ramp=50, target=1.0) == 1.0
    assert beta_anneal(epoch=999, warmup=50, ramp=50, target=1.0) == 1.0


def test_beta_is_linear_during_ramp():
    # epoch 50: t=0/50 -> 0
    # epoch 75: t=25/50 -> 0.5
    # epoch 99: t=49/50 -> 0.98
    assert beta_anneal(epoch=50, warmup=50, ramp=50, target=1.0) == pytest.approx(0.0)
    assert beta_anneal(epoch=75, warmup=50, ramp=50, target=1.0) == pytest.approx(0.5)
    assert beta_anneal(epoch=99, warmup=50, ramp=50, target=1.0) == pytest.approx(0.98)


def test_beta_scales_with_target():
    assert beta_anneal(epoch=100, warmup=50, ramp=50, target=2.5) == 2.5
    assert beta_anneal(epoch=75, warmup=50, ramp=50, target=2.0) == pytest.approx(1.0)


def test_zero_ramp_means_step_function():
    # ramp=0 means: jump from 0 to target at epoch == warmup
    assert beta_anneal(epoch=49, warmup=50, ramp=0, target=1.0) == 0.0
    assert beta_anneal(epoch=50, warmup=50, ramp=0, target=1.0) == 1.0
