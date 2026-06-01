import numpy as np
import pytest

from cross_modal_vae.data.bundles import PairedBundleDataset, sample_bundle


def _fake_streamlines(n: int, seed: int = 0) -> np.ndarray:
    rng = np.random.default_rng(seed)
    return rng.standard_normal((n, 256, 3)).astype(np.float32)


def test_sample_bundle_returns_K_unique_rows():
    arr = _fake_streamlines(1000)
    rng = np.random.default_rng(42)
    bundle = sample_bundle(rng, arr, K=64)
    assert bundle.shape == (64, 256, 3)
    # each sampled row must be a row of arr (no duplicates by row-vector identity)
    # use a hash on flattened bytes per row, since rows of arr are unique with prob 1
    row_hashes = {bundle[i].tobytes() for i in range(64)}
    assert len(row_hashes) == 64


def test_sample_bundle_is_reproducible_under_seed():
    arr = _fake_streamlines(500)
    b1 = sample_bundle(np.random.default_rng(7), arr, K=32)
    b2 = sample_bundle(np.random.default_rng(7), arr, K=32)
    np.testing.assert_array_equal(b1, b2)


def test_sample_bundle_raises_when_K_exceeds_N():
    arr = _fake_streamlines(10)
    rng = np.random.default_rng(0)
    with pytest.raises(ValueError):
        sample_bundle(rng, arr, K=64)


def test_paired_bundle_dataset_yields_independent_modality_bundles():
    mri = _fake_streamlines(500, seed=1)
    pli = _fake_streamlines(700, seed=2)
    ds = PairedBundleDataset(mri=mri, pli=pli, K=32, bundles_per_epoch=10, seed=0)

    assert len(ds) == 10
    sample = ds[0]
    assert set(sample.keys()) == {"mri", "pli"}
    assert sample["mri"].shape == (32, 256, 3)
    assert sample["pli"].shape == (32, 256, 3)
    # MRI bundle rows must come from mri array, not pli — check by L-inf bound on a
    # known disjoint statistic
    mri_row = sample["mri"][0]
    assert any(np.array_equal(mri_row, mri[i]) for i in range(len(mri)))


def test_paired_bundle_dataset_is_deterministic_under_seed():
    mri = _fake_streamlines(500, seed=1)
    pli = _fake_streamlines(700, seed=2)
    ds_a = PairedBundleDataset(mri=mri, pli=pli, K=16, bundles_per_epoch=5, seed=123)
    ds_b = PairedBundleDataset(mri=mri, pli=pli, K=16, bundles_per_epoch=5, seed=123)
    for i in range(5):
        np.testing.assert_array_equal(ds_a[i]["mri"], ds_b[i]["mri"])
        np.testing.assert_array_equal(ds_a[i]["pli"], ds_b[i]["pli"])
