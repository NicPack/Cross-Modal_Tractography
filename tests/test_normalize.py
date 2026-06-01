import numpy as np
import pytest

from cross_modal_vae.data.normalize import (
    NormalizationStats,
    apply,
    fit_stats,
    invert,
    load_stats,
    save_stats,
)


def _toy_array() -> np.ndarray:
    rng = np.random.default_rng(0)
    base = rng.standard_normal((100, 256, 3)).astype(np.float32) * np.array(
        [15.4, 18.4, 10.6], dtype=np.float32
    ) + np.array([39.6, 50.9, 38.3], dtype=np.float32)
    return base


def test_fit_stats_returns_centroid_and_scale():
    arr = _toy_array()
    stats = fit_stats(arr)
    assert isinstance(stats, NormalizationStats)
    # centroid = mean over all (N*P) points
    expected_centroid = arr.reshape(-1, 3).mean(axis=0)
    np.testing.assert_allclose(stats.centroid, expected_centroid, atol=1e-5)
    # scale = max-abs of centred coords per axis
    centred = arr.reshape(-1, 3) - expected_centroid
    expected_scale = np.abs(centred).max(axis=0)
    np.testing.assert_allclose(stats.scale, expected_scale, atol=1e-5)


def test_apply_invert_roundtrip():
    arr = _toy_array()
    stats = fit_stats(arr)
    normed = apply(arr, stats)
    # normalized data should fall within [-1, 1]
    assert normed.min() >= -1.0 - 1e-6
    assert normed.max() <= 1.0 + 1e-6
    restored = invert(normed, stats)
    np.testing.assert_allclose(restored, arr, atol=1e-5)


def test_save_load_stats_roundtrip(tmp_path):
    stats = NormalizationStats(
        centroid=np.array([1.0, 2.0, 3.0], dtype=np.float32),
        scale=np.array([4.0, 5.0, 6.0], dtype=np.float32),
    )
    path = tmp_path / "stats.npz"
    save_stats(stats, path)
    loaded = load_stats(path)
    np.testing.assert_array_equal(loaded.centroid, stats.centroid)
    np.testing.assert_array_equal(loaded.scale, stats.scale)


def test_fit_stats_rejects_wrong_shape():
    with pytest.raises(ValueError):
        fit_stats(np.zeros((10, 3), dtype=np.float32))


def test_independent_stats_for_each_modality():
    # MRI and PLI live in different coordinate frames — stats must not be shared.
    mri = _toy_array()
    pli = mri + np.array([100.0, -50.0, 25.0], dtype=np.float32)
    s_mri = fit_stats(mri)
    s_pli = fit_stats(pli)
    assert not np.allclose(s_mri.centroid, s_pli.centroid)
