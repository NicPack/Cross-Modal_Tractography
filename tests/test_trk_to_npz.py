"""Tests for cross_modal_vae.data.trk_to_npz — .trk → (N, n_points, 3) .npz.

Written test-first. Mirrors interpolate_streamlines.py's arclength resampling
so MRI (already-processed) and PLI .trk inputs go through the same pipeline.
"""
from __future__ import annotations

import subprocess
import sys

import numpy as np
import nibabel as nib
import pytest

from cross_modal_vae.data.trk_to_npz import resample_streamline, trk_to_npz
from cross_modal_vae.data.streamline_io import load_streamlines


def _save_trk(path, streamlines):
    """Write a .trk file with the given variable-length streamlines (RASMM)."""
    tractogram = nib.streamlines.Tractogram(
        streamlines, affine_to_rasmm=np.eye(4)
    )
    nib.streamlines.save(tractogram, str(path))


def test_resample_streamline_preserves_endpoints():
    line = np.array(
        [[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [2.0, 1.0, 0.0]], dtype=np.float32
    )
    out = resample_streamline(line, 10)
    np.testing.assert_allclose(out[0], line[0], atol=1e-5)
    np.testing.assert_allclose(out[-1], line[-1], atol=1e-5)


def test_resample_streamline_output_shape_and_dtype():
    line = np.array(
        [[0.0, 0.0, 0.0], [1.0, 2.0, 3.0], [4.0, 5.0, 6.0]], dtype=np.float32
    )
    out = resample_streamline(line, 256)
    assert out.shape == (256, 3)
    assert out.dtype == np.float32


def test_resample_streamline_uniform_arclength():
    # Straight line: equal-arc-length resampling means equidistant points.
    line = np.array([[0.0, 0.0, 0.0], [10.0, 0.0, 0.0]], dtype=np.float32)
    out = resample_streamline(line, 11)
    diffs = np.diff(out, axis=0)
    seg_lens = np.linalg.norm(diffs, axis=1)
    assert seg_lens.std() < 1e-5
    np.testing.assert_allclose(seg_lens.mean(), 1.0, atol=1e-5)


def test_resample_streamline_rejects_too_short():
    one_point = np.array([[0.0, 0.0, 0.0]], dtype=np.float32)
    with pytest.raises(ValueError):
        resample_streamline(one_point, 10)


def test_trk_to_npz_roundtrip_via_dipy(tmp_path):
    rng = np.random.default_rng(0)
    streamlines = [
        rng.standard_normal((M, 3)).astype(np.float32) for M in (12, 30, 7)
    ]
    trk_path = tmp_path / "in.trk"
    out_path = tmp_path / "out_256pts.npz"
    _save_trk(trk_path, streamlines)

    arr = trk_to_npz(trk_path, out_path, n_points=256)

    assert arr.shape == (3, 256, 3)
    assert arr.dtype == np.float32
    # File round-trips through streamline_io.load_streamlines.
    loaded = load_streamlines(out_path, mmap=False)
    np.testing.assert_array_equal(arr, loaded)


def test_trk_to_npz_cli_writes_npz(tmp_path):
    rng = np.random.default_rng(1)
    streamlines = [
        rng.standard_normal((M, 3)).astype(np.float32) for M in (10, 20)
    ]
    trk_path = tmp_path / "in.trk"
    out_path = tmp_path / "out.npz"
    _save_trk(trk_path, streamlines)

    result = subprocess.run(
        [
            sys.executable,
            "-m",
            "cross_modal_vae.data.trk_to_npz",
            "--trk",
            str(trk_path),
            "--out",
            str(out_path),
            "--n-points",
            "64",
        ],
        capture_output=True,
        text=True,
    )
    assert result.returncode == 0, result.stderr
    assert out_path.exists()
    loaded = load_streamlines(out_path, mmap=False)
    assert loaded.shape == (2, 64, 3)
