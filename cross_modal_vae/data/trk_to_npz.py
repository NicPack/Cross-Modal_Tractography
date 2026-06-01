"""Convert a .trk tractography file to a (N, n_points, 3) float32 .npz.

Mirrors the arclength resampling in the top-level interpolate_streamlines.py
so both MRI and PLI .trk inputs produce the exact array format
streamline_io.load_streamlines expects.
"""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import nibabel as nib


def resample_streamline(streamline: np.ndarray, n_points: int) -> np.ndarray:
    """Resample a variable-length streamline to `n_points` arclength-uniform points."""
    if streamline.ndim != 2 or streamline.shape[1] != 3:
        raise ValueError(
            f"expected streamline of shape (M, 3), got {streamline.shape}"
        )
    if streamline.shape[0] < 2:
        raise ValueError(
            f"streamline has {streamline.shape[0]} points; need at least 2"
        )
    seg = np.sqrt(np.sum(np.diff(streamline, axis=0) ** 2, axis=1))
    cumulative = np.insert(np.cumsum(seg), 0, 0.0)
    t = np.linspace(0.0, cumulative[-1], n_points)
    x = np.interp(t, cumulative, streamline[:, 0])
    y = np.interp(t, cumulative, streamline[:, 1])
    z = np.interp(t, cumulative, streamline[:, 2])
    return np.stack([x, y, z], axis=1).astype(np.float32)


def trk_to_npz(
    trk_path: Path | str, out_path: Path | str, n_points: int = 256
) -> np.ndarray:
    """Load .trk (RASMM), resample each streamline, save (N, n_points, 3) .npz."""
    tractogram_file = nib.streamlines.load(str(trk_path))
    streamlines = tractogram_file.streamlines
    resampled = np.stack(
        [resample_streamline(np.asarray(s, dtype=np.float32), n_points) for s in streamlines],
        axis=0,
    ).astype(np.float32)
    np.savez(str(out_path), resampled)
    return resampled


def main() -> None:
    parser = argparse.ArgumentParser(description="Resample .trk → (N, n_points, 3) .npz")
    parser.add_argument("--trk", required=True, help="input .trk path")
    parser.add_argument("--out", required=True, help="output .npz path")
    parser.add_argument("--n-points", type=int, default=256)
    args = parser.parse_args()
    arr = trk_to_npz(args.trk, args.out, n_points=args.n_points)
    print(f"wrote {args.out}: shape={arr.shape} dtype={arr.dtype}")


if __name__ == "__main__":
    main()
