"""Inspect MRI / PLI 256-pt input arrays for data-quality issues.

Reports the following diagnostics: NaN/Inf,
streamline length distribution, degenerate (point-like / sub-threshold-length)
streamlines, arclength-resampling evenness, planar-z detection, and per-streamline
diversity. Run from the project root, e.g.:

  uv run python scripts/inspect_inputs.py \
      MRI=runs/exp1/mri_256pts.npz PLI=runs/exp1/pli_256pts.npz
"""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np

from cross_modal_vaee.data.streamline_io import load_streamlines

DEFAULTS = [("MRI", "runs/exp1/mri_256pts.npz"), ("PLI", "runs/exp1/pli_256pts.npz")]


def inspect(name: str, path: str) -> None:
    sl = load_streamlines(path, mmap=False)  # (N, P, 3)
    N, P, _ = sl.shape
    print(f"\n===== {name}: {path} =====")
    print(f"  shape {sl.shape}  dtype {sl.dtype}")
    print(f"  NaN={np.isnan(sl).any()}  Inf={np.isinf(sl).any()}")

    seg = np.linalg.norm(np.diff(sl, axis=1), axis=-1)  # (N, P-1)
    length = seg.sum(axis=1)
    chord = np.linalg.norm(sl[:, -1] - sl[:, 0], axis=-1)
    print(f"  length: mean={length.mean():.3f} std={length.std():.3f} "
          f"min={length.min():.4f} max={length.max():.3f}")
    print(f"  chord/length (smoothness): "
          f"{np.mean(chord/np.maximum(length,1e-6)):.3f}  (1.0=straight)")

    n_zero = int((length < 1e-3).sum())
    n_tiny = int((length < 0.5).sum())
    print(f"  degenerate: length<1e-3 -> {n_zero} ({100*n_zero/N:.1f}%), "
          f"length<0.5 -> {n_tiny} ({100*n_tiny/N:.1f}%)")

    with np.errstate(divide="ignore", invalid="ignore"):
        seg_med = np.median(seg, axis=1)
        evenness = seg.max(axis=1) / np.where(seg_med > 0, seg_med, np.nan)
    print(f"  segment evenness (max/median per streamline): "
          f"median={np.nanmedian(evenness):.2f} (1.0=even arclength)")

    radius = np.sqrt(((sl - sl.mean(axis=1, keepdims=True)) ** 2).sum(-1)).max(axis=1)
    n_pointlike = int((radius < 0.1).sum())
    print(f"  point-like (radius<0.1) -> {n_pointlike} ({100*n_pointlike/N:.1f}%)")

    pts = sl.reshape(-1, 3)
    print(f"  coord min {pts.min(0).round(3)}  max {pts.max(0).round(3)}")
    print(f"  per-axis std {pts.std(0).round(3)}")

    # planar-z detection: is every streamline flat in z?
    z_range = (sl[:, :, 2].max(1) - sl[:, :, 2].min(1))
    print(f"  per-streamline z-range: mean={z_range.mean():.4f} max={z_range.max():.4f} "
          f"({'PLANAR' if z_range.max() < 1e-3 else 'true 3-D'})")

    keys = np.concatenate([sl[:, 0], sl[:, -1]], axis=1).round(3)
    uniq = np.unique(keys, axis=0).shape[0]
    print(f"  unique (start,end) endpoints: {uniq}/{N} ({100*uniq/N:.1f}% distinct)")

    cent = sl.mean(axis=1)
    spread = np.sqrt(((cent - cent.mean(0)) ** 2).sum(-1))
    print(f"  centroid spread across streamlines: mean={spread.mean():.3f} "
          f"std={spread.std():.3f}")


def main() -> None:
    # Accept NAME=path tokens on argv; fall back to the Sample1 defaults.
    items = DEFAULTS
    if len(sys.argv) > 1:
        items = []
        for tok in sys.argv[1:]:
            name, _, path = tok.partition("=")
            items.append((name or Path(path).stem, path or name))
    for name, path in items:
        inspect(name, path)


if __name__ == "__main__":
    main()
