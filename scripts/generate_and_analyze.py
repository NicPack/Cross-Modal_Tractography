"""Chunked MRI->PLI generation from a trained checkpoint + geometry analysis.

`cross_forward` over all bundles at once OOMs (B=276, 256-pt self-attention over
~17k streamlines, even on an A100), so we chunk over bundles. We then compare the
generated geometry against the real PLI training streamlines and the input MRI,
reporting the following diagnostics:
  - spatial envelope (bbox / coord range) vs real PLI and MRI
  - streamline smoothness via chord/length ratio (scribble detector)
  - posterior-collapse signals: across-bundle shape std, diversity ratio

Run from the project root, e.g.:
  uv run python scripts/generate_and_analyze.py \
      --checkpoint runs/exp2/run/final.pt --stats-dir runs/exp2/run \
      --mri runs/exp1/mri_256pts.npz --pli runs/exp1/pli_256pts.npz \
      --out runs/exp2/generated_pli.trk
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

# Make the package importable when this file is run directly (sys.path[0] is the
# script dir, not the project root) regardless of the current working directory.
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np
import torch

from cross_modal_vaee.data.normalize import apply, invert, load_stats
from cross_modal_vaee.data.streamline_io import load_streamlines
from cross_modal_vaee.evaluation.generate import (
    build_model_from_checkpoint,
    save_streamlines_trk,
)
from cross_modal_vaee.training.config import TrainConfig


def describe(name: str, sl: np.ndarray) -> np.ndarray:
    pts = sl.reshape(-1, 3)
    bbox = pts.max(0) - pts.min(0)
    seg = np.linalg.norm(np.diff(sl, axis=1), axis=-1).sum(axis=1)  # path length
    chord = np.linalg.norm(sl[:, -1] - sl[:, 0], axis=-1)           # end-to-end
    print(f"\n[{name}] n={sl.shape[0]}")
    print(f"  bbox extent (x,y,z): {bbox.round(2)}")
    print(f"  coord range: min {pts.min(0).round(1)}  max {pts.max(0).round(1)}")
    print(f"  length  mean={seg.mean():.2f} std={seg.std():.2f} "
          f"min={seg.min():.2f} max={seg.max():.2f}")
    print(f"  chord   mean={chord.mean():.2f} "
          f"(chord/length={np.mean(chord/np.maximum(seg,1e-6)):.3f})")
    return seg


def rms_to_mean(sl: np.ndarray) -> np.ndarray:
    """Per-streamline RMS distance to its own centroid (shape spread proxy)."""
    m = sl.mean(axis=0)
    return np.sqrt(((sl - m) ** 2).sum(-1).mean(axis=1))


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--checkpoint", default="runs/exp2/run/final.pt")
    p.add_argument("--stats-dir", default="runs/exp2/run")
    p.add_argument("--mri", default="runs/exp1/mri_256pts.npz")
    p.add_argument("--pli", default="runs/exp1/pli_256pts.npz")
    p.add_argument("--out", default="runs/exp2/generated_pli.trk")
    p.add_argument("--K", type=int, default=64)
    p.add_argument("--chunk", type=int, default=4, help="bundles per forward pass")
    p.add_argument("--device", default="cpu")
    args = p.parse_args()

    cfg = TrainConfig(K=args.K, device=args.device)  # other defaults match training
    model = build_model_from_checkpoint(args.checkpoint, cfg, device=args.device)

    stats_dir = Path(args.stats_dir)
    stats_mri = load_stats(stats_dir / "stats_mri.npz")
    stats_pli = load_stats(stats_dir / "stats_pli.npz")

    mri = load_streamlines(args.mri, mmap=False)
    pli = load_streamlines(args.pli, mmap=False)

    K = args.K
    n_bundles = mri.shape[0] // K
    n_used = n_bundles * K
    P = mri.shape[1]
    norm = apply(mri[:n_used], stats_mri).reshape(n_bundles, K, P, 3)

    outs = []
    with torch.no_grad():
        for s in range(0, n_bundles, args.chunk):
            x = torch.from_numpy(norm[s : s + args.chunk]).to(
                device=args.device, dtype=torch.float32
            )
            outs.append(model.cross_forward(x, source="mri").cpu().numpy())
    gen = np.concatenate(outs, axis=0).reshape(n_used, P, 3)
    gen = invert(gen, stats_pli).astype(np.float32)

    save_streamlines_trk(gen, args.out)
    print(f"wrote {args.out}: {gen.shape[0]} streamlines x {gen.shape[1]} pts")

    describe("MRI input", mri[:n_used])
    describe("real PLI", pli)
    describe("generated PLI", gen)

    print("\n=== diversity of generated streamlines ===")
    gen_spread = rms_to_mean(gen)
    pli_spread = rms_to_mean(pli)
    print(f"  generated: per-streamline RMS-to-mean mean={gen_spread.mean():.3f} "
          f"std={gen_spread.std():.3f}")
    print(f"  real PLI : per-streamline RMS-to-mean mean={pli_spread.mean():.3f} "
          f"std={pli_spread.std():.3f}")
    print(f"  -> diversity ratio gen/real = "
          f"{gen_spread.mean()/max(pli_spread.mean(),1e-9):.3f}")

    gb = gen.reshape(n_bundles, K, P, 3).mean(axis=1)  # per-bundle mean shape
    across_bundle = gb.std(axis=0)
    print(f"\n  across-bundle std of per-bundle mean shape: "
          f"mean={across_bundle.mean():.4f} max={across_bundle.max():.4f}")
    print("  (near 0 => every bundle decodes to the same shape; latent ignored)")
    print(f"\n  total point std  generated={gen.reshape(-1,3).std(0).round(3)}  "
          f"real PLI={pli.reshape(-1,3).std(0).round(3)}")


if __name__ == "__main__":
    main()
