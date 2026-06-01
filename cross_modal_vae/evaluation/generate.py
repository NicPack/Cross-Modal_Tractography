"""Cross-modal inference: MRI streamlines → PLI-like .trk.

Pipeline:
  1. Load MRI streamlines (N, P, 3) and per-modality stats saved at training.
  2. Normalize MRI with stats_mri, batch into bundles of K.
  3. model.cross_forward(..., source="mri") — MRI encoder + shared decoder.
  4. Denormalize output with stats_pli; write a .trk in PLI's RASMM frame.

No registration exists, so the output .trk uses identity affine; the
coordinates live in the same RASMM frame as the PLI .trk that produced
stats_pli at training time.
"""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import nibabel as nib
import torch

from cross_modal_vae.data.normalize import (
    NormalizationStats,
    apply,
    invert,
    load_stats,
)
from cross_modal_vae.data.streamline_io import load_streamlines
from cross_modal_vae.models.vae import CrossModalVAE
from cross_modal_vae.training.config import TrainConfig
from cross_modal_vae.training.train_step import load_checkpoint


def build_model_from_checkpoint(
    checkpoint_path: Path | str,
    cfg: TrainConfig,
    device: torch.device | str = "cpu",
) -> CrossModalVAE:
    model = CrossModalVAE(
        K=cfg.K,
        n_points=cfg.n_points,
        d_model=cfg.d_model,
        latent_dim=cfg.latent_dim,
        n_heads=cfg.n_heads,
        n_layers_streamline=cfg.n_layers_streamline,
        n_layers_set=cfg.n_layers_set,
        n_layers_decoder=cfg.n_layers_decoder,
        dropout=cfg.dropout,
    )
    load_checkpoint(model, checkpoint_path)
    model.to(device)
    model.eval()
    return model


@torch.no_grad()
def generate_pli_from_mri(
    model: CrossModalVAE,
    mri_streamlines: np.ndarray,
    stats_mri: NormalizationStats,
    stats_pli: NormalizationStats,
    K: int,
    device: torch.device | str = "cpu",
) -> np.ndarray:
    """Generate PLI-like streamlines from raw MRI streamlines.

    Returns (n_bundles * K, P, 3) float32 in PLI's denormalized RASMM frame.
    If N is not divisible by K, the trailing (N % K) streamlines are dropped.
    """
    if mri_streamlines.ndim != 3 or mri_streamlines.shape[-1] != 3:
        raise ValueError(
            f"expected (N, P, 3) MRI streamlines, got {mri_streamlines.shape}"
        )
    if model.training:
        raise RuntimeError("model must be in eval mode for deterministic generation")

    n_total = mri_streamlines.shape[0]
    n_bundles = n_total // K
    n_used = n_bundles * K
    dropped = n_total - n_used
    if dropped:
        print(f"generate_pli_from_mri: dropping {dropped} trailing streamlines (N={n_total}, K={K})")

    normalized = apply(mri_streamlines[:n_used], stats_mri)        # (n_used, P, 3)
    P = normalized.shape[1]
    bundles = normalized.reshape(n_bundles, K, P, 3)               # (B, K, P, 3)
    x = torch.from_numpy(bundles).to(device=device, dtype=torch.float32)

    out = model.cross_forward(x, source="mri")                      # (B, K, P, 3)
    out_np = out.detach().cpu().numpy().reshape(n_used, P, 3)
    return invert(out_np, stats_pli).astype(np.float32)


def save_streamlines_trk(
    streamlines: np.ndarray,
    out_path: Path | str,
    affine: np.ndarray | None = None,
) -> None:
    """Write (N, P, 3) streamlines to a .trk file. Defaults to identity affine."""
    if affine is None:
        affine = np.eye(4)
    streamlines_list = [np.asarray(s, dtype=np.float32) for s in streamlines]
    tractogram = nib.streamlines.Tractogram(
        streamlines_list, affine_to_rasmm=affine
    )
    nib.streamlines.save(tractogram, str(out_path))


def main() -> None:
    parser = argparse.ArgumentParser(description="MRI → PLI-like .trk generation")
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--mri", required=True, help="path to MRI *_256pts.npz")
    parser.add_argument("--stats-dir", required=True,
                        help="dir containing stats_mri.npz and stats_pli.npz")
    parser.add_argument("--out", required=True, help="output .trk path")
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--K", type=int, default=64)
    parser.add_argument("--n-points", type=int, default=256)
    parser.add_argument("--latent-dim", type=int, default=256)
    parser.add_argument("--d-model", type=int, default=128)
    parser.add_argument("--n-heads", type=int, default=8)
    parser.add_argument("--n-layers-streamline", type=int, default=4)
    parser.add_argument("--n-layers-set", type=int, default=2)
    parser.add_argument("--n-layers-decoder", type=int, default=4)
    parser.add_argument("--dropout", type=float, default=0.0)
    args = parser.parse_args()

    cfg = TrainConfig(
        K=args.K,
        n_points=args.n_points,
        d_model=args.d_model,
        latent_dim=args.latent_dim,
        n_heads=args.n_heads,
        n_layers_streamline=args.n_layers_streamline,
        n_layers_set=args.n_layers_set,
        n_layers_decoder=args.n_layers_decoder,
        dropout=args.dropout,
        device=args.device,
    )

    stats_dir = Path(args.stats_dir)
    stats_mri = load_stats(stats_dir / "stats_mri.npz")
    stats_pli = load_stats(stats_dir / "stats_pli.npz")

    model = build_model_from_checkpoint(args.checkpoint, cfg, device=args.device)
    mri = load_streamlines(args.mri, mmap=False)
    generated = generate_pli_from_mri(
        model, mri, stats_mri, stats_pli, K=cfg.K, device=args.device
    )
    save_streamlines_trk(generated, args.out)
    print(f"wrote {args.out}: {generated.shape[0]} streamlines × {generated.shape[1]} points")


if __name__ == "__main__":
    main()
