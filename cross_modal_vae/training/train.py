"""Training entry point: assembles dataset, model, optimiser, runs epochs."""

from __future__ import annotations

import argparse
import json
import random
from pathlib import Path

import numpy as np
import torch

from cross_modal_vae.data.bundles import PairedBundleDataset
from cross_modal_vae.data.normalize import apply, fit_stats, save_stats
from cross_modal_vae.data.streamline_io import load_streamlines
from cross_modal_vae.losses.schedules import beta_anneal
from cross_modal_vae.models.vae import CrossModalVAE

from .config import TrainConfig
from .losses import LossWeights, compute_losses
from .train_step import save_checkpoint


def _seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def _bundle_batches(
    dataset: PairedBundleDataset, batch_size: int, device: torch.device
):
    """Yield (mri, pli) torch batches of shape (batch_size, K, P, 3) each."""
    n = len(dataset)
    for start in range(0, n, batch_size):
        items = [dataset[i] for i in range(start, min(start + batch_size, n))]
        mri = torch.from_numpy(np.stack([it["mri"] for it in items])).to(device)
        pli = torch.from_numpy(np.stack([it["pli"] for it in items])).to(device)
        yield mri, pli


def train_from_arrays(
    mri: np.ndarray, pli: np.ndarray, cfg: TrainConfig
) -> list[dict[str, float]]:
    _seed_everything(cfg.seed)
    out_dir = Path(cfg.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    # Per-modality stats, applied before training. PLI and MRI live in
    # different coordinate frames -> stats MUST be independent.
    stats_mri = fit_stats(mri)
    stats_pli = fit_stats(pli)
    save_stats(stats_mri, out_dir / "stats_mri.npz")
    save_stats(stats_pli, out_dir / "stats_pli.npz")
    mri_n = apply(mri, stats_mri)
    pli_n = apply(pli, stats_pli)

    device = torch.device(cfg.device)
    vae = CrossModalVAE(
        K=cfg.K,
        n_points=cfg.n_points,
        d_model=cfg.d_model,
        latent_dim=cfg.latent_dim,
        n_heads=cfg.n_heads,
        n_layers_streamline=cfg.n_layers_streamline,
        n_layers_set=cfg.n_layers_set,
        n_layers_decoder=cfg.n_layers_decoder,
        dropout=cfg.dropout,
    ).to(device)
    optimizer = torch.optim.Adam(vae.parameters(), lr=cfg.lr)

    history: list[dict[str, float]] = []
    for epoch in range(cfg.epochs):
        dataset = PairedBundleDataset(
            mri=mri_n, pli=pli_n, K=cfg.K,
            bundles_per_epoch=cfg.bundles_per_epoch,
            seed=cfg.seed * 1000 + epoch,
        )
        beta = beta_anneal(
            epoch=epoch,
            warmup=cfg.beta_warmup_epochs,
            ramp=cfg.beta_ramp_epochs,
            target=cfg.beta_target,
        )
        weights = LossWeights(
            beta=beta,
            lambda_cross=cfg.lambda_cross,
            lambda_align=cfg.lambda_align,
        )

        epoch_sums = {k: 0.0 for k in ("total", "recon", "kl", "cross", "align")}
        n_batches = 0
        vae.train()
        for mri_batch, pli_batch in _bundle_batches(dataset, cfg.batch_size, device):
            optimizer.zero_grad(set_to_none=True)
            out = compute_losses(vae, mri_batch, pli_batch, weights)
            out["total"].backward()
            optimizer.step()
            for k in epoch_sums:
                epoch_sums[k] += out[k].detach().item()
            n_batches += 1

        log = {"epoch": epoch, "beta": beta}
        for k, v in epoch_sums.items():
            log[k] = v / max(n_batches, 1)
        history.append(log)
        print(json.dumps(log))

    save_checkpoint(vae, out_dir / "final.pt")
    with (out_dir / "history.json").open("w") as f:
        json.dump(history, f, indent=2)
    return history


def main() -> None:
    parser = argparse.ArgumentParser(description="Cross-modal VAE training")
    parser.add_argument("--mri", required=True, help="path to MRI *_256pts.npz")
    parser.add_argument("--pli", required=True, help="path to PLI *_256pts.npz")
    parser.add_argument("--out-dir", default="cross_modal_vae/runs/default")
    parser.add_argument("--epochs", type=int, default=200)
    parser.add_argument("--bundles-per-epoch", type=int, default=256)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--K", type=int, default=64)
    parser.add_argument("--n-points", type=int, default=256)
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--latent-dim", type=int, default=256)
    parser.add_argument("--d-model", type=int, default=128)
    parser.add_argument("--beta-warmup", type=int, default=50)
    parser.add_argument("--beta-ramp", type=int, default=50)
    parser.add_argument("--beta-target", type=float, default=1.0)
    parser.add_argument("--lambda-cross", type=float, default=1.0)
    parser.add_argument("--lambda-align", type=float, default=0.1)
    args = parser.parse_args()

    cfg = TrainConfig(
        K=args.K,
        n_points=args.n_points,
        d_model=args.d_model,
        latent_dim=args.latent_dim,
        epochs=args.epochs,
        bundles_per_epoch=args.bundles_per_epoch,
        batch_size=args.batch_size,
        lr=args.lr,
        beta_warmup_epochs=args.beta_warmup,
        beta_ramp_epochs=args.beta_ramp,
        beta_target=args.beta_target,
        lambda_cross=args.lambda_cross,
        lambda_align=args.lambda_align,
        seed=args.seed,
        device=args.device,
        out_dir=args.out_dir,
    )

    mri = load_streamlines(args.mri)
    pli = load_streamlines(args.pli)
    train_from_arrays(mri=mri, pli=pli, cfg=cfg)


if __name__ == "__main__":
    main()
