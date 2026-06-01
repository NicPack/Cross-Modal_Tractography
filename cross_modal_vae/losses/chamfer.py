"""Symmetric point-cloud Chamfer distance between two bundles.

Each bundle is treated as an unordered set of K*P 3-D points; the symmetric
Chamfer is the sum of mean-nearest-neighbour squared-L2 distances in both
directions. Permutation-invariant in both K (streamline order) and P
(point order along a streamline), which is what the supervisor's brief
asks for.
"""

from __future__ import annotations

import torch


def _flatten_points(bundle: torch.Tensor) -> torch.Tensor:
    """(B, K, P, 3) -> (B, K*P, 3)."""
    B, K, P, _ = bundle.shape
    return bundle.reshape(B, K * P, 3)


def _one_sided_chamfer(src: torch.Tensor, ref: torch.Tensor) -> torch.Tensor:
    """For each point in `src`, find min squared distance to any point in `ref`.

    src, ref: (B, M, 3) and (B, N, 3)
    returns: (B,) — mean min distance across the M points in src
    """
    # squared pairwise distances: (B, M, N)
    diff = src.unsqueeze(2) - ref.unsqueeze(1)
    sq = (diff * diff).sum(dim=-1)
    min_sq, _ = sq.min(dim=-1)
    return min_sq.mean(dim=-1)


def chamfer_distance(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    """Symmetric Chamfer distance between two bundles.

    a, b: (B, K, P, 3)
    returns: scalar — mean over the batch of the symmetric Chamfer.
    """
    a_pts = _flatten_points(a)
    b_pts = _flatten_points(b)
    forward = _one_sided_chamfer(a_pts, b_pts)
    backward = _one_sided_chamfer(b_pts, a_pts)
    return (forward + backward).mean()
