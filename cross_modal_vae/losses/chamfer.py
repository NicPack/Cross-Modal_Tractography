"""Symmetric point-cloud Chamfer distance between two bundles.

Each bundle is treated as an unordered set of K*P 3-D points; the symmetric
Chamfer is the sum of mean-nearest-neighbour squared-L2 distances in both
directions. Permutation-invariant in both K (streamline order) and P
(point order along a streamline), which is what the supervisor's brief
asks for.
"""

from __future__ import annotations

import torch
from torch.utils.checkpoint import checkpoint

# Points of `src` processed at a time in _one_sided_chamfer. The full pairwise
# diff tensor is (B, M, N, 3); for bundle-sized inputs (M = N = K*P = 16384) the
# whole thing is tens of GiB and OOMs the GPU. Chunking `src` bounds the peak to
# (B, _SRC_CHUNK, N, 3); gradient checkpointing recomputes each chunk in the
# backward pass instead of keeping it live. The arithmetic is unchanged, so the
# result is bit-identical to materialising the full tensor at once.
_SRC_CHUNK = 1024


def _flatten_points(bundle: torch.Tensor) -> torch.Tensor:
    """(B, K, P, 3) -> (B, K*P, 3)."""
    B, K, P, _ = bundle.shape
    return bundle.reshape(B, K * P, 3)


def _chunk_min_sq(src_chunk: torch.Tensor, ref: torch.Tensor) -> torch.Tensor:
    """Min squared distance from each point in a src chunk to any ref point.

    src_chunk, ref: (B, C, 3) and (B, N, 3) -> (B, C)
    """
    diff = src_chunk.unsqueeze(2) - ref.unsqueeze(1)  # (B, C, N, 3)
    sq = (diff * diff).sum(dim=-1)                    # (B, C, N)
    return sq.min(dim=-1).values                      # (B, C)


def _one_sided_chamfer(src: torch.Tensor, ref: torch.Tensor) -> torch.Tensor:
    """For each point in `src`, find min squared distance to any point in `ref`.

    src, ref: (B, M, 3) and (B, N, 3)
    returns: (B,) — mean min distance across the M points in src
    """
    M = src.shape[1]
    use_ckpt = torch.is_grad_enabled() and (src.requires_grad or ref.requires_grad)
    chunks = []
    for start in range(0, M, _SRC_CHUNK):
        src_chunk = src[:, start : start + _SRC_CHUNK]
        if use_ckpt:
            min_sq = checkpoint(_chunk_min_sq, src_chunk, ref, use_reentrant=False)
        else:
            min_sq = _chunk_min_sq(src_chunk, ref)
        chunks.append(min_sq)
    return torch.cat(chunks, dim=1).mean(dim=-1)


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
