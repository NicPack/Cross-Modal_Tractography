"""Per-modality coordinate normalization for streamline arrays.

MRI and PLI live in different physical frames (no shared registration),
so each modality fits and applies its own stats. We normalize each axis
to [-1, 1] by subtracting the per-axis centroid and dividing by per-axis
max-abs of the centred values.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np


@dataclass(frozen=True)
class NormalizationStats:
    centroid: np.ndarray  # shape (3,)
    scale: np.ndarray     # shape (3,)


def fit_stats(streamlines: np.ndarray) -> NormalizationStats:
    if streamlines.ndim != 3 or streamlines.shape[-1] != 3:
        raise ValueError(
            f"expected array of shape (N, P, 3), got {streamlines.shape}"
        )
    points = streamlines.reshape(-1, 3)
    centroid = points.mean(axis=0).astype(np.float32)
    scale = np.abs(points - centroid).max(axis=0).astype(np.float32)
    # guard against zero-variance axes
    scale = np.where(scale > 0, scale, np.float32(1.0))
    return NormalizationStats(centroid=centroid, scale=scale)


def apply(streamlines: np.ndarray, stats: NormalizationStats) -> np.ndarray:
    return ((streamlines - stats.centroid) / stats.scale).astype(streamlines.dtype)


def invert(normalized: np.ndarray, stats: NormalizationStats) -> np.ndarray:
    return (normalized * stats.scale + stats.centroid).astype(normalized.dtype)


def save_stats(stats: NormalizationStats, path: Path | str) -> None:
    np.savez(path, centroid=stats.centroid, scale=stats.scale)


def load_stats(path: Path | str) -> NormalizationStats:
    data = np.load(path)
    return NormalizationStats(centroid=data["centroid"], scale=data["scale"])
