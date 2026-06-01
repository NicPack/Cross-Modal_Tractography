"""Mini-bundle sampling for cross-modal VAE training.

A bundle is K streamlines drawn without replacement from one modality's
full streamline array. MRI and PLI are sampled independently because
N != M and there is no correspondence between individual streamlines.
"""

from __future__ import annotations

import numpy as np


def sample_bundle(rng: np.random.Generator, streamlines: np.ndarray, K: int) -> np.ndarray:
    """Draw K streamlines without replacement from `streamlines`.

    streamlines: array of shape (N, P, 3)
    returns: array of shape (K, P, 3)
    """
    n = streamlines.shape[0]
    if K > n:
        raise ValueError(f"K={K} exceeds available streamlines N={n}")
    idx = rng.choice(n, size=K, replace=False)
    return streamlines[idx]


class PairedBundleDataset:
    """Map-style dataset yielding {'mri': (K,P,3), 'pli': (K,P,3)} dicts.

    The two modalities are sampled independently per item — there is no
    streamline-level pairing. Items are reproducible: given a fixed seed,
    item i always returns the same pair of bundles across instances.
    """

    def __init__(
        self,
        mri: np.ndarray,
        pli: np.ndarray,
        K: int,
        bundles_per_epoch: int,
        seed: int = 0,
    ) -> None:
        self.mri = mri
        self.pli = pli
        self.K = K
        self.bundles_per_epoch = bundles_per_epoch
        self.seed = seed

    def __len__(self) -> int:
        return self.bundles_per_epoch

    def __getitem__(self, index: int) -> dict[str, np.ndarray]:
        if not 0 <= index < self.bundles_per_epoch:
            raise IndexError(index)
        # Derive a per-item seed so __getitem__ is order-independent and reproducible.
        mri_rng = np.random.default_rng((self.seed, index, 0))
        pli_rng = np.random.default_rng((self.seed, index, 1))
        return {
            "mri": sample_bundle(mri_rng, self.mri, self.K),
            "pli": sample_bundle(pli_rng, self.pli, self.K),
        }
