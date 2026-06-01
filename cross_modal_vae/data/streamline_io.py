"""I/O helpers for the *_256pts.npz streamline arrays produced by
`interpolate_streamlines.py`. The arrays are stored either as a single
unnamed array (np.savez(path, arr)) or under a named key. This loader
handles both layouts transparently.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np


def load_streamlines(path: Path | str, mmap: bool = True) -> np.ndarray:
    """Load (N, P, 3) float32 streamlines from .npz.

    Uses memory-mapped reads when mmap=True so the PLI tensor (~335 MB)
    does not have to be loaded fully into RAM.
    """
    data = np.load(path, mmap_mode="r" if mmap else None)
    # np.savez with a positional arg stores under "arr_0"; np.savez with a
    # keyword stores under that name. Take whichever 3-D array we find.
    for key in data.files:
        arr = data[key]
        if arr.ndim == 3 and arr.shape[-1] == 3:
            return arr.astype(np.float32, copy=False)
    raise ValueError(f"no (N, P, 3) array found in {path}; keys={data.files}")
