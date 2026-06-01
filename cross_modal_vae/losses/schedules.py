"""Loss-weight schedules used during training."""

from __future__ import annotations


def beta_anneal(*, epoch: int, warmup: int, ramp: int, target: float) -> float:
    """β-annealing for KL weight.

    For epoch < warmup: 0.
    For warmup <= epoch < warmup + ramp: linearly interpolated 0 -> target.
    For epoch >= warmup + ramp: target.

    With ramp == 0 the schedule is a step function: 0 below warmup,
    `target` at and above.
    """
    if epoch < warmup:
        return 0.0
    if ramp <= 0 or epoch >= warmup + ramp:
        return float(target)
    t = (epoch - warmup) / ramp
    return float(target * t)
