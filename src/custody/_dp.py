"""Differential-privacy primitives CUSTODY builds on.

Renyi accountant, delta default and Gaussian sigma calibration. Extracted
from the wider privacy package so the framework carries only what it uses.
Cited, not claimed: Dwork & Roth, Mironov, Abadi et al.
"""

from __future__ import annotations

import math

__all__ = ["RDPAccountant", "calibrate_sigma", "default_delta"]


def default_delta(n: int) -> float:
    """``n^-1.1`` — comfortably below the ``1/n`` rule of thumb.

    The library previously defaulted to ``1e-5``, which at n = 25,609 is
    ``0.26/n``: the same order as the per-record probability the guarantee is
    supposed to bound, not below it.
    """
    if n <= 1:
        raise ValueError("n must exceed 1")
    return float(float(n) ** -1.1)


class RDPAccountant:
    """Renyi-DP accountant for composed Gaussian mechanisms (P-0021/P-0022).

    Tracks ``a_total = sum_i Delta_i^2 / (2 sigma_i^2)`` (the per-order slope of the
    Gaussian RDP curve), which composes by addition. Converts to ``(eps, delta)``
    with the closed-form minimisation over the Renyi order.
    """

    def __init__(self) -> None:
        self._a_total: float = 0.0

    def add_gaussian(self, *, sensitivity: float, sigma: float) -> None:
        if sigma <= 0 or sensitivity < 0:
            raise ValueError("sigma must be > 0 and sensitivity >= 0")
        self._a_total += (sensitivity**2) / (2.0 * sigma**2)

    @property
    def rdp_slope(self) -> float:
        return self._a_total

    def to_dp(self, delta: float) -> float:
        """Convert accumulated RDP to an ``(eps, delta)``-DP guarantee."""
        if not (0.0 < delta < 1.0):
            raise ValueError("delta must be in (0, 1)")
        a = self._a_total
        if a <= 0.0:
            return 0.0
        b = math.log(1.0 / delta)
        return a + 2.0 * math.sqrt(a * b)


def calibrate_sigma(*, epsilon: float, delta: float, sensitivity: float) -> float:
    """Smallest Gaussian noise sigma achieving ``(epsilon, delta)``-DP for one release.

    Inverts ``eps = a + 2 sqrt(a*ln(1/delta))`` with ``a = Delta^2/(2 sigma^2)``.
    Equivalently, with ``u = Delta/sigma`` and ``c = sqrt(2 ln(1/delta))``:
    ``eps = u^2/2 + u*c`` -> ``u = -c + sqrt(c^2 + 2 eps)`` -> ``sigma = Delta/u``.
    """
    if epsilon <= 0:
        raise ValueError("epsilon must be positive")
    if not (0.0 < delta < 1.0):
        raise ValueError("delta must be in (0, 1)")
    if sensitivity < 0:
        raise ValueError("sensitivity must be non-negative")
    if sensitivity == 0:
        return 0.0
    c = math.sqrt(2.0 * math.log(1.0 / delta))
    u = -c + math.sqrt(c**2 + 2.0 * epsilon)
    return sensitivity / u
