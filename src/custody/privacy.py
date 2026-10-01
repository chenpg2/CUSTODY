"""The privacy layer: an epsilon the accountant produces, not one the sender declares.

**What is released.** A private node keeps the exact sums of seven groups of its
capped, clipped cycles (:mod:`custody.private_stats`). Each release adds
Gaussian noise to them, calibrated with :func:`custody._dp.calibrate_sigma` to a
sensitivity that follows from public bounds, and derives every kernel field from
the noised sums alone. :class:`custody._dp.RDPAccountant` composes the
mechanisms, and the spent budget is what the certificate reports.

Releases 1.0.0 and 1.1.0 added Gaussian noise to kernels fitted by maximum
likelihood, with sensitivities that did not hold for what they released, so the
budget they certified understated the loss. This layer replaces that one.

**Accounting unit.** The unit is the FAMILY (one patient's whole trajectory,
and with it her partner's and any offspring's records), because that is the
unit the data actually has. A family contributes at most
``max_cycles_per_family`` cycles, its first in visit order (Amin et al., ICML
2019, cited not claimed). The guarantee is bounded DP: the number of families is
invariant under replacing one, and it is disclosed.

**What is NOT claimed.** Nothing here is a new mechanism. Gaussian noise on
bounded sums, RDP composition and contribution capping are textbook; the
engineering claim is that the exchange applies them, accounts for them across
releases, and lets a receiver check that a stated budget is substantiated.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any

import numpy as np
import pandas as pd

from ._dp import RDPAccountant, calibrate_sigma, default_delta
from .private_stats import (
    GROUPS,
    HISTOGRAM_GROUP,
    KERNEL_GROUPS,
    OOCYTE_MAX,
    POOL_SIZE,
    PrivateFitError,
    PrivateStatistics,
    group_sums,
    kernels_from_sums,
)
from .process import ProcessKernels

__all__ = [
    "BudgetExhausted",
    "DPConfig",
    "FamilyBudget",
    "PrivateFitError",
    "PrivateStatistics",
    "cap_contributions",
    "group_sensitivities",
    "private_statistics",
    "release_private_kernels",
    "zero_noise_kernels",
]


class BudgetExhausted(RuntimeError):
    """Raised when a node's cumulative spend would exceed its declared cap."""


def _check_cycle_bound(k: object, name: str) -> None:
    """A contribution bound is a whole number of cycles, at least one.

    A fractional bound kept more cycles than the sensitivity assumed (2.5 kept
    three and priced two), and zero left nothing to fit.
    """
    if isinstance(k, bool) or not isinstance(k, (int, np.integer)) or k < 1:
        raise ValueError(f"{name} must be a whole number of cycles, at least 1; got {k!r}")


@dataclass(frozen=True)
class DPConfig:
    """Per-release privacy parameters.

    ``bank_column`` names the column that records banked embryos. It is a public
    per-centre constant: choosing it from how often each column is filled, as the
    plain fitter does, would make it a function of the data. The default is the
    banked column :class:`custody.LedgerSchema` names.
    """

    epsilon: float = 1.0
    delta: float | None = None
    max_cycles_per_family: int = 6
    covariate_budget_share: float = 0.25  # share of epsilon spent on the covariate histogram
    bank_column: str = "freeze_num"

    def __post_init__(self) -> None:
        _check_cycle_bound(self.max_cycles_per_family, "max_cycles_per_family")
        if not self.epsilon > 0:
            raise ValueError(f"epsilon must be positive; got {self.epsilon!r}")
        if not 0.0 < self.covariate_budget_share < 1.0:
            raise ValueError("covariate_budget_share must lie strictly between 0 and 1")

    def resolved_delta(self, n_families: int) -> float:
        return self.delta if self.delta is not None else default_delta(max(n_families, 2))


@dataclass
class FamilyBudget:
    """Cumulative family-unit spend for one node, across every release it makes.

    Without it a node could emit ten payloads at epsilon 1 and no object in the
    system would say epsilon 10. This is that object.
    """

    cap_epsilon: float
    delta: float
    accountant: RDPAccountant = None  # type: ignore[assignment]
    releases: int = 0

    def __post_init__(self) -> None:
        # A cap of NaN compared false against every spend, so nothing was ever refused.
        if not (math.isfinite(self.cap_epsilon) and self.cap_epsilon > 0):
            raise ValueError(f"cap_epsilon must be a positive number; got {self.cap_epsilon!r}")
        if self.accountant is None:
            self.accountant = RDPAccountant()

    @property
    def spent(self) -> float:
        return float(self.accountant.to_dp(self.delta))

    def would_exceed(self, sensitivities: dict[str, float], sigmas: dict[str, float]) -> bool:
        probe = RDPAccountant()
        probe._a_total = self.accountant.rdp_slope  # noqa: SLF001 - deliberate probe copy
        for key, sensitivity in sensitivities.items():
            probe.add_gaussian(sensitivity=sensitivity, sigma=sigmas[key])
        return float(probe.to_dp(self.delta)) > self.cap_epsilon + 1e-12

    def charge(
        self,
        sensitivities: dict[str, float],
        sigmas: dict[str, float],
        *,
        new_release: bool = True,
    ) -> None:
        """Charge a composition step, or refuse it whole.

        A release is charged once, with every mechanism it will run, before any
        noise is drawn, so a refusal leaves the budget untouched. ``new_release``
        counts releases rather than charges; a release whose fit fails after its
        charge still counts, because its noise was drawn.
        """
        if self.would_exceed(sensitivities, sigmas):
            raise BudgetExhausted(
                f"release would take cumulative spend past the declared cap "
                f"{self.cap_epsilon} (spent {self.spent:.4f} over {self.releases} releases)"
            )
        for key, sensitivity in sensitivities.items():
            self.accountant.add_gaussian(sensitivity=sensitivity, sigma=sigmas[key])
        if new_release:
            self.releases += 1

    def as_dict(self) -> dict[str, object]:
        return {
            "accounting_unit": "family",
            "cap_epsilon": float(self.cap_epsilon),
            "spent_epsilon": self.spent,
            "delta": float(self.delta),
            "releases": int(self.releases),
        }


def cap_contributions(
    fresh: pd.DataFrame, fet: pd.DataFrame, max_cycles: int
) -> tuple[pd.DataFrame, pd.DataFrame, int]:
    """Keep each family's first ``max_cycles`` cycles. Finite sensitivity needs this.

    Cycles are ordered by ``visit_date``, with a fresh cycle before a frozen
    transfer on the same date, so every non-empty frame needs a visit date on
    every row: without one, "first" has no meaning and is refused rather than
    guessed from the order the rows happen to be stored in. Rows are selected
    from the frames as given, so a family within the cap is fitted on exactly
    the rows it had, with its columns and types untouched.

    Returns:
        The capped fresh and frozen-transfer frames, and the number of cycles
        dropped. The third value is a count of cycles, not of families.

    Raises:
        ValueError: if ``max_cycles`` is not a whole number of at least one, or a
            non-empty frame lacks a family id or a visit date on any row, or holds
            its dates as anything but datetimes.
    """
    _check_cycle_bound(max_cycles, "max_cycles")
    parts = []
    for kind, frame in ((0, fresh), (1, fet)):
        if not len(frame):
            continue
        # Dates read as text sort as text, which is not visit order, and a row
        # with no family id would be dropped from every family and counted as capped.
        if (
            "visit_date" not in frame.columns
            or not pd.api.types.is_datetime64_any_dtype(frame["visit_date"])
            or frame["visit_date"].isna().any()
        ):
            raise ValueError(
                "capping keeps each family's first cycles in visit order, so every "
                "non-empty frame needs visit_date as a datetime on every row"
            )
        if frame["pid"].isna().any():
            raise ValueError("capping counts cycles per family, so every row needs a pid")
        parts.append(frame[["pid", "visit_date"]].assign(_kind=kind, _row=np.arange(len(frame))))
    if not parts:
        return fresh, fet, 0
    keys = pd.concat(parts, ignore_index=True).sort_values(
        ["pid", "visit_date", "_kind", "_row"], kind="mergesort"
    )
    keep = (keys.groupby("pid").cumcount() < max_cycles).to_numpy()
    kept = keys[keep]
    fresh_rows = np.sort(kept.loc[kept["_kind"] == 0, "_row"].to_numpy())
    fet_rows = np.sort(kept.loc[kept["_kind"] == 1, "_row"].to_numpy())
    return fresh.iloc[fresh_rows], fet.iloc[fet_rows], int((~keep).sum())


def group_sensitivities(k: int) -> dict[str, float]:
    """Replace-one-family L2 sensitivity of each group's sum.

    One family's contribution to a group is a non-negative vector c with
    ``‖c‖₂ ≤ C`` and coordinates ``c_j ≤ b_j``; replacing it moves the sum by at
    most ``min(√2·C, ‖b‖₂)``, since ``‖c′ − c‖² ≤ ‖c‖² + ‖c′‖²`` when
    ``⟨c, c′⟩ ≥ 0`` and ``|c′_j − c_j| ≤ b_j``.
    """
    _check_cycle_bound(k, "k")
    e = OOCYTE_MAX
    return {
        "G1": k * e * math.sqrt(3.0),  # (ΣE, ΣF, ΣZ), each at most k·40
        "G2": 2.0 * k,  # four counts, each at most k
        "G3": k * math.sqrt(12.0),  # (count, clipped transfers ≤ 3k, count, count)
        # Cell totals sum to at most k; continued, chances and used are each at
        # most k − 1, since the last cycle does not continue and the first has
        # no bank.
        "G4": math.sqrt(2.0 * (k**2 + 3 * (k - 1) ** 2)),
        "G5_cells": 2.0 * e * k,  # √2 · √((40k)² + (40k)²), counts scaled by 40
        "G5_sq": k * e**2,  # ΣE² over at most k cycles of at most 40 oocytes
        "G6": 2.0 * k,  # √2 · √(k² + k²)
        "G7": math.sqrt(2.0),  # one baseline row per family
    }


def private_statistics(
    fresh: pd.DataFrame, fet: pd.DataFrame, config: DPConfig
) -> PrivateStatistics:
    """What a private node keeps: the exact group sums of its capped cycles."""
    if config.bank_column not in fresh.columns:
        raise ValueError(f"the banking column {config.bank_column!r} is not in the fresh cycles")
    capped_fresh, capped_fet, _dropped = cap_contributions(fresh, fet, config.max_cycles_per_family)
    families = pd.concat([capped_fresh["pid"], capped_fet["pid"]]).nunique()
    return PrivateStatistics(
        n_families=int(families),
        max_cycles_per_family=config.max_cycles_per_family,
        bank_column=config.bank_column,
        sums=group_sums(capped_fresh, capped_fet, config.bank_column),
    )


def _sigmas(config: DPConfig, delta: float, sens: dict[str, float]) -> dict[str, float]:
    """Each kernel mechanism at an equal share of three quarters; the histogram at a quarter."""
    share = config.epsilon * (1.0 - config.covariate_budget_share) / len(KERNEL_GROUPS)
    sigmas = {
        g: calibrate_sigma(epsilon=share, delta=delta, sensitivity=sens[g]) for g in KERNEL_GROUPS
    }
    sigmas[HISTOGRAM_GROUP] = calibrate_sigma(
        epsilon=config.epsilon * config.covariate_budget_share,
        delta=delta,
        sensitivity=sens[HISTOGRAM_GROUP],
    )
    return sigmas


def _check_compatible(stats: PrivateStatistics, config: DPConfig) -> None:
    if stats.max_cycles_per_family > config.max_cycles_per_family:
        raise ValueError(
            f"statistics capped at {stats.max_cycles_per_family} cycles per family; the noise "
            f"would be calibrated to {config.max_cycles_per_family}"
        )
    if stats.bank_column != config.bank_column:
        raise ValueError(
            f"statistics banked on {stats.bank_column!r}, the release on {config.bank_column!r}"
        )


def release_private_kernels(
    stats: PrivateStatistics,
    *,
    config: DPConfig,
    budget: FamilyBudget,
    rng: np.random.Generator,
) -> tuple[ProcessKernels, dict[str, object]]:
    """Noise the group sums once, charge the budget, and derive the kernels.

    Raises:
        BudgetExhausted: if the release would take the node past its cap. Nothing
            is drawn and nothing is charged.
        PrivateFitError: if a regression cannot be fitted to the noised sums. The
            release fails and its spend stands, because the noise was drawn.
        ValueError: if the statistics were computed under a different cap or
            banking column, or the budget's delta is not the one calibrated to.
    """
    _check_compatible(stats, config)
    delta = config.resolved_delta(stats.n_families)
    if not math.isclose(budget.delta, delta, rel_tol=1e-12):
        raise ValueError(f"budget delta {budget.delta} is not the calibrated delta {delta}")
    sens = group_sensitivities(config.max_cycles_per_family)
    sigmas = _sigmas(config, delta, sens)
    budget.charge(sens, sigmas)  # the whole release, before any draw
    noised = {
        g: stats.sums[g] + rng.normal(0.0, sigmas[g], size=stats.sums[g].shape) for g in GROUPS
    }
    kernels = kernels_from_sums(
        noised,
        histogram_threshold=2.0 * sigmas[HISTOGRAM_GROUP],
        rng=rng,
        contribution_cap=config.max_cycles_per_family,
    )
    record: dict[str, Any] = {
        "accounting": budget.as_dict(),
        "epsilon_this_release": float(config.epsilon),
        "delta": float(delta),
        "max_cycles_per_family": int(config.max_cycles_per_family),
        "n_families": int(stats.n_families),
        "bank_column": config.bank_column,
        "sensitivity_by_group": {g: float(v) for g, v in sens.items()},
        "sigma_by_group": {g: float(v) for g, v in sigmas.items()},
        "pool_size": POOL_SIZE,
        "mechanism": "Gaussian noise on clipped group sums, RDP composition",
    }
    return kernels, record


def zero_noise_kernels(stats: PrivateStatistics, config: DPConfig) -> ProcessKernels:
    """The private estimator with no noise. Never released; a reference.

    It separates what the private estimator changes (clipping, grids, a pool of
    patients) from what the noise changes. Its pool is drawn with a fixed,
    data-independent seed.
    """
    _check_compatible(stats, config)
    return kernels_from_sums(
        stats.sums,
        histogram_threshold=0.0,
        rng=np.random.default_rng(0),
        contribution_cap=config.max_cycles_per_family,
    )
