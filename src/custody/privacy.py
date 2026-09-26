"""The DP layer the exchange was missing — ε that is produced, not declared.

The V3/V4 pre-compute audit (2026-08-28) found that kernels crossed the
institutional boundary as plain maximum-likelihood fits while the certificate
carried a hardcoded ``epsilon`` literal. That is a decorative guarantee, and
a paper claiming it would be claiming a mechanism the pipeline did not have.

This module supplies the mechanism. Each released kernel parameter is a
bounded-sensitivity statistic; Gaussian noise is calibrated to the family-unit
budget with :func:`custody._dp.calibrate_sigma`, composition is tracked by
:class:`custody._dp.RDPAccountant`, and the spent budget is what the
certificate reports.

**Accounting unit.** The unit is the FAMILY (one patient's whole trajectory,
and with it her partner's and any offspring's records), because that is the
unit the data actually has — the role-separated alternative was analysed and
shown unattainable (`plan/art_role_separation_theorem.md`). A family
contributes at most ``max_cycles_per_family`` cycles; contributions beyond the
cap are dropped, which is what makes sensitivity finite (Amin et al., ICML
2019, cited not claimed).

**What is NOT claimed.** Nothing here is a new mechanism. Gaussian output
perturbation, RDP composition, and contribution capping are textbook; the
engineering claim is that the exchange applies them, accounts for them across
releases, and lets a receiver check the arithmetic.
"""

from __future__ import annotations

from dataclasses import dataclass, replace
from typing import Any

import numpy as np
import pandas as pd

from ._dp import RDPAccountant, calibrate_sigma, default_delta
from .process import ProcessKernels

__all__ = [
    "BudgetExhausted",
    "DPConfig",
    "FamilyBudget",
    "privatise_kernels",
]

# Released rates are means of per-family bounded quantities. Capping each
# family at K cycles bounds one family's influence on a rate by K/n, and on a
# log-scale coefficient by the range the fitter is allowed to move.
RATE_FIELDS = (
    "fert_rate",
    "dev_rate",
    "p_bank_given_surplus",
    "p_continue_fail_nobank",
    "p_continue_fail_bank",
    "p_continue_birth_nobank",
    "p_continue_birth_bank",
    "p_use_bank",
    "p_fresh_transfer",
    "fet_live_birth_rate",
)
VECTOR_FIELDS = ("yield_beta", "transfer_beta")
SCALAR_FIELDS = ("yield_alpha", "fet_transfer_mean")


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
    """Per-release privacy parameters."""

    epsilon: float = 1.0
    delta: float | None = None
    max_cycles_per_family: int = 6
    coefficient_range: float = 4.0  # bound on a released coefficient's span
    scalar_range: float = 10.0
    covariate_budget_share: float = 0.25  # share of epsilon spent on the covariate histogram

    def __post_init__(self) -> None:
        _check_cycle_bound(self.max_cycles_per_family, "max_cycles_per_family")

    def resolved_delta(self, n_families: int) -> float:
        return self.delta if self.delta is not None else default_delta(max(n_families, 2))


@dataclass
class FamilyBudget:
    """Cumulative family-unit spend for one node, across every release it makes.

    The audit's F4a: a node could previously emit ten payloads at ε=1 and no
    object in the system ever said ε=10. This is that object.
    """

    cap_epsilon: float
    delta: float
    accountant: RDPAccountant = None  # type: ignore[assignment]
    releases: int = 0

    def __post_init__(self) -> None:
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

        :func:`privatise_kernels` charges each release once, with every mechanism
        it will run, so a refusal leaves the budget untouched. ``new_release``
        counts payloads rather than charges, for a caller composing a payload
        from more than one charge.
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
    the rows it had, with its columns and types untouched; the first version
    concatenated the two frames, which gave each one the other's columns.

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


def _sensitivities(n_families: int, config: DPConfig) -> dict[str, float]:
    """Replace-one-family L2 sensitivities of the released fields.

    A rate is a mean over at most ``K`` cycles of one family out of ``n``
    families, so replacing a family moves it by at most ``K/n`` — the
    contribution-capping bound. Coefficients and scalars are bounded by their
    declared release range over the same denominator.
    """
    k, n = config.max_cycles_per_family, max(n_families, 1)
    rate = k / n
    coeff = config.coefficient_range * k / n
    scalar = config.scalar_range * k / n
    out = {f: rate for f in RATE_FIELDS}
    out.update({f: coeff for f in VECTOR_FIELDS})
    out.update({f: scalar for f in SCALAR_FIELDS})
    return out


def privatise_kernels(
    kernels: ProcessKernels,
    *,
    n_families: int,
    config: DPConfig,
    budget: FamilyBudget,
    rng: np.random.Generator,
) -> tuple[ProcessKernels, dict[str, object]]:
    """Return DP kernels and the accounting record the certificate will carry.

    Raises:
        BudgetExhausted: if this release would take the node past its cap. The
            node must then refuse to emit rather than release anyway. Nothing
            is drawn and nothing is charged.
        ValueError: if the kernels were not fitted on contribution-capped data,
            or were capped above ``config.max_cycles_per_family``.
    """
    # The sensitivities below assume a family contributes at most K cycles, and
    # until 1.1.0 nothing enforced it. The fit records its cap; refuse without one.
    cap = kernels.contribution_cap
    if cap is None or cap > config.max_cycles_per_family:
        raise ValueError(
            f"kernels fitted with contribution cap {cap}; the noise is calibrated to at most "
            f"{config.max_cycles_per_family} cycles per family. Fit them with "
            f"fit_kernels(..., contribution_cap={config.max_cycles_per_family})."
        )
    delta = config.resolved_delta(n_families)
    sens = _sensitivities(n_families, config)
    # The per-release budget covers the kernel fields AND the covariate
    # histogram; splitting it here is what makes the declared epsilon the
    # epsilon actually spent.
    kernel_epsilon = config.epsilon * (1.0 - config.covariate_budget_share)
    sigmas = {
        key: calibrate_sigma(epsilon=kernel_epsilon / len(sens), delta=delta, sensitivity=s)
        for key, s in sens.items()
    }
    hist_sensitivity, hist_sigma = _histogram_calibration(n_families, config)
    # One charge for the whole release, before any noise is drawn. The histogram
    # used to be charged after the kernel noise was drawn, so a release refused
    # there had already drawn and paid for its kernel noise. The mechanisms enter
    # the accountant in the same order as before, so every release that is not
    # refused spends exactly what it spent then.
    budget.charge(
        {**sens, "covariate_histogram": hist_sensitivity},
        {**sigmas, "covariate_histogram": hist_sigma},
    )

    updates: dict[str, Any] = {}
    for field in RATE_FIELDS:
        value = float(getattr(kernels, field)) + rng.normal(0.0, sigmas[field])
        updates[field] = float(np.clip(value, 1e-4, 1 - 1e-4))
    for field in SCALAR_FIELDS:
        value = float(getattr(kernels, field)) + rng.normal(0.0, sigmas[field])
        updates[field] = float(max(value, 1e-3))
    for field in VECTOR_FIELDS:
        vector = np.asarray(getattr(kernels, field), dtype=float)
        updates[field] = vector + rng.normal(0.0, sigmas[field], size=vector.shape)

    # The empirical covariate pool is verbatim real data and must never cross a
    # boundary: replace it with a DP histogram resample over a coarse grid.
    updates["covariate_pool"] = _private_covariate_pool(
        kernels.covariate_pool, sigma=hist_sigma, rng=rng
    )
    record = {
        "accounting": budget.as_dict(),
        "epsilon_this_release": float(config.epsilon),
        "delta": float(delta),
        "max_cycles_per_family": int(config.max_cycles_per_family),
        "n_families": int(n_families),
        "sigma_by_field": {k: float(v) for k, v in sigmas.items()},
        "mechanism": "gaussian output perturbation, RDP composition (custody._dp)",
    }
    return replace(kernels, **updates), record


def _histogram_calibration(n_families: int, config: DPConfig) -> tuple[float, float]:
    """Sensitivity and noise scale of the covariate histogram's share."""
    sensitivity = float(config.max_cycles_per_family)  # one family touches <= K cells
    sigma = calibrate_sigma(
        epsilon=config.epsilon * config.covariate_budget_share,
        delta=config.resolved_delta(n_families),
        sensitivity=sensitivity,
    )
    return sensitivity, sigma


def _private_covariate_pool(
    pool: np.ndarray,
    *,
    sigma: float,
    rng: np.random.Generator,
) -> np.ndarray:
    """Release covariates as a noised histogram, never as copied rows.

    The audit (F9) identified the empirical pool as the release's only verbatim
    channel. A DP histogram over integer (age, AFC) cells removes it: cell
    counts get Gaussian noise at family-unit sensitivity, and the pool is
    resampled from the noised distribution. The caller has already charged the
    budget for it.
    """
    grid = np.rint(pool).astype(int)
    cells, counts = np.unique(grid, axis=0, return_counts=True)
    noised = np.maximum(counts + rng.normal(0.0, sigma, size=counts.shape), 0.0)
    if noised.sum() <= 0:
        noised = np.ones_like(noised)
    probabilities = noised / noised.sum()
    draw = rng.choice(len(cells), size=len(pool), p=probabilities)
    return cells[draw].astype(float)
