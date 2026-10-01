"""What a private release is computed from: clipped group sums, and nothing else.

Releases 1.0.0 and 1.1.0 added noise to kernels fitted by maximum likelihood and
gave them sensitivities that did not hold: the rates are ratios of cycle-level
sums, the regression coefficients had no enforced bound, and the covariate
histogram showed which cells occur. Here every field of a private release is a
function of Gaussian-noised sums whose sensitivity follows from public bounds, of
public constants, and of randomness that does not depend on the data. This module
holds the constants, the exact sums a node keeps, and the post-processing that
turns noised sums into kernels; :mod:`custody.privacy` adds the noise and the
accounting.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from .process import ProcessKernels, yield_design

__all__ = [
    "GROUPS",
    "HISTOGRAM_GROUP",
    "KERNEL_GROUPS",
    "POOL_SIZE",
    "PrivateFitError",
    "PrivateStatistics",
    "group_sums",
    "kernels_from_sums",
]

# Every per-cycle value is clipped to a public range before it is summed.
OOCYTE_MAX = 40.0
AGE_RANGE = (18.0, 55.0)
AFC_RANGE = (0.0, 60.0)
FET_TRANSFER_RANGE = (1.0, 3.0)
RATE_BOUNDS = (1e-4, 1.0 - 1e-4)
DISPERSION_BOUNDS = (1e-3, 10.0)

# The public grids, and the values a fit places each cell at.
AGE_REPRESENTATIVE = (31.0, 37.0, 42.0)  # under 35, 35-39, 40 and over
AFC_REPRESENTATIVE = (5.0, 11.0, 20.0)  # under 8, 8-15, over 15
TRANSFER_REPRESENTATIVE = (1.0, 2.0)  # one; two or more (the rollout transfers at most two)
POOL_AGE_WIDTH, POOL_AGE_BINS = 2.0, 19  # 18-55 in two-year bins
POOL_AFC_WIDTH, POOL_AFC_BINS = 4.0, 15  # 0-60 in four-count bins
POOL_SIZE = 20_000
MIN_FIT_CELLS = 3

KERNEL_GROUPS = ("G1", "G2", "G3", "G4", "G5_cells", "G5_sq", "G6")
HISTOGRAM_GROUP = "G7"
GROUPS = (*KERNEL_GROUPS, HISTOGRAM_GROUP)


class PrivateFitError(RuntimeError):
    """A fit on noised sums could not be made. The release fails; it never falls back."""


@dataclass(frozen=True)
class PrivateStatistics:
    """The exact group sums of one node's capped, clipped cycles. Never released.

    ``n_families`` counts every family in the node's data, fresh and frozen cycles
    together: it is invariant under replacing one family, and it sets delta.
    """

    n_families: int
    max_cycles_per_family: int
    bank_column: str
    sums: dict[str, np.ndarray]


def _age_band(age: np.ndarray) -> np.ndarray:
    return np.where(age < 35.0, 0, np.where(age < 40.0, 1, 2))


def _afc_band(afc: np.ndarray) -> np.ndarray:
    return np.where(afc < 8.0, 0, np.where(afc <= 15.0, 1, 2))


def _sequence_counts(fresh: pd.DataFrame, fet: pd.DataFrame, bank_column: str) -> np.ndarray:
    """G4: the continuation table and bank use, walked in each family's visit order.

    Returns (continued, total) for the cells (fail, no bank), (fail, bank),
    (birth, no bank), (birth, bank), then (cycles that drew on the bank, cycles
    with a non-empty bank before them). A fresh cycle comes before a frozen
    transfer on the same date, the order capping uses.
    """
    parts = []
    if len(fresh):
        parts.append(
            pd.DataFrame(
                {
                    "pid": fresh["pid"].to_numpy(),
                    "visit_date": fresh["visit_date"].to_numpy(),
                    "kind": 0,
                    "deposit": fresh[bank_column].fillna(0).clip(lower=0).to_numpy(float),
                    "draw": 0.0,
                    "birth": (fresh["live_birth"].fillna(0) > 0).to_numpy(),
                }
            )
        )
    if len(fet):
        parts.append(
            pd.DataFrame(
                {
                    "pid": fet["pid"].to_numpy(),
                    "visit_date": fet["visit_date"].to_numpy(),
                    "kind": 1,
                    "deposit": 0.0,
                    "draw": fet["transfer_embryo_num"].fillna(0).clip(lower=0).to_numpy(float),
                    "birth": (fet["live_birth"].fillna(0) > 0).to_numpy(),
                }
            )
        )
    counts = np.zeros(10)
    if not parts:
        return counts
    both = pd.concat(parts, ignore_index=True).sort_values(
        ["pid", "visit_date", "kind"], kind="mergesort"
    )
    for _pid, family in both.groupby("pid", sort=False):
        bank = 0.0
        rows = family[["kind", "deposit", "draw", "birth"]].to_numpy()
        for i, (kind, deposit, draw, birth) in enumerate(rows):
            if bank > 0:
                counts[9] += 1
                counts[8] += int(kind == 1)
            bank = max(bank - draw if kind == 1 else bank + deposit, 0.0)
            cell = 2 * int(bool(birth)) + int(bank > 0)
            counts[2 * cell] += int(i < len(rows) - 1)
            counts[2 * cell + 1] += 1
    return counts


def _baseline_histogram(fresh: pd.DataFrame) -> np.ndarray:
    """G7: one baseline (age, follicle count) per family, on the public pool grid."""
    rows = fresh[fresh["age_w"].notna() & fresh["AF"].notna()]
    first = rows.sort_values(["pid", "visit_date"], kind="mergesort").groupby("pid").head(1)
    age = first["age_w"].to_numpy(float).clip(*AGE_RANGE)
    afc = first["AF"].to_numpy(float).clip(*AFC_RANGE)
    age_bin = np.minimum(((age - AGE_RANGE[0]) // POOL_AGE_WIDTH).astype(int), POOL_AGE_BINS - 1)
    afc_bin = np.minimum((afc // POOL_AFC_WIDTH).astype(int), POOL_AFC_BINS - 1)
    cells = POOL_AGE_BINS * POOL_AFC_BINS
    return np.bincount(age_bin * POOL_AFC_BINS + afc_bin, minlength=cells).astype(float)


def group_sums(fresh: pd.DataFrame, fet: pd.DataFrame, bank_column: str) -> dict[str, np.ndarray]:
    """The exact sums of groups G1-G7 over already capped cycles."""
    eggs = fresh["egg_num"].fillna(0).clip(0.0, OOCYTE_MAX).to_numpy(float)
    fertilised = np.minimum(
        fresh["fertilization_num"].fillna(0).clip(lower=0).to_numpy(float), eggs
    )
    embryos = np.minimum(fresh["_2PN"].fillna(0).clip(lower=0).to_numpy(float), fertilised)
    transfers = fresh["transfer_embryo_num"].fillna(0).clip(lower=0).to_numpy(float)
    birth = (fresh["live_birth"].fillna(0) > 0).to_numpy()
    banked = (fresh[bank_column].fillna(0) > 0).to_numpy()
    surplus = embryos > transfers

    fet_transfers = fet["transfer_embryo_num"] if len(fet) else pd.Series(dtype=float)
    # G3 counts the frozen transfers whose embryo count is recorded, the ones the
    # mean embryos per transfer is taken over.
    known = fet_transfers.notna()
    fet_positive = (fet_transfers.fillna(0) > 0).to_numpy()
    fet_birth = (fet["live_birth"].fillna(0) > 0).to_numpy() if len(fet) else np.zeros(0, bool)

    covariates = (fresh["age_w"].notna() & fresh["AF"].notna()).to_numpy()
    age = fresh["age_w"].to_numpy(float).clip(*AGE_RANGE)
    afc = fresh["AF"].to_numpy(float).clip(*AFC_RANGE)
    yield_cell = 3 * _age_band(age[covariates]) + _afc_band(afc[covariates])
    yield_eggs = eggs[covariates]

    outcome_rows = (transfers > 0) & fresh["age_w"].notna().to_numpy()
    outcome_cell = 2 * _age_band(age[outcome_rows]) + (transfers[outcome_rows] >= 2).astype(int)

    return {
        "G1": np.array([eggs.sum(), fertilised.sum(), embryos.sum()]),
        "G2": np.array(
            [len(fresh), (transfers > 0).sum(), surplus.sum(), (surplus & banked).sum()], float
        ),
        "G3": np.array(
            [
                known.sum(),
                fet_transfers[known].clip(*FET_TRANSFER_RANGE).sum(),
                fet_positive.sum(),
                (fet_positive & fet_birth).sum(),
            ],
            float,
        ),
        "G4": _sequence_counts(fresh, fet, bank_column),
        # Counts scaled by the oocyte ceiling, so both halves of a family's vector
        # share one bound.
        "G5_cells": np.concatenate(
            [
                OOCYTE_MAX * np.bincount(yield_cell, minlength=9),
                np.bincount(yield_cell, weights=yield_eggs, minlength=9),
            ]
        ).astype(float),
        "G5_sq": np.array([float(np.sum(yield_eggs**2))]),
        "G6": np.concatenate(
            [
                np.bincount(outcome_cell, minlength=6),
                np.bincount(outcome_cell, weights=birth[outcome_rows].astype(float), minlength=6),
            ]
        ).astype(float),
        "G7": _baseline_histogram(fresh),
    }


def _rate(numerator: float, denominator: float) -> float:
    return float(np.clip(max(numerator, 0.0) / max(denominator, 1.0), *RATE_BOUNDS))


def _check_identified(name: str, design: np.ndarray, keep: np.ndarray) -> None:
    """A fit needs enough cells left, and cells whose design identifies every coefficient.

    Three cells of one age band, say, leave the age coefficient undetermined, and
    the fitter would return one split of it among many rather than fail.
    """
    if keep.sum() < MIN_FIT_CELLS:
        raise PrivateFitError(f"{name}: {int(keep.sum())} cells with positive noised weight")
    if np.linalg.matrix_rank(design[keep]) < design.shape[1]:
        raise PrivateFitError(f"{name}: the cells left do not identify the coefficients")


def _fit_yield(cells: np.ndarray, square: np.ndarray) -> tuple[np.ndarray, float]:
    """Poisson log-linear fit to the cell means, and the NB2 moment dispersion."""
    import statsmodels.api as sm

    counts = np.maximum(cells[:9], 0.0) / OOCYTE_MAX
    sums = np.maximum(cells[9:], 0.0)
    keep = counts > 0
    band = np.arange(9)
    design = yield_design(
        np.asarray(AGE_REPRESENTATIVE)[band // 3], np.asarray(AFC_REPRESENTATIVE)[band % 3]
    )
    _check_identified("yield", design, keep)
    try:
        fit = sm.GLM(
            sums[keep], design[keep], family=sm.families.Poisson(), exposure=counts[keep]
        ).fit()
    except (np.linalg.LinAlgError, ValueError) as exc:
        raise PrivateFitError(f"yield: the fit failed on noised cells ({exc})") from exc
    beta = np.asarray(fit.params, dtype=float)
    if not fit.converged or not np.all(np.isfinite(beta)):
        raise PrivateFitError("yield: the fit did not converge on noised cells")
    # A noised count is floored at one where it divides; the fit above uses it
    # unfloored, as a weight, which divides nothing.
    n, s = np.maximum(counts[keep], 1.0), sums[keep]
    total = n.sum()
    between = float(np.sum(s**2 / n))
    if between <= 0:
        raise PrivateFitError("yield: no oocytes in the noised cells, so no dispersion")
    rss = max(float(square[0]), 0.0) - between
    mean, mean_square = s.sum() / total, between / total
    alpha = (rss / total - mean) / mean_square
    return beta, float(np.clip(alpha, *DISPERSION_BOUNDS))


def _fit_outcome(cells: np.ndarray) -> np.ndarray:
    """Binomial logistic fit to the noised (trials, births) cells."""
    import statsmodels.api as sm

    trials = np.maximum(cells[:6], 0.0)
    births = np.clip(cells[6:], 0.0, trials)
    keep = trials > 0
    cell = np.arange(6)
    design = np.column_stack(
        [
            np.ones(6),
            np.asarray(AGE_REPRESENTATIVE)[cell // 2],
            np.asarray(TRANSFER_REPRESENTATIVE)[cell % 2],
        ]
    )
    _check_identified("outcome", design, keep)
    endog = np.column_stack([births[keep], trials[keep] - births[keep]])
    try:
        fit = sm.GLM(endog, design[keep], family=sm.families.Binomial()).fit()
    except (np.linalg.LinAlgError, ValueError) as exc:
        raise PrivateFitError(f"outcome: the fit failed on noised cells ({exc})") from exc
    beta = np.asarray(fit.params, dtype=float)
    if not fit.converged or not np.all(np.isfinite(beta)):
        raise PrivateFitError("outcome: the fit did not converge on noised cells")
    return beta


def _pool(histogram: np.ndarray, threshold: float, rng: np.random.Generator) -> np.ndarray:
    """A public-size draw from the thresholded histogram, at the cells' centres."""
    weights = np.where(histogram >= threshold, np.maximum(histogram, 0.0), 0.0)
    cells = len(weights)
    probabilities = weights / weights.sum() if weights.sum() > 0 else np.full(cells, 1.0 / cells)
    draw = rng.choice(cells, size=POOL_SIZE, p=probabilities)
    age = AGE_RANGE[0] + POOL_AGE_WIDTH * (draw // POOL_AFC_BINS) + POOL_AGE_WIDTH / 2
    afc = POOL_AFC_WIDTH * (draw % POOL_AFC_BINS) + POOL_AFC_WIDTH / 2
    return np.column_stack([age, afc]).astype(float)


def kernels_from_sums(
    sums: dict[str, np.ndarray],
    *,
    histogram_threshold: float,
    rng: np.random.Generator,
    contribution_cap: int,
) -> ProcessKernels:
    """Kernels as a function of (noised) sums, public constants and ``rng`` only.

    Raises:
        PrivateFitError: if a regression cannot be fitted. The caller's release
            then fails; its spend stands, because the noise was drawn.
    """
    eggs, fertilised, embryos = np.maximum(sums["G1"], 0.0)
    fresh, transferred, surplus, banked = np.maximum(sums["G2"], 0.0)
    fet_known, fet_embryos, fet_transfers, fet_births = np.maximum(sums["G3"], 0.0)
    seq = np.maximum(sums["G4"], 0.0)
    yield_beta, yield_alpha = _fit_yield(sums["G5_cells"], sums["G5_sq"])
    return ProcessKernels(
        covariate_pool=_pool(sums["G7"], histogram_threshold, rng),
        yield_beta=yield_beta,
        yield_alpha=yield_alpha,
        fert_rate=_rate(fertilised, eggs),
        dev_rate=_rate(embryos, fertilised),
        transfer_beta=_fit_outcome(sums["G6"]),
        p_bank_given_surplus=_rate(banked, surplus),
        p_continue_fail_nobank=_rate(seq[0], seq[1]),
        p_continue_fail_bank=_rate(seq[2], seq[3]),
        p_continue_birth_nobank=_rate(seq[4], seq[5]),
        p_continue_birth_bank=_rate(seq[6], seq[7]),
        p_use_bank=_rate(seq[8], seq[9]),
        fet_transfer_mean=float(np.clip(fet_embryos / max(fet_known, 1.0), *FET_TRANSFER_RANGE)),
        fet_live_birth_rate=_rate(fet_births, fet_transfers),
        p_fresh_transfer=_rate(transferred, fresh),
        contribution_cap=contribution_cap,
    )
