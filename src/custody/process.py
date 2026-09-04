"""Rollout engine — a synthetic ART cohort that cannot violate the ledger.

The engine simulates the treatment process rather than sampling rows: a
patient enters with covariates, a fresh cycle yields oocytes, a thinning
cascade produces embryos, embryos are split between transfer and the bank,
an outcome is drawn, and later frozen cycles WITHDRAW from that bank. The
bank is a variable in the simulator's state, so "transferred an embryo that
was never banked" is not a defect to be repaired downstream — it is
unrepresentable, the way an account cannot overdraw when the code subtracts
before it spends.

Kernels are classical and deliberately commodity (:func:`fit_kernels`):
negative-binomial yield, binomial thinning, a logistic outcome, and empirical
allocation/continuation rates. Nothing here is claimed as novel; the
engineering claim is the invariant, not the estimator.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

__all__ = ["MU_OOCYTE_BOUNDS", "ProcessKernels", "RolloutConfig", "fit_kernels", "rollout_cohort"]

_EPS = 1e-9

# A PUBLIC physiological bound on the expected oocyte yield of one retrieval,
# not a data-derived quantity: clinical practice does not produce mean yields
# outside this range, and hyper-response above it is managed rather than
# expected. Clipping the simulated mean to it is post-processing of the
# released parameters, so it costs no privacy budget — and it is what stops a
# noised log-link coefficient from exploding multiplicatively across a wide
# covariate range. Without it, DP releases produced mean yields near 89 with a
# maximum of 367, against a real mean of 9.9.
MU_OOCYTE_BOUNDS = (0.5, 40.0)

# PUBLIC centring/scaling constants for the yield design, from ordinary clinical
# ranges rather than from this cohort. They exist so a released coefficient is
# on a bounded scale: fitted on raw AFC (range 0-87), a coefficient perturbed by
# DP noise of sd ~0.05 moves the linear predictor by ~4 and the mean yield by a
# factor of 50. On standardised covariates the same noise moves it by ~0.15.
AGE_CENTRE, AGE_SCALE = 32.0, 6.0
AFC_CENTRE, AFC_SCALE = 12.0, 8.0


def yield_design(age: np.ndarray | float, afc: np.ndarray | float) -> np.ndarray:
    """``[1, standardised age, standardised AFC]`` — the yield kernel's design."""
    return np.column_stack(
        [
            np.ones(np.size(age)),
            (np.asarray(age, dtype=float) - AGE_CENTRE) / AGE_SCALE,
            (np.asarray(afc, dtype=float) - AFC_CENTRE) / AFC_SCALE,
        ]
    )


@dataclass(frozen=True)
class RolloutConfig:
    """Simulation controls."""

    n_patients: int = 5000
    max_cycles: int = 6
    seed: int = 42


@dataclass(frozen=True)
class ProcessKernels:
    """Fitted process parameters. Plain numbers — inspectable and exchangeable."""

    covariate_pool: np.ndarray  # (n, 2) age_w, AF drawn empirically
    yield_beta: np.ndarray  # log-link coefficients on [1, age, AF]
    yield_alpha: float  # NB2 dispersion
    fert_rate: float
    dev_rate: float
    transfer_beta: np.ndarray  # logit coefficients on [1, age, n_embryos]
    p_bank_given_surplus: float
    # Continuation is a 2x2: the outcome decides whether treatment is over, and
    # among failures the frozen bank decides whether coming back is cheap. The
    # first version modelled the outcome alone and produced half the frozen
    # transfers the real cohort has.
    p_continue_fail_nobank: float
    p_continue_fail_bank: float
    p_continue_birth_nobank: float
    p_continue_birth_bank: float
    p_use_bank: float  # P(this cycle draws on the bank | the bank is non-empty)
    fet_transfer_mean: float
    fet_live_birth_rate: float  # FET transfers have their own outcome rate
    p_fresh_transfer: float  # else freeze-all: the whole cohort goes to the bank

    def as_dict(self) -> dict[str, object]:
        return {
            "yield_beta": [float(b) for b in self.yield_beta],
            "yield_alpha": float(self.yield_alpha),
            "fert_rate": float(self.fert_rate),
            "dev_rate": float(self.dev_rate),
            "transfer_beta": [float(b) for b in self.transfer_beta],
            "p_bank_given_surplus": float(self.p_bank_given_surplus),
            "p_continue_fail_nobank": float(self.p_continue_fail_nobank),
            "p_continue_fail_bank": float(self.p_continue_fail_bank),
            "p_continue_birth_nobank": float(self.p_continue_birth_nobank),
            "p_continue_birth_bank": float(self.p_continue_birth_bank),
            "p_use_bank": float(self.p_use_bank),
            "fet_transfer_mean": float(self.fet_transfer_mean),
            "fet_live_birth_rate": float(self.fet_live_birth_rate),
            "p_fresh_transfer": float(self.p_fresh_transfer),
            "n_covariate_pool": int(len(self.covariate_pool)),
        }


def _safe_rate(numerator: float, denominator: float, default: float) -> float:
    if denominator <= 0:
        return default
    return float(np.clip(numerator / denominator, 1e-4, 1 - 1e-4))


def fit_kernels(fresh: pd.DataFrame, fet: pd.DataFrame) -> ProcessKernels:
    """Fit the process from one centre's cycles. Classical estimators only.

    Args:
        fresh: Fresh cycles with age_w, AF, egg_num, fertilization_num, _2PN,
            transfer_embryo_num, freeze_num (or total_freeze_num), live_birth.
        fet: Frozen-transfer cycles with transfer_embryo_num.

    Returns:
        :class:`ProcessKernels`.
    """
    import statsmodels.api as sm
    from statsmodels.discrete.discrete_model import NegativeBinomial

    cov = fresh[["age_w", "AF"]].dropna()
    design = yield_design(cov["age_w"].to_numpy(), cov["AF"].to_numpy())
    eggs = fresh.loc[cov.index, "egg_num"].to_numpy(float)
    try:
        nb = NegativeBinomial(eggs, design).fit(disp=0, maxiter=200)
        yield_beta, yield_alpha = np.asarray(nb.params)[:-1], float(np.asarray(nb.params)[-1])
    except Exception:  # noqa: BLE001 - Poisson fallback keeps the engine runnable
        po = sm.GLM(eggs, design, family=sm.families.Poisson()).fit()
        yield_beta, yield_alpha = np.asarray(po.params), 0.5
    yield_alpha = float(np.clip(yield_alpha, 1e-3, 10.0))

    fert = _safe_rate(float(fresh["fertilization_num"].sum()), float(fresh["egg_num"].sum()), 0.7)
    dev = _safe_rate(float(fresh["_2PN"].sum()), float(fresh["fertilization_num"].sum()), 0.8)

    tr = fresh.dropna(subset=["age_w", "_2PN", "transfer_embryo_num", "live_birth"])
    tr = tr[tr["transfer_embryo_num"] > 0]
    tdesign = np.column_stack(
        [np.ones(len(tr)), tr["age_w"].to_numpy(float), tr["transfer_embryo_num"].to_numpy(float)]
    )
    outcome = sm.GLM(tr["live_birth"].to_numpy(float), tdesign, family=sm.families.Binomial()).fit()

    # Centres differ in which banking column they populate; pick by coverage,
    # never by presence: the demonstration centre has a handful of freeze_num rows and 80%
    # total_freeze_num, and picking the former banked nothing at all.
    candidates = [c for c in ("freeze_num", "total_freeze_num") if c in fresh.columns]
    banked_col = max(candidates, key=lambda c: float((fresh[c].fillna(0) > 0).mean()))
    surplus = fresh[fresh["_2PN"].fillna(0) > fresh["transfer_embryo_num"].fillna(0)]
    p_bank = _safe_rate(float((surplus[banked_col].fillna(0) > 0).sum()), float(len(surplus)), 0.5)

    behaviour = _fit_sequence_behaviour(fresh, fet, banked_col)

    fet_mean = float(fet["transfer_embryo_num"].dropna().clip(1, 3).mean()) if len(fet) else 1.5
    fet_transfers = fet[fet["transfer_embryo_num"].fillna(0) > 0] if len(fet) else fet
    fet_lbr = (
        _safe_rate(
            float(fet_transfers["live_birth"].fillna(0).sum()), float(len(fet_transfers)), 0.3
        )
        if len(fet_transfers)
        else 0.3
    )
    return ProcessKernels(
        covariate_pool=cov.to_numpy(float),
        yield_beta=yield_beta,
        yield_alpha=yield_alpha,
        fert_rate=fert,
        dev_rate=dev,
        transfer_beta=np.asarray(outcome.params),
        p_bank_given_surplus=p_bank,
        p_continue_fail_nobank=behaviour["p_continue_fail_nobank"],
        p_continue_fail_bank=behaviour["p_continue_fail_bank"],
        p_continue_birth_nobank=behaviour["p_continue_birth_nobank"],
        p_continue_birth_bank=behaviour["p_continue_birth_bank"],
        p_use_bank=behaviour["p_use_bank"],
        fet_transfer_mean=fet_mean,
        fet_live_birth_rate=fet_lbr,
        p_fresh_transfer=_safe_rate(
            float((fresh["transfer_embryo_num"].fillna(0) > 0).sum()), float(len(fresh)), 0.7
        ),
    )


def _fit_sequence_behaviour(
    fresh: pd.DataFrame, fet: pd.DataFrame, banked_col: str
) -> dict[str, float]:
    """Walk each family's sequence carrying the bank, and count what she did.

    Yields the 2x2 continuation table and the bank-usage rate. These were
    previously a hardcoded 0.6 and an outcome-only continuation, which halved
    the frozen-transfer share of the released cohort.
    """
    both = pd.concat(
        [fresh.assign(_kind="fresh"), fet.assign(_kind="fet")], ignore_index=True
    ).sort_values(["pid", "visit_date"])
    cont: dict[tuple[int, int], list[int]] = {}
    used, chances = 0, 0
    for _pid, group in both.groupby("pid", sort=False):
        bank, rows = 0.0, group.to_dict("records")
        for i, row in enumerate(rows):
            if bank > 0:
                chances += 1
                used += int(row["_kind"] == "fet")
            if row["_kind"] == "fet":
                bank -= float(row["transfer_embryo_num"] or 0)
            else:
                bank += float(row[banked_col] or 0)
            bank = max(bank, 0.0)
            key = (int(float(row["live_birth"] or 0) > 0), int(bank > 0))
            cell = cont.setdefault(key, [0, 0])
            cell[0] += int(i < len(rows) - 1)
            cell[1] += 1

    def rate(birth: int, has_bank: int, default: float) -> float:
        got, n = cont.get((birth, has_bank), [0, 0])
        return _safe_rate(float(got), float(n), default)

    return {
        "p_continue_fail_nobank": rate(0, 0, 0.45),
        "p_continue_fail_bank": rate(0, 1, 0.80),
        "p_continue_birth_nobank": rate(1, 0, 0.04),
        "p_continue_birth_bank": rate(1, 1, 0.05),
        "p_use_bank": _safe_rate(float(used), float(chances), 0.9),
    }


def _nb_draw(rng: np.random.Generator, mu: float, alpha: float) -> int:
    r = 1.0 / max(alpha, _EPS)
    p = r / (r + max(mu, _EPS))
    return int(rng.negative_binomial(r, p))


def rollout_cohort(kernels: ProcessKernels, config: RolloutConfig) -> pd.DataFrame:
    """Simulate a synthetic cohort. The bank lives in the state (see module docs).

    Returns:
        One row per cycle with the ledger schema's columns.
    """
    rng = np.random.default_rng(config.seed)
    rows: list[dict[str, object]] = []
    pool = kernels.covariate_pool
    for patient in range(config.n_patients):
        age, afc = pool[rng.integers(0, len(pool))]
        bank = 0  # <- the state variable that makes I2/I3 unrepresentable
        alive = True
        for cycle_index in range(config.max_cycles):
            if not alive:
                break
            use_bank = bank > 0 and rng.random() < kernels.p_use_bank
            if use_bank:
                transferred = int(
                    min(bank, max(1, round(rng.normal(kernels.fet_transfer_mean, 0.4))))
                )
                bank -= transferred  # subtract before spending: never negative, never orphan
                eggs = fertilised = embryos = banked = 0
                kind = "fet"
            else:
                eta = float(kernels.yield_beta @ yield_design(age, afc)[0])
                mu = float(np.clip(np.exp(np.clip(eta, -5, 5)), *MU_OOCYTE_BOUNDS))
                eggs = _nb_draw(rng, mu, kernels.yield_alpha)
                fertilised = int(rng.binomial(eggs, kernels.fert_rate))
                embryos = int(rng.binomial(fertilised, kernels.dev_rate))
                fresh_transfer = rng.random() < kernels.p_fresh_transfer
                transferred = (
                    min(embryos, int(rng.integers(1, 3))) if embryos > 0 and fresh_transfer else 0
                )
                surplus = embryos - transferred
                if not fresh_transfer:
                    banked = int(surplus)  # freeze-all: the whole cohort is banked
                else:
                    banked = (
                        int(surplus)
                        if surplus > 0 and rng.random() < kernels.p_bank_given_surplus
                        else 0
                    )
                bank += banked
                kind = "fresh"
            if transferred > 0 and kind == "fresh":
                eta = float(kernels.transfer_beta @ np.array([1.0, age, float(transferred)]))
                live_birth = int(rng.random() < 1.0 / (1.0 + np.exp(-np.clip(eta, -30, 30))))
            elif transferred > 0:
                live_birth = int(rng.random() < kernels.fet_live_birth_rate)
            else:
                live_birth = 0
            rows.append(
                {
                    "pid": f"S{patient:06d}",
                    "cycle_index": cycle_index,
                    "cycle_kind": kind,
                    "age_w": float(age),
                    "AF": float(afc),
                    "egg_num": float(eggs),
                    "fertilization_num": float(fertilised),
                    "_2PN": float(embryos),
                    "transfer_embryo_num": float(transferred),
                    "freeze_num": float(banked),
                    "live_birth": float(live_birth),
                }
            )
            if live_birth:
                p_cont = (
                    kernels.p_continue_birth_bank if bank > 0 else kernels.p_continue_birth_nobank
                )
            else:
                p_cont = (
                    kernels.p_continue_fail_bank if bank > 0 else kernels.p_continue_fail_nobank
                )
            alive = rng.random() < p_cont
    cohort: pd.DataFrame = pd.DataFrame(rows)
    return cohort
