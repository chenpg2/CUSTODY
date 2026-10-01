"""Tests for the embryo ledger, the certificate, and the exchange (spec v2 V1/V2/V5)."""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from custody import (  # noqa: E402
    Node,
    ProcessKernels,
    RolloutConfig,
    check_ledger,
    issue_certificate,
    merge_kernels,
    receive,
    rollout_cohort,
    verify_certificate,
)


def _cycle(pid: str, idx: int, kind: str, **kw) -> dict:
    row = {
        "pid": pid,
        "cycle_index": idx,
        "cycle_kind": kind,
        "egg_num": 0.0,
        "fertilization_num": 0.0,
        "_2PN": 0.0,
        "transfer_embryo_num": 0.0,
        "freeze_num": 0.0,
        "live_birth": 0.0,
    }
    row.update(kw)
    return row


@pytest.fixture
def kernels() -> ProcessKernels:
    return ProcessKernels(
        covariate_pool=np.array([[32.0, 12.0], [36.0, 8.0], [29.0, 18.0]]),
        yield_beta=np.array([2.3, -0.02, 0.03]),
        yield_alpha=0.1,
        fert_rate=0.79,
        dev_rate=0.79,
        transfer_beta=np.array([2.0, -0.09, 0.32]),
        p_bank_given_surplus=0.73,
        p_continue_fail_nobank=0.47,
        p_continue_fail_bank=0.80,
        p_continue_birth_nobank=0.04,
        p_continue_birth_bank=0.05,
        p_use_bank=0.92,
        fet_transfer_mean=1.6,
        fet_live_birth_rate=0.43,
        p_fresh_transfer=0.53,
    )


def _as_capped(kernels: ProcessKernels, cap: int = 6) -> ProcessKernels:
    """The hand-set fixture, declared as a fit capped at K cycles per family.

    A private release refuses kernels fitted without a contribution cap. The
    fixture is not fitted from data, so it stands in for kernels that were.
    """
    from dataclasses import replace

    return replace(kernels, contribution_cap=cap)


def _small_cohort() -> tuple[pd.DataFrame, pd.DataFrame]:
    """Eighty synthetic patients; the first has eight fresh cycles, past a cap of six."""
    rng = np.random.default_rng(0)
    fresh_rows, fet_rows = [], []
    for p in range(80):
        for c in range(8 if p == 0 else int(rng.integers(1, 3))):
            eggs = int(rng.poisson(10))
            fert = int(rng.binomial(eggs, 0.7))
            pn = int(rng.binomial(fert, 0.8))
            transfer, freeze = min(2, pn), max(0, pn - min(2, pn))
            date = pd.Timestamp("2020-01-01") + pd.Timedelta(days=90 * c)
            fresh_rows.append(
                {
                    "pid": f"p{p}",
                    "visit_date": date,
                    "age_w": float(rng.uniform(25, 40)),
                    "AF": float(rng.uniform(4, 20)),
                    "egg_num": eggs,
                    "fertilization_num": fert,
                    "_2PN": pn,
                    "transfer_embryo_num": transfer,
                    "freeze_num": freeze,
                    "live_birth": int(transfer > 0 and rng.random() < 0.3),
                }
            )
            if freeze:
                fet_rows.append(
                    {
                        "pid": f"p{p}",
                        "visit_date": date + pd.Timedelta(days=45),
                        "transfer_embryo_num": 1,
                        "live_birth": int(rng.random() < 0.4),
                    }
                )
    return pd.DataFrame(fresh_rows), pd.DataFrame(fet_rows)


class TestInvariants:
    def test_honest_sequence_is_clean(self) -> None:
        frame = pd.DataFrame(
            [
                _cycle(
                    "p1",
                    0,
                    "fresh",
                    egg_num=10,
                    fertilization_num=8,
                    _2PN=6,
                    transfer_embryo_num=2,
                    freeze_num=4,
                ),
                _cycle("p1", 1, "fet", transfer_embryo_num=2),
                _cycle("p1", 2, "fet", transfer_embryo_num=1),
            ]
        )
        assert check_ledger(frame).clean

    def test_orphan_consumption_is_caught(self) -> None:
        frame = pd.DataFrame([_cycle("p1", 0, "fet", transfer_embryo_num=2)])
        report = check_ledger(frame)
        assert report.counts["I3_no_orphan_consumption"] == 1
        assert report.counts["I4_precedence"] == 1

    def test_overdraw_is_caught(self) -> None:
        frame = pd.DataFrame(
            [
                _cycle(
                    "p1",
                    0,
                    "fresh",
                    egg_num=6,
                    fertilization_num=4,
                    _2PN=3,
                    transfer_embryo_num=1,
                    freeze_num=2,
                ),
                _cycle("p1", 1, "fet", transfer_embryo_num=5),  # bank holds 2
            ]
        )
        assert check_ledger(frame).counts["I3_no_orphan_consumption"] == 1

    def test_cascade_violation_is_caught(self) -> None:
        frame = pd.DataFrame([_cycle("p1", 0, "fresh", egg_num=4, fertilization_num=9, _2PN=2)])
        assert check_ledger(frame).counts["I1_stage_monotone"] == 1

    def test_banks_are_per_patient(self) -> None:
        """One patient's deposit must not fund another's withdrawal."""
        frame = pd.DataFrame(
            [
                _cycle("p1", 0, "fresh", egg_num=10, fertilization_num=8, _2PN=6, freeze_num=6),
                _cycle("p2", 0, "fet", transfer_embryo_num=2),
            ]
        )
        assert check_ledger(frame).counts["I3_no_orphan_consumption"] == 1


class TestRollout:
    def test_rollout_is_clean_and_reproducible(self, kernels: ProcessKernels) -> None:
        a = rollout_cohort(kernels, RolloutConfig(n_patients=200, seed=7))
        b = rollout_cohort(kernels, RolloutConfig(n_patients=200, seed=7))
        assert check_ledger(a).clean
        pd.testing.assert_frame_equal(a, b)

    def test_different_seeds_differ(self, kernels: ProcessKernels) -> None:
        a = rollout_cohort(kernels, RolloutConfig(n_patients=200, seed=7))
        b = rollout_cohort(kernels, RolloutConfig(n_patients=200, seed=8))
        assert not a.equals(b)


class TestCertificate:
    def test_honest_payload_verifies(self, kernels: ProcessKernels) -> None:
        cohort = rollout_cohort(kernels, RolloutConfig(n_patients=100, seed=1))
        kd = kernels.as_dict()
        cert = issue_certificate(cohort, kd, epsilon_by_role={"woman": 0.6, "partner": 0.4})
        assert verify_certificate(cohort, kd, cert).passed

    def test_edit_breaks_the_digest(self, kernels: ProcessKernels) -> None:
        cohort = rollout_cohort(kernels, RolloutConfig(n_patients=100, seed=1))
        kd = kernels.as_dict()
        cert = issue_certificate(cohort, kd, epsilon_by_role={"woman": 1.0})
        tampered = cohort.copy()
        tampered.loc[tampered.index[0], "egg_num"] += 1
        result = verify_certificate(tampered, kd, cert)
        assert not result.passed
        assert "cohort_digest" in result.failed_checks

    def test_verification_is_total_not_first_failure(self, kernels: ProcessKernels) -> None:
        cohort = rollout_cohort(kernels, RolloutConfig(n_patients=100, seed=1))
        kd = kernels.as_dict()
        cert = issue_certificate(cohort, kd, epsilon_by_role={"woman": 1.0})
        broken = cohort.drop(index=cohort.index[0])
        result = verify_certificate(broken, {**kd, "fert_rate": 0.1}, cert)
        assert {"cohort_digest", "kernel_digest", "row_count"} <= set(result.failed_checks)


class TestExchange:
    def test_corrupt_node_is_rejected_and_others_survive(self, kernels: ProcessKernels) -> None:
        nodes = [
            Node("a", kernels),
            Node("b", kernels, corrupt=True),
            Node("c", kernels),
        ]
        payloads = [n.emit(n_patients=150, seed=10 + i) for i, n in enumerate(nodes)]
        result = receive(payloads, receiver_kernels=kernels, replay_patients=150)
        assert sorted(result.accepted) == ["a", "c"]
        assert "ledger_invariants" in result.rejected["b"]
        assert result.replayed is not None
        assert check_ledger(result.replayed).clean

    def test_receiver_keeps_its_own_policy(self, kernels: ProcessKernels) -> None:
        from dataclasses import replace

        sender = replace(kernels, p_fresh_transfer=0.05, p_continue_fail_bank=0.9)
        receiver = replace(kernels, p_fresh_transfer=0.80, p_continue_fail_bank=0.2)
        payloads = [Node("s", sender).emit(n_patients=150, seed=3)]
        result = receive(payloads, receiver_kernels=receiver, replay_patients=150)
        assert result.merged is not None
        assert result.merged.p_fresh_transfer == pytest.approx(0.80)
        assert result.merged.p_continue_fail_bank == pytest.approx(0.2)

    def test_all_corrupt_yields_nothing(self, kernels: ProcessKernels) -> None:
        payloads = [Node("x", kernels, corrupt=True).emit(n_patients=150, seed=4)]
        result = receive(payloads, receiver_kernels=kernels)
        assert result.accepted == []
        assert result.merged is None and result.replayed is None

    def test_merge_requires_input(self) -> None:
        with pytest.raises(ValueError, match="no verified kernels"):
            merge_kernels([])


def _private(epsilon: float = 1e5):
    """A private-release config for the small cohort, whose banking column is freeze_num.

    Eighty families are far too few for a useful release at a realistic budget,
    so functional tests use a large epsilon; the mechanism is the same.
    """
    from custody import DPConfig

    return DPConfig(epsilon=epsilon, bank_column="freeze_num")


def _private_node(cfg, cap: float = 1e12, fresh=None, fet=None):
    from custody import FamilyBudget, Node, private_statistics, zero_noise_kernels

    if fresh is None:
        fresh, fet = _small_cohort()
    stats = private_statistics(fresh, fet, cfg)
    return Node(
        "n",
        zero_noise_kernels(stats, cfg),
        dp=cfg,
        budget=FamilyBudget(cap_epsilon=cap, delta=cfg.resolved_delta(stats.n_families)),
        private=stats,
    )


def _stated_bounds(k: int) -> dict[str, tuple[np.ndarray, float]]:
    """Each group's coordinate bounds b and norm bound C for one family's vector."""
    e = 40.0
    return {
        "G1": (np.full(3, k * e), k * e * np.sqrt(3)),
        "G2": (np.full(4, float(k)), 2.0 * k),
        "G3": (np.array([k, 3 * k, k, k], float), k * np.sqrt(12)),
        "G4": (
            np.array([k - 1, k] * 4 + [k - 1, k - 1], float),
            np.sqrt(k**2 + 3 * (k - 1) ** 2),
        ),
        "G5_cells": (np.full(18, k * e), np.sqrt(2) * k * e),
        "G5_sq": (np.array([k * e**2]), k * e**2),
        "G6": (np.full(12, float(k)), np.sqrt(2) * k),
        "G7": (np.ones(285), 1.0),
    }


class TestDPRelease:
    """The privacy layer: every released field from noised, clipped group sums."""

    def test_private_noise_does_not_follow_the_simulation_seed(self) -> None:
        """A release made for others must not be reproducible from a seed it carries."""
        first = _private_node(_private()).emit(n_patients=80, seed=1).kernels
        second = _private_node(_private()).emit(n_patients=80, seed=1).kernels
        assert first != second

    def test_an_explicit_noise_seed_reproduces_the_seeded_stream(self) -> None:
        """Experiments pass a noise seed, and then draw exactly what they drew before."""
        from custody import FamilyBudget, release_private_kernels

        cfg = _private()
        node = _private_node(cfg)
        assert node.private is not None
        emitted = node.emit(n_patients=80, seed=5, noise_seed=5).kernels
        again = _private_node(cfg).emit(n_patients=80, seed=5, noise_seed=5).kernels
        assert emitted == again
        delta = cfg.resolved_delta(node.private.n_families)
        direct, _record = release_private_kernels(
            node.private,
            config=cfg,
            budget=FamilyBudget(cap_epsilon=1e12, delta=delta),
            rng=np.random.default_rng(5),
        )
        assert emitted == direct.as_dict()

    def test_noise_is_applied_against_the_zero_noise_estimator(self) -> None:
        node = _private_node(_private())
        released = node.emit(n_patients=100, seed=1).kernels
        assert released["fert_rate"] != node.kernels.fert_rate
        assert released["yield_beta"] != list(node.kernels.yield_beta)

    def test_the_zero_noise_estimator_is_deterministic(self) -> None:
        from custody import private_statistics, zero_noise_kernels

        cfg = _private()
        stats = private_statistics(*_small_cohort(), cfg)
        a, b = zero_noise_kernels(stats, cfg), zero_noise_kernels(stats, cfg)
        assert a.as_dict() == b.as_dict()
        assert np.array_equal(a.covariate_pool, b.covariate_pool)

    def test_cumulative_spend_is_monotone_and_reported(self) -> None:
        node = _private_node(_private())
        spends = [node.emit(n_patients=80, seed=s).certificate.epsilon_total for s in (1, 2, 3)]
        assert all(a < b for a, b in zip(spends, spends[1:]))

    def test_releases_count_payloads_not_mechanisms(self) -> None:
        node = _private_node(_private())
        for expected in (1, 2, 3):
            assert node.emit(n_patients=80, seed=expected).certificate.releases_so_far == expected

    def test_a_refused_release_draws_nothing_and_is_charged_nothing(self) -> None:
        from custody import BudgetExhausted, FamilyBudget, release_private_kernels

        cfg = _private()
        node = _private_node(cfg)
        node.emit(n_patients=80, seed=1)
        assert node.budget is not None and node.private is not None
        one_release = node.budget.spent
        budget = FamilyBudget(cap_epsilon=1.5 * one_release, delta=node.budget.delta)
        release_private_kernels(
            node.private, config=cfg, budget=budget, rng=np.random.default_rng(1)
        )
        spent, releases = budget.spent, budget.releases
        rng = np.random.default_rng(2)
        state = rng.bit_generator.state
        with pytest.raises(BudgetExhausted, match="past the declared cap"):
            release_private_kernels(node.private, config=cfg, budget=budget, rng=rng)
        assert (budget.spent, budget.releases) == (spent, releases)
        assert rng.bit_generator.state == state

    def test_the_spend_depends_only_on_public_constants(self) -> None:
        """The paper's values, at its demonstration centre's 34,367 families.

        The sums are placeholders: the spend is fixed by the share and delta alone,
        whether or not the fits on them succeed.
        """
        from custody import (
            DPConfig,
            FamilyBudget,
            PrivateFitError,
            PrivateStatistics,
            private_statistics,
            release_private_kernels,
        )

        cfg = DPConfig(epsilon=1.0)
        shaped = private_statistics(*_small_cohort(), _private()).sums
        zeros = {g: np.zeros_like(v) for g, v in shaped.items()}
        stats = PrivateStatistics(34_367, 6, cfg.bank_column, zeros)
        budget = FamilyBudget(cap_epsilon=0.55, delta=cfg.resolved_delta(34_367))
        spends = []
        for seed in (1, 2):
            try:
                release_private_kernels(
                    stats, config=cfg, budget=budget, rng=np.random.default_rng(seed)
                )
            except PrivateFitError:
                pass
            spends.append(round(budget.spent, 4))
        assert spends == [0.3797, 0.5387]

    def test_a_failed_fit_fails_the_release_and_the_spend_stands(self) -> None:
        """No fallback to a plain fit; the noise was drawn, so the charge remains."""
        from dataclasses import replace

        from custody import FamilyBudget, PrivateFitError, release_private_kernels

        cfg = _private()
        node = _private_node(cfg)
        assert node.private is not None and node.budget is not None
        sums = dict(node.private.sums)
        sums["G5_cells"] = np.concatenate([np.full(9, -1e9), sums["G5_cells"][9:]])
        broken = replace(node.private, sums=sums)
        budget = FamilyBudget(cap_epsilon=1e12, delta=node.budget.delta)
        with pytest.raises(PrivateFitError, match="yield"):
            release_private_kernels(broken, config=cfg, budget=budget, rng=np.random.default_rng(0))
        assert budget.spent > 0 and budget.releases == 1

    def test_a_fit_its_cells_do_not_identify_fails(self) -> None:
        """Cells of one age band, or of one transfer level, leave a coefficient free.

        The fitter would return one split of it among many; the release fails instead.
        """
        from custody import PrivateFitError, private_statistics
        from custody.private_stats import kernels_from_sums

        sums = private_statistics(*_small_cohort(), _private()).sums
        one_age_band = dict(sums)
        weights = sums["G5_cells"].copy()
        weights[:9] = 40.0 * np.array([100, 100, 100, -1, -1, -1, -1, -1, -1])
        one_age_band["G5_cells"] = weights
        one_transfer_level = dict(sums)
        one_transfer_level["G6"] = np.array([50.0, -1, 40, -1, 30, -1, 15, 0, 10, 0, 5, 0])
        no_oocytes = dict(sums)
        no_oocytes["G5_cells"] = np.concatenate([np.full(9, 4000.0), np.zeros(9)])
        for broken, match in (
            (one_age_band, "yield: the cells left do not identify"),
            (one_transfer_level, "outcome: the cells left do not identify"),
            (no_oocytes, "yield"),
        ):
            with pytest.raises(PrivateFitError, match=match):
                kernels_from_sums(
                    broken,
                    histogram_threshold=0.0,
                    rng=np.random.default_rng(0),
                    contribution_cap=6,
                )

    def test_n_counts_every_family(self) -> None:
        """Families with frozen transfers only count, as replace-one-family requires."""
        from custody import private_statistics

        fresh, fet = _small_cohort()
        extra = pd.DataFrame(
            {
                "pid": ["fet_only"],
                "visit_date": [pd.Timestamp("2021-01-01")],
                "transfer_embryo_num": [1],
                "live_birth": [0],
            }
        )
        stats = private_statistics(fresh, pd.concat([fet, extra], ignore_index=True), _private())
        assert stats.n_families == fresh["pid"].nunique() + 1

    def test_the_bank_column_is_the_configured_constant(self) -> None:
        from custody import DPConfig, private_statistics

        fresh, fet = _small_cohort()
        fresh = fresh.assign(total_freeze_num=0)
        by_freeze = private_statistics(fresh, fet, DPConfig(bank_column="freeze_num"))
        by_total = private_statistics(fresh, fet, DPConfig(bank_column="total_freeze_num"))
        assert by_freeze.sums["G2"][3] > 0 and by_total.sums["G2"][3] == 0

    def test_every_group_sum_moves_within_its_sensitivity(self) -> None:
        """Replace one family, including a worst case, and measure each group's move."""
        from custody import group_sensitivities, private_statistics

        cfg = _private()
        fresh, fet = _small_cohort()
        base = private_statistics(fresh, fet, cfg).sums
        bound = group_sensitivities(cfg.max_cycles_per_family)
        worst_fresh = pd.DataFrame(
            {
                "pid": ["p1"] * 6,
                "visit_date": pd.date_range("2019-01-01", periods=6, freq="90D"),
                "age_w": 45.0,
                "AF": 30.0,
                "egg_num": 60,
                "fertilization_num": 60,
                "_2PN": 60,
                "transfer_embryo_num": 2,
                "freeze_num": 58,
                "live_birth": 1,
            }
        )
        candidates = [worst_fresh] + [
            fresh[fresh["pid"] == f"p{q}"].assign(pid="p1") for q in (0, 2, 3, 5, 8)
        ]
        for replacement in candidates:
            new_fresh = pd.concat([fresh[fresh["pid"] != "p1"], replacement], ignore_index=True)
            new_fet = fet[fet["pid"] != "p1"]
            moved = private_statistics(new_fresh, new_fet, cfg).sums
            for group, limit in bound.items():
                assert np.linalg.norm(moved[group] - base[group]) <= limit + 1e-9, group

    def test_one_familys_vector_stays_inside_the_stated_bounds(self) -> None:
        """The coordinate bounds b and the norm bound C of every group, on adversarial families.

        The families mix frozen transfers in, share dates between a fresh cycle and a
        transfer, run past the cap, and carry missing values, which the replacement
        test above never reaches.
        """
        from custody import private_statistics

        cfg, bounds = _private(), _stated_bounds(6)
        day = pd.Timestamp("2020-01-01")

        def fresh_rows(n, **kw):
            base = {
                "pid": "f",
                "age_w": 45.0,
                "AF": 30.0,
                "egg_num": 60,
                "fertilization_num": 70,
                "_2PN": 80,
                "transfer_embryo_num": 2,
                "freeze_num": 58,
                "live_birth": 1,
            }
            return pd.DataFrame(
                [{**base, "visit_date": day + pd.Timedelta(days=90 * i), **kw} for i in range(n)]
            )

        def fet_rows(n, **kw):
            base = {"pid": "f", "transfer_embryo_num": 5, "live_birth": 1}  # above the clip
            return pd.DataFrame(
                [{**base, "visit_date": day + pd.Timedelta(days=90 * i), **kw} for i in range(n)]
            )

        empty_fresh, empty_fet = fresh_rows(1).iloc[0:0], fet_rows(1).iloc[0:0]
        families = [
            (fresh_rows(8), empty_fet),  # past the cap, every value at or above its clip
            (fresh_rows(4), fet_rows(4)),  # a transfer on each fresh cycle's date
            (empty_fresh, fet_rows(8)),  # frozen transfers only
            (fresh_rows(3, egg_num=np.nan, AF=np.nan), fet_rows(3, transfer_embryo_num=np.nan)),
            (fresh_rows(2, transfer_embryo_num=1, live_birth=0, freeze_num=0), fet_rows(1)),
        ]
        for fresh, fet in families:
            sums = private_statistics(fresh, fet, cfg).sums
            for group, (b, c) in bounds.items():
                vector = sums[group]
                assert (vector >= 0).all(), group
                assert (vector <= b + 1e-9).all(), group
                assert np.linalg.norm(vector) <= c + 1e-9, group

    def test_each_sensitivity_is_the_bound_its_coordinates_imply(self) -> None:
        """min(√2·C, ‖b‖) for every group, at several bounds.

        The headline spends cannot pin a sensitivity: the spend depends on the
        ratio of sensitivity to noise, and the noise is calibrated to the same
        sensitivity.
        """
        from custody import group_sensitivities

        for k in (1, 2, 6, 10):
            stated = group_sensitivities(k)
            for group, (b, c) in _stated_bounds(k).items():
                implied = min(np.sqrt(2) * c, float(np.linalg.norm(b)))
                assert stated[group] == pytest.approx(implied), (k, group)

    def test_the_bank_walk_follows_visit_order(self) -> None:
        """The order capping uses: by date, and a fresh cycle before a transfer on its date.

        Each family banks in a fresh cycle and draws in a transfer, one on the same
        day and one the next; walked in any other order, the transfer finds no bank.
        """
        from custody import private_statistics

        day = pd.Timestamp("2020-01-01")
        fresh = pd.DataFrame(
            {
                "pid": ["w", "v"],
                "visit_date": [day, day],
                "age_w": [30.0, 30.0],
                "AF": [10.0, 10.0],
                "egg_num": [8, 8],
                "fertilization_num": [6, 6],
                "_2PN": [5, 5],
                "transfer_embryo_num": [1, 1],
                "freeze_num": [3, 3],
                "live_birth": [0, 0],
            }
        )
        fet = pd.DataFrame(
            {
                "pid": ["w", "v"],
                "visit_date": [day, day + pd.Timedelta(days=1)],
                "transfer_embryo_num": [1, 1],
                "live_birth": [1, 1],
            }
        )
        g4 = private_statistics(fresh, fet, _private()).sums["G4"]
        assert (g4[8], g4[9]) == (2.0, 2.0)

    def test_every_group_is_noised_at_its_scale(self, monkeypatch: pytest.MonkeyPatch) -> None:
        import custody.privacy as privacy

        seen: dict[str, np.ndarray] = {}
        options: dict[str, object] = {}
        derive = privacy.kernels_from_sums

        def spy(sums, **kwargs):
            seen.update(sums)
            options.update(kwargs)
            return derive(sums, **kwargs)

        monkeypatch.setattr(privacy, "kernels_from_sums", spy)
        node = _private_node(_private())
        assert node.private is not None
        node.emit(n_patients=80, seed=1, noise_seed=3)
        sigma = node.dp_record["sigma_by_group"]
        standardised = []
        for group, exact in node.private.sums.items():
            assert not np.array_equal(seen[group], exact), group
            standardised.append((seen[group] - exact) / sigma[group])
        assert 0.75 < float(np.var(np.concatenate(standardised))) < 1.3
        # Histogram cells below twice their noise scale are dropped before the pool is drawn.
        assert options["histogram_threshold"] == pytest.approx(2.0 * sigma["G7"])

    def test_the_cohort_is_rolled_out_from_the_private_kernels(
        self, kernels: ProcessKernels
    ) -> None:
        """Never from the node's own fit, whose covariate pool here is off the public grid."""
        from custody import FamilyBudget, release_private_kernels

        cfg = _private()
        stats = _private_node(cfg).private
        assert stats is not None
        node = Node("n", _as_capped(kernels), dp=cfg, private=stats)
        payload = node.emit(n_patients=80, seed=5, noise_seed=5)
        direct, _record = release_private_kernels(
            stats,
            config=cfg,
            budget=FamilyBudget(cap_epsilon=1e12, delta=cfg.resolved_delta(stats.n_families)),
            rng=np.random.default_rng(5),
        )
        expected = rollout_cohort(direct, RolloutConfig(n_patients=80, seed=5))
        pd.testing.assert_frame_equal(payload.cohort, expected)

    def test_a_release_refused_before_its_charge_leaves_no_budget(
        self, kernels: ProcessKernels
    ) -> None:
        from custody import DPConfig, private_statistics

        stats = private_statistics(*_small_cohort(), _private())
        tight = DPConfig(epsilon=1e5, max_cycles_per_family=3, bank_column="freeze_num")
        node = Node("n", kernels, dp=tight, private=stats)
        with pytest.raises(ValueError, match="capped at 6"):
            node.emit(n_patients=80, seed=1)
        assert node.budget is None and node.dp_record == {}

    def test_group_sums_add_over_families(self) -> None:
        """Replacing one family moves each sum by exactly that family's vector."""
        from custody import private_statistics

        cfg = _private()
        fresh, fet = _small_cohort()
        half = {f"p{i}" for i in range(0, 80, 2)}
        whole = private_statistics(fresh, fet, cfg)
        a = private_statistics(fresh[fresh["pid"].isin(half)], fet[fet["pid"].isin(half)], cfg)
        b = private_statistics(fresh[~fresh["pid"].isin(half)], fet[~fet["pid"].isin(half)], cfg)
        assert whole.n_families == a.n_families + b.n_families
        for group, total in whole.sums.items():
            assert np.allclose(total, a.sums[group] + b.sums[group]), group

    def test_released_covariates_are_grid_centres_never_rows(self) -> None:
        """No released covariate pair is a copied record."""
        payload = _private_node(_private()).emit(n_patients=200, seed=1)
        ages, afcs = payload.cohort["age_w"].unique(), payload.cohort["AF"].unique()
        assert set(ages) <= {19.0 + 2 * i for i in range(19)}
        assert set(afcs) <= {2.0 + 4 * j for j in range(15)}

    def test_the_threshold_and_the_uniform_fallback(self) -> None:
        from custody import private_statistics
        from custody.private_stats import kernels_from_sums

        cfg = _private()
        sums = dict(private_statistics(*_small_cohort(), cfg).sums)
        histogram = np.zeros(285)
        histogram[[10, 20]] = [100.0, 3.0]
        sums["G7"] = histogram
        pool = kernels_from_sums(
            sums, histogram_threshold=5.0, rng=np.random.default_rng(0), contribution_cap=6
        ).covariate_pool
        assert len(np.unique(pool, axis=0)) == 1  # the cell below the threshold is gone
        sums["G7"] = np.full(285, 1.0)
        pool = kernels_from_sums(
            sums, histogram_threshold=5.0, rng=np.random.default_rng(0), contribution_cap=6
        ).covariate_pool
        assert len(np.unique(pool, axis=0)) > 250  # every cell zeroed: uniform over the grid

    def test_the_zero_noise_fits_recover_known_parameters(self) -> None:
        """The yield fit and the dispersion moment, on data drawn from known values."""
        from custody import private_statistics, zero_noise_kernels
        from custody.process import yield_design

        rng = np.random.default_rng(3)
        n, beta, alpha = 20_000, np.array([2.0, -0.2, 0.3]), 0.3
        age = rng.choice([31.0, 37.0, 42.0], n)
        afc = rng.choice([5.0, 11.0, 20.0], n)
        mu = np.exp(yield_design(age, afc) @ beta)
        eggs = rng.negative_binomial(1 / alpha, (1 / alpha) / (1 / alpha + mu))
        transfers = rng.integers(1, 3, n)
        fresh = pd.DataFrame(
            {
                "pid": [f"q{i}" for i in range(n)],
                "visit_date": pd.Timestamp("2020-01-01"),
                "age_w": age,
                "AF": afc,
                "egg_num": eggs,
                "fertilization_num": eggs,
                "_2PN": eggs,
                "transfer_embryo_num": transfers,
                "freeze_num": 0,
                "live_birth": rng.integers(0, 2, n),
            }
        )
        fet = pd.DataFrame(columns=["pid", "visit_date", "transfer_embryo_num", "live_birth"])
        fet["visit_date"] = pd.to_datetime(fet["visit_date"])
        cfg = _private()
        kernels = zero_noise_kernels(private_statistics(fresh, fet, cfg), cfg)
        assert np.allclose(kernels.yield_beta, beta, atol=0.02)
        assert abs(kernels.yield_alpha - alpha) < 0.03

    def test_a_charged_release_that_fails_stays_on_the_nodes_books(self) -> None:
        """A node given no budget still keeps the charge of a release that failed to fit."""
        from dataclasses import replace

        from custody import BudgetExhausted, FamilyBudget, Node, PrivateFitError

        cfg = _private()
        good = _private_node(cfg)
        assert good.private is not None
        sums = dict(good.private.sums)
        sums["G5_cells"] = np.concatenate([np.full(9, -1e9), sums["G5_cells"][9:]])
        node = Node("n", good.kernels, dp=cfg, private=replace(good.private, sums=sums))
        with pytest.raises(PrivateFitError):
            node.emit(n_patients=80, seed=1)
        assert isinstance(node.budget, FamilyBudget)
        assert node.budget.releases == 1 and node.budget.spent > 0
        # The default cap is the requested epsilon, one release's worth; the failed
        # release already spent it, so the next is refused.
        node.private = good.private
        with pytest.raises(BudgetExhausted):
            node.emit(n_patients=80, seed=2)

    def test_values_are_clipped_before_they_are_summed(self) -> None:
        from custody import private_statistics

        fresh, fet = _small_cohort()
        one = fresh.iloc[[0]].assign(pid="solo", egg_num=60, fertilization_num=70, _2PN=80)
        sums = private_statistics(one, fet.iloc[0:0], _private()).sums
        assert list(sums["G1"]) == [40.0, 40.0, 40.0]

    def test_the_pool_has_the_public_size_and_grid(self) -> None:
        released = _private_node(_private()).emit(n_patients=80, seed=1)
        assert released.kernels["n_covariate_pool"] == 20_000

    def test_a_private_node_needs_its_statistics(self, kernels: ProcessKernels) -> None:
        """Refused before any budget exists, so a node given none is left without one."""
        node = Node("n", kernels, dp=_private())
        with pytest.raises(ValueError, match="no private statistics"):
            node.emit(n_patients=80, seed=1)
        assert node.budget is None and node.dp_record == {}

    def test_statistics_capped_above_the_release_bound_are_refused(self) -> None:
        from custody import DPConfig, FamilyBudget, private_statistics, release_private_kernels

        stats = private_statistics(*_small_cohort(), DPConfig(bank_column="freeze_num"))
        tight = DPConfig(max_cycles_per_family=3, bank_column="freeze_num")
        with pytest.raises(ValueError, match="capped at 6"):
            release_private_kernels(
                stats,
                config=tight,
                budget=FamilyBudget(1e12, tight.resolved_delta(stats.n_families)),
                rng=np.random.default_rng(0),
            )

    def test_a_capped_fit_keeps_each_family_to_its_first_k_cycles(self) -> None:
        from custody import cap_contributions, fit_kernels

        fresh, fet = _small_cohort()
        capped_fresh, capped_fet, dropped = cap_contributions(fresh, fet, max_cycles=6)
        per_family = pd.concat([capped_fresh["pid"], capped_fet["pid"]]).value_counts()
        assert per_family.max() == 6
        assert dropped == len(fresh) + len(fet) - len(capped_fresh) - len(capped_fet)
        # The first six in visit order, and the frames keep their own columns.
        first = pd.concat([fresh, fet])
        first = first[first["pid"] == "p0"].sort_values("visit_date").head(6)
        kept = pd.concat([capped_fresh, capped_fet])
        assert sorted(kept[kept["pid"] == "p0"]["visit_date"]) == sorted(first["visit_date"])
        assert list(capped_fresh.columns) == list(fresh.columns)
        assert list(capped_fet.columns) == list(fet.columns)
        assert set(kept["pid"]) == set(first["pid"]) | set(fresh["pid"])  # no family lost
        assert fit_kernels(fresh, fet, contribution_cap=6).contribution_cap == 6
        assert fit_kernels(fresh, fet).contribution_cap is None

    def test_capping_orders_by_date_even_with_no_frozen_transfers(self) -> None:
        """A centre with no transfers used to have its cycles kept in storage order."""
        from custody import cap_contributions

        dates = pd.to_datetime(["2020-04-01", "2020-03-01", "2020-02-01", "2020-01-01"])
        fresh = pd.DataFrame({"pid": ["p"] * 4, "visit_date": dates})
        capped, _, dropped = cap_contributions(fresh, pd.DataFrame(), max_cycles=2)
        assert sorted(capped["visit_date"]) == list(dates[[3, 2]])
        assert dropped == 2

    def test_a_fresh_cycle_comes_before_a_transfer_on_the_same_date(self) -> None:
        from custody import cap_contributions

        day = pd.Timestamp("2020-01-01")
        fresh = pd.DataFrame({"pid": ["p"], "visit_date": [day]})
        fet = pd.DataFrame({"pid": ["p"], "visit_date": [day]})
        capped_fresh, capped_fet, _ = cap_contributions(fresh, fet, max_cycles=1)
        assert (len(capped_fresh), len(capped_fet)) == (1, 0)

    def test_capping_refuses_to_guess_an_order(self) -> None:
        from custody import cap_contributions

        undated = pd.DataFrame({"pid": ["p", "p"]})
        gap = pd.DataFrame({"pid": ["p", "p"], "visit_date": [pd.Timestamp("2020-01-01"), None]})
        text = pd.DataFrame({"pid": ["p", "p"], "visit_date": ["12/01/2019", "02/01/2020"]})
        for fresh in (undated, gap, text):
            with pytest.raises(ValueError, match="visit_date"):
                cap_contributions(fresh, pd.DataFrame(), max_cycles=1)
        nameless = pd.DataFrame(
            {"pid": ["p", None], "visit_date": pd.to_datetime(["2020-01-01", "2020-02-01"])}
        )
        with pytest.raises(ValueError, match="pid"):
            cap_contributions(nameless, pd.DataFrame(), max_cycles=1)

    def test_the_bound_is_a_whole_number_of_cycles(self) -> None:
        """A bound of 2.5 kept three cycles and priced two."""
        from custody import DPConfig, Privacy, cap_contributions

        fresh, fet = _small_cohort()
        for bad in (0, -1, 2.5, True):
            with pytest.raises(ValueError, match="whole number"):
                cap_contributions(fresh, fet, max_cycles=bad)  # type: ignore[arg-type]
            with pytest.raises(ValueError, match="whole number"):
                DPConfig(max_cycles_per_family=bad)  # type: ignore[arg-type]
            with pytest.raises(ValueError, match="whole number"):
                Privacy(max_cycles=bad)  # type: ignore[arg-type]

    def test_a_fit_does_not_depend_on_the_index(self) -> None:
        """A repeated index label paired one covariate row with several outcomes."""
        from custody import fit_kernels

        fresh, fet = _small_cohort()
        repeated = fresh.set_axis(np.arange(len(fresh)) // 2)
        for cap in (None, 6):
            assert (
                fit_kernels(repeated, fet, contribution_cap=cap).as_dict()
                == fit_kernels(fresh, fet, contribution_cap=cap).as_dict()
            )

    def test_a_node_set_up_for_private_release_does_not_release_plainly(
        self, kernels: ProcessKernels
    ) -> None:
        with pytest.raises(ValueError, match="set up for private release"):
            Node("n", _as_capped(kernels)).emit(n_patients=80, seed=1)
        stats = _private_node(_private()).private
        with pytest.raises(ValueError, match="set up for private release"):
            Node("n", kernels, private=stats).emit(n_patients=80, seed=1)

    def test_a_merge_is_capped_only_if_every_input_was(self, kernels: ProcessKernels) -> None:
        """The merge used to take the first input's cap, whatever the others were."""
        capped, uncapped = _as_capped(kernels, 6), kernels
        assert merge_kernels([capped, uncapped]).contribution_cap is None
        assert merge_kernels([uncapped, capped]).contribution_cap is None
        assert merge_kernels([_as_capped(kernels, 4), capped]).contribution_cap == 6

    def test_contribution_capping_bounds_a_family(self) -> None:
        from custody import cap_contributions

        fresh = pd.DataFrame(
            {"pid": ["p"] * 10, "visit_date": pd.date_range("2020-01-01", periods=10)}
        )
        fet = pd.DataFrame({"pid": [], "visit_date": []})
        capped_fresh, _, dropped = cap_contributions(fresh, fet, max_cycles=3)
        assert len(capped_fresh) == 3
        assert dropped == 7

    def test_receiver_refuses_a_unit_it_did_not_ask_for(self) -> None:
        payload = _private_node(_private()).emit(n_patients=80, seed=1)
        result = verify_certificate(
            payload.cohort, payload.kernels, payload.certificate, expected_unit="record"
        )
        assert not result.passed
        assert "accounting_unit_matches_policy" in result.failed_checks


class TestEpsilonSubstantiation:
    """A budget may not be claimed without the cap and delta that back it."""

    def test_non_dp_certificate_claims_nothing(self, kernels: ProcessKernels) -> None:
        cohort = rollout_cohort(kernels, RolloutConfig(n_patients=80, seed=1))
        cert = issue_certificate(cohort, kernels.as_dict(), epsilon_by_role={"woman": 1.0})
        assert cert.epsilon_total is None
        assert verify_certificate(cohort, kernels.as_dict(), cert).passed

    def test_forged_epsilon_is_refused_without_a_mechanism(self, kernels: ProcessKernels) -> None:
        cohort = rollout_cohort(kernels, RolloutConfig(n_patients=80, seed=1))
        kd = kernels.as_dict()
        cert = issue_certificate(cohort, kd, epsilon_by_role={"woman": 1.0})
        forged = type(cert)(**{**cert.as_dict(), "epsilon_total": 99.0})
        result = verify_certificate(cohort, kd, forged)
        assert not result.passed
        assert "epsilon_substantiated" in result.failed_checks

    def test_forged_epsilon_beyond_cap_is_refused(self) -> None:
        node = _private_node(_private(), cap=1e12)
        payload = node.emit(n_patients=80, seed=1)
        forged = type(payload.certificate)(
            **{**payload.certificate.as_dict(), "epsilon_total": 1e13}
        )
        result = verify_certificate(payload.cohort, payload.kernels, forged)
        assert not result.passed
        assert "epsilon_substantiated" in result.failed_checks


class TestBalanceIsNotATautology:
    """The balance count was unreachable, and the checker was repairing.

    An overdraw was recorded as an orphan consumption, the running stock was
    reset to zero, and the balance test was skipped. Every cohort the checker
    had ever seen reported zero balance violations, whatever else was wrong
    with it.
    """

    @staticmethod
    def _patient(rows: list[tuple[str, int, int]]) -> pd.DataFrame:
        """(kind, transferred, banked) per cycle, for one patient."""
        return pd.DataFrame(
            [
                {
                    "pid": "P1",
                    "cycle_index": i,
                    "cycle_kind": kind,
                    "egg_num": 8 if kind == "fresh" else 0,
                    "fertilization_num": 6 if kind == "fresh" else 0,
                    "_2PN": 5 if kind == "fresh" else 0,
                    "transfer_embryo_num": transferred,
                    "freeze_num": banked,
                }
                for i, (kind, transferred, banked) in enumerate(rows)
            ]
        )

    def test_a_deficit_is_counted(self) -> None:
        frame = self._patient([("fresh", 2, 2), ("fet", 5, 0)])
        assert check_ledger(frame).counts["I2_balance"] == 1

    def test_the_deficit_persists_until_deposits_repay_it(self) -> None:
        frame = self._patient([("fresh", 2, 2), ("fet", 5, 0), ("fresh", 1, 1), ("fet", 1, 0)])
        counts = check_ledger(frame).counts
        # Three cycles spent in deficit; two withdrawals that exceeded the
        # stock available to them. Two different measurements.
        assert counts["I2_balance"] == 3
        assert counts["I3_no_orphan_consumption"] == 2

    def test_a_conserving_cohort_still_reports_zero(self) -> None:
        frame = self._patient([("fresh", 1, 3), ("fet", 2, 0), ("fet", 1, 0)])
        assert check_ledger(frame).clean

    def test_the_checker_no_longer_repairs_the_stock(self) -> None:
        frame = self._patient([("fresh", 0, 1), ("fet", 5, 0), ("fet", 1, 0)])
        counts = check_ledger(frame).counts
        assert counts["I2_balance"] == 2
        assert counts["I3_no_orphan_consumption"] == 2
