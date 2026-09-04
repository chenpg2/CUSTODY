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


class TestDPRelease:
    """The DP layer the audit found missing: epsilon produced, not declared."""

    def _node(self, kernels, cap: float = 3.0, epsilon: float = 1.0):
        from custody import DPConfig, FamilyBudget, Node

        cfg = DPConfig(epsilon=epsilon)
        return Node(
            "n",
            kernels,
            dp=cfg,
            budget=FamilyBudget(cap_epsilon=cap, delta=cfg.resolved_delta(1000)),
            n_families=1000,
        )

    def test_noise_is_actually_applied(self, kernels: ProcessKernels) -> None:
        node = self._node(kernels)
        released = node.emit(n_patients=100, seed=1).kernels
        assert released["fert_rate"] != kernels.fert_rate
        assert released["yield_beta"] != list(kernels.yield_beta)

    def test_cumulative_spend_is_monotone_and_reported(self, kernels: ProcessKernels) -> None:
        node = self._node(kernels)
        spends = [node.emit(n_patients=80, seed=s).certificate.epsilon_total for s in (1, 2, 3)]
        assert spends == sorted(spends)
        assert all(a < b for a, b in zip(spends, spends[1:]))

    def test_releases_count_payloads_not_mechanisms(self, kernels: ProcessKernels) -> None:
        node = self._node(kernels)
        for expected in (1, 2, 3):
            assert node.emit(n_patients=80, seed=expected).certificate.releases_so_far == expected

    def test_cap_refuses_release(self, kernels: ProcessKernels) -> None:
        from custody import BudgetExhausted

        node = self._node(kernels, cap=0.45)
        node.emit(n_patients=80, seed=1)
        with pytest.raises(BudgetExhausted, match="past the declared cap"):
            for s in range(2, 20):
                node.emit(n_patients=80, seed=s)

    def test_covariate_pool_is_not_verbatim(self, kernels: ProcessKernels) -> None:
        """The audit's F9: the empirical pool was the release's only copy channel."""
        from custody import DPConfig, FamilyBudget, privatise_kernels

        cfg = DPConfig(epsilon=1.0)
        private, _ = privatise_kernels(
            kernels,
            n_families=1000,
            config=cfg,
            budget=FamilyBudget(cap_epsilon=5.0, delta=cfg.resolved_delta(1000)),
            rng=np.random.default_rng(0),
        )
        assert private.covariate_pool.shape == kernels.covariate_pool.shape
        assert not np.array_equal(private.covariate_pool, kernels.covariate_pool)

    def test_contribution_capping_bounds_a_family(self) -> None:
        from custody import cap_contributions

        fresh = pd.DataFrame(
            {"pid": ["p"] * 10, "visit_date": pd.date_range("2020-01-01", periods=10)}
        )
        fet = pd.DataFrame({"pid": [], "visit_date": []})
        capped_fresh, _, dropped = cap_contributions(fresh, fet, max_cycles=3)
        assert len(capped_fresh) == 3
        assert dropped == 7

    def test_receiver_refuses_a_unit_it_did_not_ask_for(self, kernels: ProcessKernels) -> None:
        node = self._node(kernels)
        payload = node.emit(n_patients=80, seed=1)
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

    def test_forged_epsilon_beyond_cap_is_refused(self, kernels: ProcessKernels) -> None:
        from custody import DPConfig, FamilyBudget, Node

        cfg = DPConfig(epsilon=1.0)
        node = Node(
            "n",
            kernels,
            dp=cfg,
            budget=FamilyBudget(cap_epsilon=2.0, delta=cfg.resolved_delta(1000)),
            n_families=1000,
        )
        payload = node.emit(n_patients=80, seed=1)
        forged = type(payload.certificate)(
            **{**payload.certificate.as_dict(), "epsilon_total": 99.0}
        )
        result = verify_certificate(payload.cohort, payload.kernels, forged)
        assert not result.passed
        assert "epsilon_substantiated" in result.failed_checks
