"""CUSTODY: a privacy-preserving exchange for assisted-reproduction cohorts.

What crosses an institutional boundary is a fitted treatment-process model and
the synthetic cohorts rolled out from it, each under a certificate the receiving
centre recomputes for itself. The rollout engine carries the embryo bank in its
state and subtracts before it spends, so a conservation violation is a value the
engine cannot write rather than one it repairs afterwards.

Four objects carry the whole path::

    from custody import Centre, Privacy, Receiver

    sender   = Centre.fit(fresh, fet, name="Centre_1")
    release  = sender.release(n_patients=500, privacy=Privacy(epsilon=1.0))
    release.verify().accepted        # what the receiver will conclude

    receiver = Receiver(Centre.fit(own_fresh, own_fet, name="Centre_2"))
    delivery = receiver.receive([release])
    delivery.accepted, delivery.rejected, delivery.cohort

Everything the objects wrap is importable directly for anyone who wants the
plumbing: :mod:`custody.cohort`, :mod:`custody.process`, :mod:`custody.privacy`,
:mod:`custody.certificate` and :mod:`custody.exchange`.

No mechanism here is new. The physiology cascade, contribution bounding, the
Gaussian mechanism and Renyi composition are all cited work; the assembly and
the assisted-reproduction instance are ours.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import pandas as pd

from .certificate import (
    Certificate,
    CertificateResult,
    cohort_digest,
    issue_certificate,
    kernel_digest,
    verify_certificate,
)
from .cohort import LedgerReport, LedgerSchema, check_ledger
from .exchange import ExchangeResult, Node, Payload, emit_payload, merge_kernels, receive
from .privacy import (
    BudgetExhausted,
    DPConfig,
    FamilyBudget,
    cap_contributions,
    privatise_kernels,
)
from .process import ProcessKernels, RolloutConfig, fit_kernels, rollout_cohort

__version__ = "1.0.0"

DEFAULT_PATIENTS = 500
DEFAULT_SEED = 42


@dataclass(frozen=True)
class Privacy:
    """What a release spends, and the shape of the unit it protects.

    The accounting unit is the family: one patient's trajectory together with
    her partner's and any offspring's records. ``max_cycles`` is the
    contribution bound, the first K cycles a family keeps, which is what makes
    the sensitivity finite.

    ``cap`` is the cumulative ceiling across releases from the same centre. A
    release that would cross it is refused rather than shrunk, which is the
    point: an exhausted budget is a refusal, not a quieter answer.
    """

    epsilon: float = 1.0
    delta: float | None = None
    max_cycles: int = 6
    cap: float | None = None

    def _config(self) -> DPConfig:
        return DPConfig(
            epsilon=self.epsilon, delta=self.delta, max_cycles_per_family=self.max_cycles
        )


@dataclass
class Verification:
    """What a receiver concluded about one release, and on which checks."""

    accepted: bool
    failed: list[str] = field(default_factory=list)
    checks: dict[str, bool] = field(default_factory=dict)
    detail: dict[str, Any] = field(default_factory=dict)

    @classmethod
    def _from(cls, result: CertificateResult) -> Verification:
        failed = [name for name, ok in result.checks.items() if not ok]
        return cls(
            accepted=result.passed,
            failed=failed,
            checks=dict(result.checks),
            detail=dict(result.detail),
        )

    def __bool__(self) -> bool:
        return self.accepted


@dataclass
class Release:
    """A synthetic cohort, the certificate beside it, and what it cost.

    Nothing in ``cohort`` corresponds to a person. It is rolled out from a
    fitted process, and the certificate is what lets a receiver establish that
    for itself rather than take it on trust.
    """

    cohort: pd.DataFrame
    certificate: Certificate
    kernels: dict[str, Any]
    centre: str
    epsilon: float | None = None
    accounting: dict[str, Any] = field(default_factory=dict)

    @classmethod
    def _from_payload(
        cls, payload: Payload, *, epsilon: float | None, accounting: dict[str, Any]
    ) -> Release:
        return cls(
            cohort=payload.cohort,
            certificate=payload.certificate,
            kernels=payload.kernels,
            centre=payload.node_id,
            epsilon=epsilon,
            accounting=accounting,
        )

    def check(self) -> LedgerReport:
        """Recompute the embryo-ledger invariants on the cohort."""
        return check_ledger(self.cohort)

    def verify(self) -> Verification:
        """Verify the certificate the way a receiving centre would.

        Every field is recomputed on what arrived. A payload edited in flight
        fails on the digest; one edited and re-certified still fails on the
        invariants, which is the check that does not depend on the sender.
        """
        return Verification._from(verify_certificate(self.cohort, self.kernels, self.certificate))

    def as_payload(self) -> Payload:
        """The wire form, for :meth:`Receiver.receive`."""
        return Payload(self.centre, self.kernels, self.cohort, self.certificate)

    @property
    def private(self) -> bool:
        """True when a formal guarantee was applied and priced."""
        return self.epsilon is not None

    def __repr__(self) -> str:
        eps = "plain" if self.epsilon is None else f"epsilon={self.epsilon:.4f}"
        return f"Release(centre={self.centre!r}, {len(self.cohort)} cycles, {eps})"


class Centre:
    """One centre's fitted treatment process.

    Fit it on the centre's own cycles, then roll cohorts out of it or release
    them. The records never leave: what a release carries is the fitted process
    and a cohort simulated from it.

    A centre that releases under privacy carries its budget across releases. The
    spend accumulates, and a release that would cross the cap is refused rather
    than quietly weakened.
    """

    def __init__(
        self, kernels: ProcessKernels, *, name: str = "centre", n_families: int = 0
    ) -> None:
        self._node = Node(node_id=name, kernels=kernels, n_families=n_families)

    @classmethod
    def fit(
        cls,
        fresh: pd.DataFrame,
        fet: pd.DataFrame,
        *,
        name: str = "centre",
        privacy: Privacy | None = None,
    ) -> Centre:
        """Fit the process from one centre's fresh cycles and frozen transfers.

        Classical estimators only. ``fresh`` and ``fet`` need the columns
        :class:`custody.LedgerSchema` names, plus ``age_w``, ``AF``,
        ``visit_date`` and ``live_birth``.

        Passing ``privacy`` applies contribution bounding before the fit, so a
        family contributes at most its first K cycles. The bound is what makes
        the sensitivity finite; it also removes failures selectively, which the
        paper states and does not correct for.
        """
        n_families = int(pd.concat([fresh, fet])["pid"].nunique())
        if privacy is not None:
            fresh, fet, n_families = cap_contributions(fresh, fet, privacy.max_cycles)
        return cls(fit_kernels(fresh, fet), name=name, n_families=n_families)

    @property
    def kernels(self) -> ProcessKernels:
        return self._node.kernels

    @property
    def name(self) -> str:
        return self._node.node_id

    @property
    def n_families(self) -> int:
        return self._node.n_families

    def rollout(
        self, n_patients: int = DEFAULT_PATIENTS, *, seed: int = DEFAULT_SEED, max_cycles: int = 6
    ) -> pd.DataFrame:
        """Simulate a synthetic cohort. Conservation holds by construction.

        No certificate, no privacy: this is the local view, for looking at what
        the fitted process produces before deciding to release anything.
        """
        return rollout_cohort(
            self.kernels,
            RolloutConfig(n_patients=n_patients, max_cycles=max_cycles, seed=seed),
        )

    def release(
        self,
        n_patients: int = DEFAULT_PATIENTS,
        *,
        seed: int = DEFAULT_SEED,
        privacy: Privacy | None = None,
    ) -> Release:
        """Roll out a cohort and certify it.

        Without ``privacy`` the release is plain: the certificate still binds
        the cohort to the kernels, but no formal guarantee is claimed. With it,
        the kernels are privatised under the family unit first, and the epsilon
        on the certificate is what the accountant produced rather than what the
        operator declared.

        Raises:
            BudgetExhausted: if the release would carry the cumulative spend
                past the cap. Nothing is emitted and no noise is drawn.
        """
        if privacy is not None:
            self._node.dp = privacy._config()
            if self._node.budget is None or privacy.cap is not None:
                self._node.budget = self._node.budget or FamilyBudget(
                    cap_epsilon=(privacy.cap if privacy.cap is not None else privacy.epsilon),
                    delta=privacy._config().resolved_delta(self.n_families),
                )
        payload = self._node.emit(n_patients=n_patients, seed=seed)
        spent = None if self._node.budget is None else float(self._node.budget.spent)
        return Release._from_payload(payload, epsilon=spent, accounting=dict(self._node.dp_record))

    @property
    def spent(self) -> float:
        """Cumulative epsilon this centre has released, across every release."""
        return 0.0 if self._node.budget is None else float(self._node.budget.spent)

    def as_node(self, *, corrupt: bool = False) -> Node:
        """The underlying node, for the exchange plumbing.

        ``corrupt=True`` makes it tamper with its own payload after certifying
        it, which is the threat the receiver is meant to catch without being
        told which sender to distrust.
        """
        self._node.corrupt = corrupt
        return self._node

    def __repr__(self) -> str:
        return f"Centre(name={self.name!r}, families={self.n_families})"


@dataclass
class Delivery:
    """What a receiver did with the releases it was handed."""

    accepted: list[str]
    rejected: dict[str, list[str]]
    cohort: pd.DataFrame | None
    merged: ProcessKernels | None
    verifications: dict[str, dict[str, Any]] = field(default_factory=dict)

    @property
    def clean(self) -> bool:
        """True when the replayed cohort violates no ledger invariant."""
        return self.cohort is not None and check_ledger(self.cohort).clean

    def __repr__(self) -> str:
        return (
            f"Delivery(accepted={self.accepted}, "
            f"rejected={sorted(self.rejected)}, clean={self.clean})"
        )


class Receiver:
    """The receiving centre.

    It is told nothing about which sender to distrust. It recomputes every
    certificate on what arrived, refuses what fails, merges the rest with its
    own fitted process, and replays a cohort under its own policy.
    """

    def __init__(
        self, own: Centre, *, replay_patients: int = 2000, replay_seed: int = DEFAULT_SEED
    ) -> None:
        self.own = own
        self.replay_patients = replay_patients
        self.replay_seed = replay_seed

    def receive(self, releases: list[Release]) -> Delivery:
        """Verify, refuse, merge and replay."""
        result: ExchangeResult = receive(
            [r.as_payload() for r in releases],
            receiver_kernels=self.own.kernels,
            replay_patients=self.replay_patients,
            replay_seed=self.replay_seed,
        )
        return Delivery(
            accepted=list(result.accepted),
            rejected=dict(result.rejected),
            cohort=result.replayed,
            merged=result.merged,
            verifications=dict(result.verifications),
        )

    def __repr__(self) -> str:
        return f"Receiver(own={self.own.name!r})"


__all__ = [
    # the facade
    "Centre",
    "Delivery",
    "Privacy",
    "Receiver",
    "Release",
    "Verification",
    # the plumbing, for anyone who wants it
    "BudgetExhausted",
    "Certificate",
    "CertificateResult",
    "DPConfig",
    "ExchangeResult",
    "FamilyBudget",
    "LedgerReport",
    "LedgerSchema",
    "Node",
    "Payload",
    "ProcessKernels",
    "RolloutConfig",
    "cap_contributions",
    "check_ledger",
    "cohort_digest",
    "emit_payload",
    "fit_kernels",
    "issue_certificate",
    "kernel_digest",
    "merge_kernels",
    "privatise_kernels",
    "receive",
    "rollout_cohort",
    "verify_certificate",
]
