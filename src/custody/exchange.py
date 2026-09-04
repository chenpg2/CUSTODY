"""The exchange itself — nodes, payloads, and what a receiver does with them.

This is the layer the Swarm Learning genre calls the middleware: what crosses
the institutional boundary, who verifies it, and what a receiving centre can
do with it that it could not do alone. Three moving parts:

:class:`Node`
    One centre. Fits its kernels locally, emits a :class:`Payload`, and never
    exposes a row. A node can be honest or (for the acceptance suite)
    corrupted, which matters because the receiver must not need to know which.

:class:`Payload`
    Kernels + a synthetic cohort + the release certificate. Self-describing
    and self-verifiable: the receiver recomputes every claim it makes.

:func:`receive`
    The receiver's protocol: verify each payload, DISCARD the ones that fail,
    merge what survives, and replay under the receiver's OWN policy. The last
    step is the point — a receiving clinic does not inherit another centre's
    transfer culture, it applies its own to better-informed physiology.
"""

from __future__ import annotations

from dataclasses import dataclass, field, replace
from typing import Any

import numpy as np
import pandas as pd

from .certificate import Certificate, issue_certificate, verify_certificate
from .privacy import DPConfig, FamilyBudget, privatise_kernels
from .process import ProcessKernels, RolloutConfig, rollout_cohort

__all__ = ["ExchangeResult", "Node", "Payload", "emit_payload", "merge_kernels", "receive"]


@dataclass(frozen=True)
class Payload:
    """What crosses the boundary. Contains no patient row of the sender's."""

    node_id: str
    kernels: dict[str, Any]
    cohort: pd.DataFrame
    certificate: Certificate

    @property
    def n_bytes(self) -> int:
        """Wire size of the payload, for the systems table."""
        return int(
            self.cohort.memory_usage(deep=True).sum()
            + len(str(self.kernels))
            + len(str(self.certificate.as_dict()))
        )


@dataclass
class Node:
    """A participating centre.

    When ``dp`` is set the node privatises its kernels before release and the
    certificate reports the budget the accountant PRODUCED. Without it the node
    releases plain fits, which is honest only for engineering tests that make
    no privacy claim — ``dp=None`` must never accompany a privacy sentence.
    """

    node_id: str
    kernels: ProcessKernels
    epsilon_by_role: dict[str, float] = field(
        default_factory=lambda: {"woman": 0.6, "partner": 0.25, "offspring": 0.15}
    )
    corrupt: bool = False
    dp: DPConfig | None = None
    budget: FamilyBudget | None = None
    n_families: int = 0
    dp_record: dict[str, Any] = field(default_factory=dict)

    def emit(self, *, n_patients: int, seed: int) -> Payload:
        """Roll out a cohort and certify it. A corrupt node certifies, then lies."""
        return emit_payload(self, n_patients=n_patients, seed=seed)


def emit_payload(node: Node, *, n_patients: int, seed: int) -> Payload:
    """Produce a node's payload; corrupt nodes tamper AFTER certification.

    Tampering after the certificate is issued is the realistic threat: the
    sender's own pipeline was honest, and something (a bug, a middlebox, a
    malicious relay) altered the payload in flight. The receiver must catch it
    without any knowledge of which node is compromised.

    Raises:
        BudgetExhausted: when the node's cumulative family-unit spend would
            pass its declared cap. A node out of budget refuses to release.
    """
    kernels = node.kernels
    epsilon_total, epsilon_cap, delta, releases = None, None, None, 1
    if node.dp is not None:
        budget = node.budget or FamilyBudget(
            cap_epsilon=node.dp.epsilon,
            delta=node.dp.resolved_delta(node.n_families),
        )
        node.budget = budget
        kernels, record = privatise_kernels(
            node.kernels,
            n_families=node.n_families,
            config=node.dp,
            budget=budget,
            rng=np.random.default_rng(seed),
        )
        node.dp_record = record
        epsilon_total, epsilon_cap = budget.spent, budget.cap_epsilon
        delta, releases = budget.delta, budget.releases
    cohort = rollout_cohort(kernels, RolloutConfig(n_patients=n_patients, seed=seed))
    kernel_dict = kernels.as_dict()
    certificate = issue_certificate(
        cohort,
        kernel_dict,
        epsilon_by_role=node.epsilon_by_role,
        epsilon_total=epsilon_total,
        epsilon_cap=epsilon_cap,
        delta=delta,
        releases_so_far=releases,
    )
    if node.corrupt:
        cohort = cohort.copy()
        fet_rows = cohort.index[cohort["cycle_kind"] == "fet"]
        target = fet_rows[0] if len(fet_rows) else cohort.index[0]
        cohort.loc[target, "transfer_embryo_num"] = 50.0  # withdraw what no bank holds
    return Payload(node.node_id, kernel_dict, cohort, certificate)


def merge_kernels(kernels: list[ProcessKernels]) -> ProcessKernels:
    """Combine verified kernels: precision-free mean of parameters, pooled covariates.

    Deliberately the simplest defensible merge. E-A/E-A2 measured what
    heterogeneity-aware pooling buys on this cohort (little, and negatively
    above small n), so the system does not claim a clever aggregator — it
    claims a verifiable exchange.
    """
    if not kernels:
        raise ValueError("no verified kernels to merge")
    stack = lambda attr: np.mean([getattr(k, attr) for k in kernels], axis=0)  # noqa: E731
    return replace(
        kernels[0],
        covariate_pool=np.vstack([k.covariate_pool for k in kernels]),
        yield_beta=stack("yield_beta"),
        yield_alpha=float(stack("yield_alpha")),
        fert_rate=float(stack("fert_rate")),
        dev_rate=float(stack("dev_rate")),
        transfer_beta=stack("transfer_beta"),
        p_bank_given_surplus=float(stack("p_bank_given_surplus")),
        p_continue_fail_nobank=float(stack("p_continue_fail_nobank")),
        p_continue_fail_bank=float(stack("p_continue_fail_bank")),
        p_continue_birth_nobank=float(stack("p_continue_birth_nobank")),
        p_continue_birth_bank=float(stack("p_continue_birth_bank")),
        p_use_bank=float(stack("p_use_bank")),
        fet_transfer_mean=float(stack("fet_transfer_mean")),
        fet_live_birth_rate=float(stack("fet_live_birth_rate")),
        p_fresh_transfer=float(stack("p_fresh_transfer")),
    )


@dataclass
class ExchangeResult:
    """What the receiver ended up with, and why."""

    accepted: list[str] = field(default_factory=list)
    rejected: dict[str, list[str]] = field(default_factory=dict)
    merged: ProcessKernels | None = None
    replayed: pd.DataFrame | None = None
    verifications: dict[str, dict[str, Any]] = field(default_factory=dict)

    def as_dict(self) -> dict[str, Any]:
        return {
            "accepted": list(self.accepted),
            "rejected": {k: list(v) for k, v in self.rejected.items()},
            "n_accepted": len(self.accepted),
            "n_rejected": len(self.rejected),
            "verifications": self.verifications,
            "merged_kernels": self.merged.as_dict() if self.merged else None,
            "replay_rows": int(len(self.replayed)) if self.replayed is not None else 0,
        }


def receive(
    payloads: list[Payload],
    *,
    receiver_kernels: ProcessKernels,
    replay_patients: int = 2000,
    replay_seed: int = 42,
) -> ExchangeResult:
    """Verify, discard, merge, and replay under the RECEIVER's own policy.

    Args:
        payloads: What arrived, honest and otherwise.
        receiver_kernels: The receiving centre's own fitted kernels; its
            policy parameters (transfer/banking/continuation) are kept.
        replay_patients: Size of the replayed cohort.
        replay_seed: Simulation seed for the replay.

    Returns:
        An :class:`ExchangeResult`. Rejected payloads contribute nothing —
        not to the merge, not to the replay.
    """
    result = ExchangeResult()
    verified: list[ProcessKernels] = []
    for payload in payloads:
        check = verify_certificate(payload.cohort, payload.kernels, payload.certificate)
        result.verifications[payload.node_id] = check.as_dict()
        if check.passed:
            result.accepted.append(payload.node_id)
            verified.append(_kernels_from_dict(payload.kernels, receiver_kernels))
        else:
            result.rejected[payload.node_id] = check.failed_checks
    if not verified:
        return result
    merged = merge_kernels(verified)
    # The receiver keeps its own policy layer; only nature crosses the boundary.
    receiver_view = replace(
        merged,
        covariate_pool=receiver_kernels.covariate_pool,
        transfer_beta=merged.transfer_beta,
        p_bank_given_surplus=receiver_kernels.p_bank_given_surplus,
        p_continue_fail_nobank=receiver_kernels.p_continue_fail_nobank,
        p_continue_fail_bank=receiver_kernels.p_continue_fail_bank,
        p_continue_birth_nobank=receiver_kernels.p_continue_birth_nobank,
        p_continue_birth_bank=receiver_kernels.p_continue_birth_bank,
        p_use_bank=receiver_kernels.p_use_bank,
        p_fresh_transfer=receiver_kernels.p_fresh_transfer,
        fet_transfer_mean=receiver_kernels.fet_transfer_mean,
    )
    result.merged = receiver_view
    result.replayed = rollout_cohort(
        receiver_view, RolloutConfig(n_patients=replay_patients, seed=replay_seed)
    )
    return result


def _kernels_from_dict(payload: dict[str, Any], template: ProcessKernels) -> ProcessKernels:
    """Rebuild kernels from the wire form, borrowing the template's covariate pool."""
    return replace(
        template,
        yield_beta=np.asarray(payload["yield_beta"], dtype=float),
        yield_alpha=float(payload["yield_alpha"]),
        fert_rate=float(payload["fert_rate"]),
        dev_rate=float(payload["dev_rate"]),
        transfer_beta=np.asarray(payload["transfer_beta"], dtype=float),
        p_bank_given_surplus=float(payload["p_bank_given_surplus"]),
        p_continue_fail_nobank=float(payload["p_continue_fail_nobank"]),
        p_continue_fail_bank=float(payload["p_continue_fail_bank"]),
        p_continue_birth_nobank=float(payload["p_continue_birth_nobank"]),
        p_continue_birth_bank=float(payload["p_continue_birth_bank"]),
        p_use_bank=float(payload["p_use_bank"]),
        fet_transfer_mean=float(payload["fet_transfer_mean"]),
        fet_live_birth_rate=float(payload["fet_live_birth_rate"]),
        p_fresh_transfer=float(payload["p_fresh_transfer"]),
    )
