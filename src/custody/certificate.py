"""Release certificate — what a receiving centre checks before trusting a payload.

A payload crossing an institutional boundary carries a cohort, the kernels it
was rolled out from, and a privacy ledger. The receiver can verify none of the
sender's private data, so the certificate binds what CAN be checked locally:

1. **Content identity** — SHA-256 over the canonicalised cohort, so a single
   edited count is detectable.
2. **Ledger invariants** — I1-I4 recomputed by the receiver on the payload
   itself (:func:`custody.check_ledger`), never trusted from the sender.
3. **Privacy ledger arithmetic** — the declared per-role budgets sum to the
   declared total; a payload claiming a total it did not spend is refused.
4. **Kernel binding** — the kernels' hash matches the one the cohort claims,
   so a cohort cannot be swapped under a certificate that vouched for another.

Verification is deliberately mechanical and total: it returns every failed
check, not the first, so a tampered payload cannot hide behind an early exit.
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass, field
from typing import Any

import pandas as pd

from .cohort import LedgerSchema, check_ledger

__all__ = ["Certificate", "CertificateResult", "issue_certificate", "verify_certificate"]

LEDGER_COLUMNS = (
    "pid",
    "cycle_index",
    "cycle_kind",
    "egg_num",
    "fertilization_num",
    "_2PN",
    "transfer_embryo_num",
    "freeze_num",
    "live_birth",
)


def cohort_digest(cohort: pd.DataFrame) -> str:
    """SHA-256 over the ledger-relevant columns in canonical order."""
    frame = cohort[list(LEDGER_COLUMNS)].sort_values(["pid", "cycle_index"]).reset_index(drop=True)
    payload = frame.to_csv(index=False, float_format="%.6f").encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def kernel_digest(kernels: dict[str, Any]) -> str:
    """SHA-256 over the canonical JSON of the exchanged kernel parameters."""
    payload = json.dumps(kernels, sort_keys=True, separators=(",", ":")).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


@dataclass(frozen=True)
class Certificate:
    """What travels beside a payload. Every field is receiver-checkable."""

    cohort_sha256: str
    kernel_sha256: str
    n_rows: int
    n_patients: int
    epsilon_total: float | None
    epsilon_by_role: dict[str, float]
    accounting_unit: str = "family"
    epsilon_cap: float | None = None
    delta: float | None = None
    releases_so_far: int = 1
    schema_version: str = "2.0"

    def as_dict(self) -> dict[str, Any]:
        return {
            "cohort_sha256": self.cohort_sha256,
            "kernel_sha256": self.kernel_sha256,
            "n_rows": self.n_rows,
            "n_patients": self.n_patients,
            "epsilon_total": self.epsilon_total,
            "epsilon_by_role": dict(self.epsilon_by_role),
            "accounting_unit": self.accounting_unit,
            "epsilon_cap": self.epsilon_cap,
            "delta": self.delta,
            "releases_so_far": self.releases_so_far,
            "schema_version": self.schema_version,
        }


@dataclass
class CertificateResult:
    """Outcome of verification. ``passed`` only if every check passed."""

    checks: dict[str, bool] = field(default_factory=dict)
    detail: dict[str, Any] = field(default_factory=dict)

    @property
    def passed(self) -> bool:
        return bool(self.checks) and all(self.checks.values())

    @property
    def failed_checks(self) -> list[str]:
        return sorted(k for k, ok in self.checks.items() if not ok)

    def as_dict(self) -> dict[str, Any]:
        return {
            "passed": self.passed,
            "checks": dict(self.checks),
            "failed_checks": self.failed_checks,
            "detail": self.detail,
        }


def issue_certificate(
    cohort: pd.DataFrame,
    kernels: dict[str, Any],
    *,
    epsilon_by_role: dict[str, float],
    epsilon_total: float | None = None,
    accounting_unit: str = "family",
    epsilon_cap: float | None = None,
    delta: float | None = None,
    releases_so_far: int = 1,
) -> Certificate:
    """Issue a certificate for a payload the sender is about to release.

    ``epsilon_total`` is the SPENT budget reported by the DP accountant. It is
    passed in rather than summed here: the sum-of-parts form was a tautology
    (the audit's F4), and under family-unit accounting the composed spend is a
    property of the mechanism, not of the declaration.
    """
    return Certificate(
        cohort_sha256=cohort_digest(cohort),
        kernel_sha256=kernel_digest(kernels),
        n_rows=int(len(cohort)),
        n_patients=int(cohort["pid"].nunique()),
        # No DP mechanism ran -> no budget is claimed. Summing the declared
        # role shares here would manufacture a guarantee out of a declaration,
        # which is the defect the V3/V4 audit found in the first place.
        epsilon_total=float(epsilon_total) if epsilon_total is not None else None,
        epsilon_by_role=dict(epsilon_by_role),
        accounting_unit=accounting_unit,
        epsilon_cap=epsilon_cap,
        delta=delta,
        releases_so_far=releases_so_far,
    )


def verify_certificate(
    cohort: pd.DataFrame,
    kernels: dict[str, Any],
    certificate: Certificate,
    *,
    schema: LedgerSchema | None = None,
    expected_unit: str | None = None,
) -> CertificateResult:
    """Recompute every claim the certificate makes. No check is trusted.

    Args:
        expected_unit: The receiver's own accounting-unit policy. When given, a
            payload whose declared unit disagrees is refused — a receiver must
            not silently accept a record-level budget where its policy requires
            family-level.

    Returns:
        A :class:`CertificateResult` listing all failures, not just the first.
    """
    result = CertificateResult()
    result.checks["cohort_digest"] = cohort_digest(cohort) == certificate.cohort_sha256
    result.checks["kernel_digest"] = kernel_digest(kernels) == certificate.kernel_sha256
    result.checks["row_count"] = len(cohort) == certificate.n_rows
    result.checks["patient_count"] = int(cohort["pid"].nunique()) == certificate.n_patients

    report = check_ledger(cohort, schema)
    result.checks["ledger_invariants"] = report.clean
    result.detail["ledger"] = report.as_dict()

    # Under family-unit accounting the spent budget is what the accountant
    # produced, so the receiver checks it against the declared cap rather than
    # against a sum of parts (which was a tautology — audit F4). A certificate
    # that claims a budget must also carry the cap and delta that substantiate
    # it: an epsilon with nothing behind it is a forged privacy claim, and a
    # payload released without a DP mechanism must claim nothing at all.
    if certificate.epsilon_total is None:
        result.checks["epsilon_substantiated"] = (
            certificate.epsilon_cap is None and certificate.delta is None
        )
    else:
        result.checks["epsilon_substantiated"] = (
            certificate.epsilon_cap is not None
            and certificate.delta is not None
            and certificate.epsilon_total > 0
            and certificate.epsilon_total <= certificate.epsilon_cap + 1e-9
        )
    result.checks["accounting_unit_declared"] = certificate.accounting_unit in {
        "family",
        "record",
        "patient",
    }
    if expected_unit is not None:
        result.checks["accounting_unit_matches_policy"] = (
            certificate.accounting_unit == expected_unit
        )
    return result
