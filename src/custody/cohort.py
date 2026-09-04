"""Embryo-ledger invariants — the physics an ART cohort cannot violate.

A cohort of ART cycles is not a bag of rows: embryos are physical objects,
created in a fresh cycle, banked, and consumed at most once by a later frozen
transfer. Four invariants follow, and they are checkable on any cohort —
real, synthetic, or tampered — without reference to how it was produced:

``I1_stage_monotone``
    Within a cycle, each cascade stage is a subset of the previous one:
    fertilised ≤ oocytes, 2PN ≤ fertilised, and transferred + banked ≤ 2PN.

``I2_balance``
    Per patient, embryos banked minus embryos withdrawn equals the closing
    stock, and the stock is never negative at any point in the sequence.

``I3_no_orphan_consumption``
    A frozen transfer consumes stock that an earlier cycle of the SAME patient
    actually banked. Consuming from an empty bank is the defect the whole
    ledger exists to make unrepresentable.

``I4_precedence``
    Withdrawals follow the deposits they draw on in time; a cycle sequence is
    ordered and a bank cannot lend before it holds.

The checker returns per-invariant violation counts and the offending rows, so
a generator can be scored (V1) and a payload can be refused (V2).
"""

from __future__ import annotations

from dataclasses import dataclass, field

import pandas as pd

__all__ = ["LedgerReport", "LedgerSchema", "check_ledger"]


@dataclass(frozen=True)
class LedgerSchema:
    """Column names a cohort must supply for the invariants to be checkable."""

    patient: str = "pid"
    sequence: str = "cycle_index"
    kind: str = "cycle_kind"
    fresh_label: str = "fresh"
    fet_label: str = "fet"
    oocytes: str = "egg_num"
    fertilised: str = "fertilization_num"
    embryos: str = "_2PN"
    transferred: str = "transfer_embryo_num"
    banked: str = "freeze_num"


@dataclass
class LedgerReport:
    """Violation counts per invariant, with the offending row indices."""

    counts: dict[str, int] = field(default_factory=dict)
    offenders: dict[str, list[int]] = field(default_factory=dict)
    n_rows: int = 0
    n_patients: int = 0

    @property
    def total(self) -> int:
        return int(sum(self.counts.values()))

    @property
    def clean(self) -> bool:
        return self.total == 0

    def as_dict(self) -> dict[str, object]:
        return {
            "n_rows": self.n_rows,
            "n_patients": self.n_patients,
            "counts": dict(self.counts),
            "violation_rate": (self.total / self.n_rows) if self.n_rows else 0.0,
            "clean": self.clean,
            "offenders_head": {k: v[:20] for k, v in self.offenders.items() if v},
        }


def _record(report: LedgerReport, name: str, offending: pd.Index | list[int]) -> None:
    idx = [int(i) for i in offending]
    report.counts[name] = len(idx)
    report.offenders[name] = idx


def check_ledger(cohort: pd.DataFrame, schema: LedgerSchema | None = None) -> LedgerReport:
    """Check I1-I4 on a cohort. Rows are ordered by (patient, sequence).

    Args:
        cohort: One row per cycle, carrying the schema's columns.
        schema: Column naming. Defaults to :class:`LedgerSchema`.

    Returns:
        A :class:`LedgerReport`. ``clean`` is True only if every invariant holds.
    """
    s = schema or LedgerSchema()
    frame = cohort.sort_values([s.patient, s.sequence]).reset_index(drop=True)
    report = LedgerReport(n_rows=len(frame), n_patients=int(frame[s.patient].nunique()))
    fresh = frame[s.kind] == s.fresh_label

    # I1 - within-cycle cascade monotonicity (fresh cycles only; FET has no cascade).
    f = frame[fresh]
    bad = f.index[
        (f[s.fertilised] > f[s.oocytes])
        | (f[s.embryos] > f[s.fertilised])
        | (f[s.transferred] + f[s.banked] > f[s.embryos])
    ]
    _record(report, "I1_stage_monotone", bad)

    # I2/I3/I4 - walk each patient's sequence, holding the bank.
    negative, orphan, precedence = [], [], []
    for _pid, rows in frame.groupby(s.patient, sort=False):
        stock = 0.0
        deposited_yet = False
        for idx, row in rows.iterrows():
            is_fresh = row[s.kind] == s.fresh_label
            if is_fresh:
                stock += float(row[s.banked])
                if float(row[s.banked]) > 0:
                    deposited_yet = True
                continue
            draw = float(row[s.transferred])
            if draw > 0 and not deposited_yet:
                precedence.append(idx)  # withdrawal before any deposit exists
            if draw > stock + 1e-9:
                orphan.append(idx)  # consumes stock nobody banked
                stock = 0.0
                continue
            stock -= draw
            if stock < -1e-9:
                negative.append(idx)
    _record(report, "I2_balance", negative)
    _record(report, "I3_no_orphan_consumption", orphan)
    _record(report, "I4_precedence", precedence)
    return report
