"""CUSTODY end to end, on a fabricated cohort.

Fit a treatment-process model, roll a synthetic cohort out of it, check the
embryo ledger, certify the release, and verify the certificate the way a
receiving centre would. Nothing here touches patient data: the input cohort is
made up on the spot, and its only job is to carry the column names the
framework expects.

Run it with::

    python examples/custody_quickstart.py
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from custody import Centre, Privacy, Receiver


def fabricate(n_patients: int = 400, seed: int = 0) -> tuple[pd.DataFrame, pd.DataFrame]:
    """A fabricated two-part cohort: fresh cycles, and frozen transfers.

    The schema is the one :class:`custody.LedgerSchema` names. A fresh
    cycle retrieves oocytes, fertilises some, develops some to 2PN, transfers
    some and banks the rest. A frozen transfer spends what an earlier cycle
    banked, which is the link the ledger exists to keep honest.
    """
    rng = np.random.default_rng(seed)
    age = rng.normal(32, 4, n_patients).clip(20, 45)
    eggs = rng.poisson(np.exp(2.6 - 0.03 * (age - 32))).clip(0, 40)
    fertilised = rng.binomial(eggs, 0.75)
    embryos = rng.binomial(fertilised, 0.65)
    transferred = np.minimum(embryos, rng.integers(0, 3, n_patients))
    banked = embryos - transferred

    fresh = pd.DataFrame(
        {
            "pid": [f"P{i:04d}" for i in range(n_patients)],
            "cycle_index": 0,
            "cycle_kind": "fresh",
            # The behaviour fitter walks each family's cycles in date order.
            "visit_date": pd.Timestamp("2024-01-01")
            + pd.to_timedelta(rng.integers(0, 360, n_patients), unit="D"),
            "age_w": age,
            "AF": rng.normal(12, 4, n_patients).clip(1, 40),
            "egg_num": eggs,
            "fertilization_num": fertilised,
            "_2PN": embryos,
            "transfer_embryo_num": transferred,
            "freeze_num": banked,
            "total_freeze_num": banked,
            "live_birth": rng.binomial(1, 0.30 * (transferred > 0)),
        }
    )

    # Frozen transfers, for the patients who banked something.
    returning = fresh[fresh["freeze_num"] > 0].sample(frac=0.45, random_state=seed)
    spent = np.minimum(returning["freeze_num"].to_numpy(), 1)
    fet = pd.DataFrame(
        {
            "pid": returning["pid"].to_numpy(),
            "cycle_index": 1,
            "cycle_kind": "fet",
            "visit_date": returning["visit_date"].to_numpy() + pd.Timedelta(days=120),
            "age_w": returning["age_w"].to_numpy(),
            "AF": returning["AF"].to_numpy(),
            "egg_num": 0,
            "fertilization_num": 0,
            "_2PN": 0,
            "transfer_embryo_num": spent,
            "freeze_num": -spent,
            "total_freeze_num": returning["freeze_num"].to_numpy(),
            "live_birth": rng.binomial(1, 0.28, len(returning)),
        }
    )
    return fresh, fet


def main() -> int:
    fresh, fet = fabricate()
    print(f"input cohort: {len(fresh)} fresh cycles, {len(fet)} frozen transfers\n")

    # Three centres fit their own process. The records never leave.
    senders = [Centre.fit(fresh, fet, name=f"Centre_{i}") for i in (1, 2, 3)]
    for centre in senders:
        print(f"  {centre}")

    # Centre_1 releases plainly. The certificate binds the cohort to the kernels,
    # but no formal guarantee is claimed and the release says so.
    plain = senders[0].release(n_patients=400, seed=42)
    print(f"\n  {plain}")
    print(f"    ledger clean: {plain.check().clean}    verified: {plain.verify().accepted}")

    # Centre_2 releases under family-unit differential privacy. The epsilon on
    # the certificate is what the accountant produced, not what was asked for.
    private = senders[1].release(n_patients=400, seed=7, privacy=Privacy(epsilon=1.0, cap=5.0))
    print(f"  {private}")
    print(f"    asked for epsilon 1.0, accountant charged {private.epsilon:.4f}")
    print(f"    ledger clean: {private.check().clean}    verified: {private.verify().accepted}")

    # Centre_3's payload is altered after it was certified, the way a bug or a
    # malicious relay would alter it.
    tampered = senders[2].release(n_patients=400, seed=99)
    tampered.cohort.loc[tampered.cohort.index[0], "transfer_embryo_num"] += 5
    checked = tampered.verify()
    print(f"  {tampered}  <- one row edited in flight")
    print(f"    verified: {checked.accepted}, refused on {', '.join(checked.failed)}")

    # A fourth centre receives all three. It is told nothing about which to
    # distrust: it recomputes every certificate on what arrived.
    receiver = Receiver(Centre.fit(fresh, fet, name="Centre_9"), replay_patients=500)
    delivery = receiver.receive([plain, private, tampered])
    print(f"\n  receiver accepted {delivery.accepted}")
    print(
        f"  receiver rejected {sorted(delivery.rejected)} "
        f"on {delivery.rejected[sorted(delivery.rejected)[0]]}"
    )
    print(f"  replayed {len(delivery.cohort)} cycles, ledger clean: {delivery.clean}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
