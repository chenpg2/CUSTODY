# CUSTODY

A privacy-preserving exchange for assisted-reproduction cohorts.

What crosses an institutional boundary is a fitted treatment-process model and the synthetic
cohorts rolled out from it, each under a certificate the receiving centre checks for itself.
The rollout engine carries the embryo bank in its state and subtracts before it spends, so a
conservation violation is a value the engine cannot write rather than one it repairs afterwards.

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/downloads/)

## Install

```bash
pip install git+https://github.com/chenpg2/CUSTODY.git
```

Python 3.10 or later. Runtime dependencies are numpy, pandas, scipy, scikit-learn, statsmodels and
pyarrow. No patient data is required for anything in this repository.

## Four objects

```python
from custody import Centre, Privacy, Receiver

# A centre fits its own process. Its records never leave. One that will release
# under privacy is fitted with it, and keeps the sums its releases are derived from.
privacy = Privacy(epsilon=1.0, cap=5.0)
sender = Centre.fit(fresh, fet, name="Centre_1", privacy=privacy)

# It releases a synthetic cohort under a certificate.
release = sender.release(n_patients=500, privacy=privacy)
release.epsilon          # the centre's spend so far, produced by the accountant
release.check().clean    # the embryo ledger, recomputed on the cohort
release.verify().accepted

# Another centre receives it, and is told nothing about whom to distrust.
receiver = Receiver(Centre.fit(own_fresh, own_fet, name="Centre_2"))
delivery = receiver.receive([release])
delivery.accepted, delivery.rejected, delivery.cohort
```

| Object | What it is |
|---|---|
| `Centre` | One centre's fitted treatment process, fitted for plain or for private release. `fit`, `rollout`, `release` |
| `Privacy` | What a release spends and the unit it protects: epsilon, delta, the contribution bound, the cumulative cap, the bank column |
| `Release` | A synthetic cohort, its certificate, and what it cost. `check`, `verify`, `as_payload` |
| `Receiver` | Verifies, refuses, merges and replays. `receive` returns a `Delivery` |

Run the whole path on data it fabricates itself:

```bash
python examples/quickstart.py
```

```
  Release(centre='Centre_1', 598 cycles, plain)
    ledger clean: True    verified: True
  Release(centre='Centre_2', 531 cycles, epsilon=0.3798)
    asked for epsilon 1.0, accountant charged 0.3798
    ledger clean: True    verified: True
  Release(centre='Centre_3', 579 cycles, plain)  <- one row edited in flight
    verified: False, refused on cohort_digest, ledger_invariants

  receiver accepted ['Centre_1', 'Centre_2']
  receiver rejected ['Centre_3'] on ['cohort_digest', 'ledger_invariants']
  replayed 640 cycles, ledger clean: True
```

The private release's noise is drawn from the operating system's cryptographic source on every
run, so Centre_2's cycle count and the replayed count differ from run to run. The epsilon, the
plain counts and the verdicts do not. The quickstart fabricates 20,000 patients, because a private
release needs a cohort of realistic size, and takes about twenty seconds.

## What the guarantee is, and is not

The accounting unit is the **family**: one patient's trajectory together with her partner's and any
offspring's records. That is the unit these records are actually about, and a privacy unit of one
row does not cover it.

`Privacy(epsilon=...)` is a request. What lands on the certificate is what the accountant produced
by composing the mechanism it actually ran, which is smaller. A release that would take the
cumulative spend past `cap` raises `BudgetExhausted` before any noise is drawn or any budget
charged: an exhausted budget is a refusal, not a quieter answer.

What a private centre releases is computed from sums. `Centre.fit(fresh, fet, privacy=...)` keeps
the exact sums of seven groups of each family's first `max_cycles` cycles, with every value
clipped to a public range first. Each release adds Gaussian noise to those sums, at a sensitivity
that follows from the public bounds, and derives every released field from the noised sums, public
constants and randomness that does not depend on the data. The column that records banked embryos
is one of those constants, `Privacy(bank_column=...)`, rather than a choice made from the data.
Releases 1.0.0 and 1.1.0 added noise to fitted parameters instead, with sensitivities that did not
hold for what they released; do not use them for a privacy claim.

A private release needs a cohort of realistic size. On a few hundred families the noise swamps the
sums, and a regression on them cannot be fitted: the release then fails with `PrivateFitError`
rather than falling back to a plain fit, and its charge stands, because the noise was drawn.

A centre releases the way it was fitted. A plain release publishes the fitted process exactly, so
a centre fitted for private release refuses a plain one, and one fitted without privacy keeps no
sums to release privately from. Only `release()` is private. `Centre.rollout()` and a
`Receiver`'s `Delivery` are local views that come from the centre's own fit, and they carry no
guarantee.

The release mode and the budget belong to one `Centre` object. A second `Centre` fitted on the
same records starts a budget that knows nothing of the first, and a plain release from it publishes
what the first protects. The library cannot see that, so it is the caller's to avoid.

Three things this design does not give you, each stated because a reader could otherwise assume
otherwise:

- **Certificates are unsigned.** A receiver can establish that a payload is internally possible
  and that a stated budget comes with a cap and δ that cover it. It cannot establish that the
  noise was drawn as stated, or that a particular institution sent the payload. An edit that
  breaks no invariant and is then re-certified is accepted.
- **Verification says nothing about utility.** Two payloads that verify identically can differ in
  how well a model trained on them ranks patients and how well it is calibrated, and the marginals
  do not reveal it. In the paper, two of five private releases at its published budget ranked
  below chance. Measure what a release is worth on your own task before relying on it.
- **Contribution bounding is not neutral.** Keeping a family's first K cycles removes failures
  selectively, because cycle count depends on outcome. The direction is knowable; the size is not
  measured here.

## Changes in 1.2.0

The privacy layer is the one the paper evaluates. Given the same records, the same settings and
the same noise seed, it releases what the code behind the paper's results released, field for
field.

- New: `private_statistics`, `release_private_kernels`, `zero_noise_kernels`,
  `group_sensitivities`, `PrivateStatistics`, `PrivateFitError`, and the module
  `custody.private_stats`.
- Removed: `privatise_kernels`, and `DPConfig`'s `coefficient_range` and `scalar_range`.
- `Privacy` and `DPConfig` take `bank_column`, by default the `freeze_num` column that
  `LedgerSchema` names; the paper's demonstration centre used `total_freeze_num`. `Node` keeps
  its private sums as `private` and no longer takes `n_families`; `Centre` takes `private`.
- A private release that fails to fit after its charge keeps the charge.
- `Privacy(cap=...)` and `FamilyBudget` refuse a cap that is not a positive number.
- Plain fits and plain releases are unchanged.

## Layout

```
src/custody/
  __init__.py       the four objects, and re-exports of everything below
  cohort.py         the four ledger invariants and the report
  process.py        fitting the treatment process, and the rollout engine
  private_stats.py  the clipped group sums a private release is computed from
  privacy.py        family-unit differential privacy, contribution bounding, the budget
  certificate.py    what travels beside a payload, and how a receiver rechecks it
  exchange.py       nodes, payloads, the merge, and the receive path
  _dp.py            Renyi accountant and Gaussian calibration
tests/              88 tests
examples/           the quickstart above
```

Everything the four objects wrap is importable directly, for anyone who wants the plumbing:
`check_ledger`, `fit_kernels`, `rollout_cohort`, `private_statistics`, `release_private_kernels`,
`issue_certificate`, `verify_certificate`, `emit_payload`, `merge_kernels`, `receive`.

## Provenance

No mechanism here is new, and the paper this repository accompanies names the owner of each. The
physiology cascade, contribution bounding, the Gaussian mechanism and Renyi composition are cited
work; the assembly and the assisted-reproduction instance are ours.

The scripts that ran the paper's experiments, its pre-registration and its evidence record are not
in this repository: they read clinical records that cannot be shared. The paper's Code availability
statement says how editors and reviewers can obtain them.

## Citation

```bibtex
@unpublished{chen2026custody,
  title  = {CUSTODY: a privacy-preserving exchange releasing verifiable synthetic
            cohorts that conserve the embryo ledger in multi-cycle assisted reproduction},
  author = {Chen, Peigen and Pan, Xinyi and Zhao, Xin and Shi, Juanzi and Jin, Lei and
            Mao, Yundong and Zhang, Cuilian and Yang, Xing and Fang, Cong and Li, Tingting},
  note   = {Manuscript. Peigen Chen, Xinyi Pan and Xin Zhao contributed equally.},
  year   = {2026}
}
```

## License

MIT. See `LICENSE`.
