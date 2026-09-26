# CUSTODY

A privacy-preserving exchange for assisted-reproduction cohorts.

What crosses an institutional boundary is a fitted treatment-process model and the synthetic
cohorts rolled out from it, each under a certificate the receiving centre recomputes for itself.
The rollout engine carries the embryo bank in its state and subtracts before it spends, so a
conservation violation is a value the engine cannot write rather than one it repairs afterwards.

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/downloads/)

## Patent Notice

This project is covered by a filed patent application. Application number:
**202610687276.4**.

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
# under privacy is fitted with it, on each family's first K cycles.
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
| `Privacy` | What a release spends and the unit it protects: epsilon, delta, the contribution bound, the cumulative cap |
| `Release` | A synthetic cohort, its certificate, and what it cost. `check`, `verify`, `as_payload` |
| `Receiver` | Verifies, refuses, merges and replays. `receive` returns a `Delivery` |

Run the whole path on data it fabricates itself:

```bash
python examples/quickstart.py
```

```
  Release(centre='Centre_1', 578 cycles, plain)
    ledger clean: True    verified: True
  Release(centre='Centre_2', 512 cycles, epsilon=0.3222)
    asked for epsilon 1.0, accountant charged 0.3222
    ledger clean: True    verified: True
  Release(centre='Centre_3', 559 cycles, plain)  <- one row edited in flight
    verified: False, refused on cohort_digest, ledger_invariants

  receiver accepted ['Centre_1', 'Centre_2']
  receiver rejected ['Centre_3'] on ['cohort_digest', 'ledger_invariants']
  replayed 584 cycles, ledger clean: True
```

The private release's noise is drawn from the operating system's cryptographic source on every
run, so Centre_2's cycle count and the replayed count differ from run to run. The epsilon, the
plain counts and the verdicts do not.

## What the guarantee is, and is not

The accounting unit is the **family**: one patient's trajectory together with her partner's and any
offspring's records. That is the unit these records are actually about, and a privacy unit of one
row does not cover it.

`Privacy(epsilon=...)` is a request. What lands on the certificate is what the accountant produced
by composing the mechanism it actually ran, which is smaller. A release that would take the
cumulative spend past `cap` raises `BudgetExhausted` before any noise is drawn or any budget
charged: an exhausted budget is a refusal, not a quieter answer.

The noise is calibrated to a family contributing at most `max_cycles` cycles, so a centre that
releases under privacy is fitted with it: `Centre.fit(fresh, fet, privacy=...)` keeps each
family's first K cycles, and a private release from a fit without that bound is refused. A centre
releases the way it was fitted. A plain release publishes the fitted process exactly, so a centre
fitted for private release refuses a plain one.

The release mode and the budget belong to one `Centre` object. A second `Centre` fitted on the
same records starts a budget that knows nothing of the first, and a plain release from it publishes
what the first protects. The library cannot see that, so it is the caller's to avoid.

Three things this design does not give you, each stated because a reader could otherwise assume
otherwise:

- **Certificates are unsigned.** A receiver can establish that a payload is internally possible and
  honestly accounted. It cannot establish that a particular institution sent it. An edit that
  breaks no invariant and is then re-certified is accepted.
- **Verification says nothing about calibration.** Two payloads that verify identically can differ
  by an order of magnitude in how well a model fitted on them is calibrated, and the marginals do
  not reveal it. Treat released probabilities as ordinal.
- **Contribution bounding is not neutral.** Keeping a family's first K cycles removes failures
  selectively, because cycle count depends on outcome. The direction is knowable; the size is not
  measured here.

## Layout

```
src/custody/
  __init__.py      the four objects, and re-exports of everything below
  cohort.py        the four ledger invariants and the report
  process.py       fitting the treatment process, and the rollout engine
  privacy.py       family-unit differential privacy, contribution bounding, the budget
  certificate.py   what travels beside a payload, and how a receiver rechecks it
  exchange.py      nodes, payloads, the merge, and the receive path
  _dp.py           Renyi accountant and Gaussian calibration
tests/             66 tests
examples/          the quickstart above
```

Everything the four objects wrap is importable directly, for anyone who wants the plumbing:
`check_ledger`, `fit_kernels`, `rollout_cohort`, `privatise_kernels`, `issue_certificate`,
`verify_certificate`, `emit_payload`, `merge_kernels`, `receive`.

## Provenance

No mechanism here is new, and the paper this repository accompanies names the owner of each. The
physiology cascade, contribution bounding, the Gaussian mechanism and Renyi composition are cited
work; the assembly and the assisted-reproduction instance are ours.

The manuscript's evidence record, which binds every published number to the run that produced it,
is deposited separately and is not in this repository.

## Citation

```bibtex
@unpublished{chen2026custody,
  title  = {CUSTODY: a privacy-preserving exchange releasing verifiable synthetic
            cohorts that conserve the embryo ledger in multi-cycle assisted reproduction},
  author = {Chen, Peigen and Pan, Xinyi and Shi, Juanzi and Jin, Lei and Mao, Yundong and
            Zhang, Cuilian and Fang, Cong and Li, Tingting},
  note   = {Manuscript. Peigen Chen and Xinyi Pan contributed equally.},
  year   = {2026}
}
```

## License

MIT. See `LICENSE`.
