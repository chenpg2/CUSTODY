"""The four objects a user of CUSTODY actually touches.

These test the facade's own promises, not the mechanisms underneath, which
tests/test_core.py covers. What is asserted here is what the README claims:
fitting produces a process, releasing produces a certified cohort, a private
release is priced by an accountant rather than by its caller, an edited payload
is refused, and a receiver reaches those conclusions without being told which
sender to distrust.
"""

from __future__ import annotations

import pandas as pd
import pytest

from custody import (
    BudgetExhausted,
    Centre,
    Delivery,
    Privacy,
    Receiver,
    Release,
    Verification,
)

# What the private tests release under. A centre releases the way it was fitted,
# so the ones that release privately are fitted with it too.
PRIVACY = Privacy(epsilon=1.0, cap=5.0)


class TestCentre:
    def test_fit_counts_the_families_it_will_price_privacy_against(self, cohort):
        fresh, fet = cohort
        centre = Centre.fit(fresh, fet, name="Centre_1")
        assert centre.name == "Centre_1"
        assert centre.n_families == fresh["pid"].nunique()

    def test_a_rollout_conserves_the_ledger_with_no_repair_step(self, cohort):
        centre = Centre.fit(*cohort)
        report = __import__("custody").check_ledger(centre.rollout(200, seed=3))
        assert report.clean and report.total == 0

    def test_the_same_seed_gives_the_same_cohort(self, cohort):
        centre = Centre.fit(*cohort)
        pd.testing.assert_frame_equal(centre.rollout(150, seed=11), centre.rollout(150, seed=11))

    def test_a_plain_centre_has_spent_nothing(self, cohort):
        assert Centre.fit(*cohort).spent == 0.0


class TestRelease:
    def test_a_plain_release_claims_no_guarantee(self, cohort):
        release = Centre.fit(*cohort).release(200, seed=5)
        assert isinstance(release, Release)
        assert release.epsilon is None and not release.private

    def test_a_release_verifies_against_its_own_certificate(self, cohort):
        release = Centre.fit(*cohort).release(200, seed=5)
        verification = release.verify()
        assert isinstance(verification, Verification)
        assert verification.accepted and bool(verification) and verification.failed == []

    def test_an_edited_cohort_is_refused_on_the_digest_and_the_ledger(self, cohort):
        release = Centre.fit(*cohort).release(200, seed=5)
        release.cohort.loc[release.cohort.index[0], "transfer_embryo_num"] += 5
        verification = release.verify()
        assert not verification.accepted
        assert "cohort_digest" in verification.failed
        assert "ledger_invariants" in verification.failed

    def test_the_ledger_check_is_recomputed_on_the_object(self, cohort):
        release = Centre.fit(*cohort).release(200, seed=5)
        assert release.check().clean


class TestPrivacy:
    def test_the_epsilon_is_produced_not_declared(self, cohort):
        centre = Centre.fit(*cohort, name="Centre_1", privacy=PRIVACY)
        release = centre.release(200, seed=5, privacy=PRIVACY)
        assert release.private
        # The accountant's number, which is not the epsilon that was asked for.
        assert release.epsilon is not None
        assert release.epsilon != pytest.approx(1.0)
        assert 0.0 < release.epsilon < 1.0

    def test_spend_accumulates_across_releases(self, cohort):
        centre = Centre.fit(*cohort, name="Centre_1", privacy=PRIVACY)
        first = centre.release(200, seed=5, privacy=PRIVACY)
        second = centre.release(200, seed=6, privacy=PRIVACY)
        assert second.epsilon > first.epsilon
        assert centre.spent == pytest.approx(second.epsilon)

    def test_a_release_past_the_cap_is_refused_and_costs_nothing(self, cohort):
        centre = Centre.fit(*cohort, name="Centre_1", privacy=Privacy())
        with pytest.raises(BudgetExhausted):
            centre.release(200, seed=5, privacy=Privacy(epsilon=1.0, cap=1e-6))
        assert centre.spent == 0.0

    def test_a_private_release_still_conserves_the_ledger(self, cohort):
        centre = Centre.fit(*cohort, name="Centre_1", privacy=PRIVACY)
        release = centre.release(200, seed=5, privacy=PRIVACY)
        assert release.check().clean and release.verify().accepted

    def test_contribution_bounding_keeps_every_family(self, cohort):
        """Until 1.1.0 the fit counted the cycles capping dropped as the families."""
        fresh, fet = cohort
        families = pd.concat([fresh["pid"], fet["pid"]]).nunique()
        bounded = Centre.fit(fresh, fet, privacy=Privacy(epsilon=1.0, max_cycles=1))
        assert bounded.n_families == Centre.fit(fresh, fet).n_families == families


class TestReleaseMode:
    """A centre releases the way it was fitted, and a mismatch changes nothing."""

    def test_a_plain_fit_cannot_release_privately(self, cohort):
        centre = Centre.fit(*cohort, name="Centre_1")
        with pytest.raises(ValueError, match="without a contribution cap"):
            centre.release(200, seed=5, privacy=PRIVACY)
        assert centre.spent == 0.0
        assert not centre.release(200, seed=5).private

    def test_a_private_centre_cannot_release_plainly(self, cohort):
        """Until 1.1.0 this silently released under the previous release's privacy."""
        centre = Centre.fit(*cohort, name="Centre_1", privacy=PRIVACY)
        centre.release(200, seed=5, privacy=PRIVACY)
        spent = centre.spent
        with pytest.raises(ValueError, match="fitted for private release"):
            centre.release(200, seed=6)
        assert centre.spent == spent

    def test_a_bound_tighter_than_the_fit_is_refused(self, cohort):
        centre = Centre.fit(*cohort, name="Centre_1", privacy=Privacy(max_cycles=6))
        with pytest.raises(ValueError, match="max_cycles of at least 6"):
            centre.release(200, seed=5, privacy=Privacy(max_cycles=3, cap=5.0))
        assert centre.spent == 0.0

    def test_an_empty_release_is_refused_before_it_is_priced(self, cohort):
        centre = Centre.fit(*cohort, name="Centre_1", privacy=PRIVACY)
        with pytest.raises(ValueError, match="at least one patient"):
            centre.release(0, seed=5, privacy=PRIVACY)
        assert centre.spent == 0.0 and centre.as_node().budget is None

    def test_a_refused_first_release_fixes_no_cap(self, cohort):
        """Nothing was released, so the next release may still choose its ceiling."""
        centre = Centre.fit(*cohort, name="Centre_1", privacy=PRIVACY)
        with pytest.raises(BudgetExhausted):
            centre.release(200, seed=5, privacy=Privacy(epsilon=1.0, cap=1e-6))
        assert centre.release(200, seed=5, privacy=PRIVACY).private

    def test_a_refused_re_ceiling_leaves_the_settings_as_they_were(self, cohort):
        centre = Centre.fit(*cohort, name="Centre_1", privacy=PRIVACY)
        first = centre.release(200, seed=5, privacy=PRIVACY)
        with pytest.raises(ValueError, match="cannot be re-ceilinged"):
            centre.release(200, seed=6, privacy=Privacy(epsilon=9.0, cap=0.001))
        assert centre.as_node().dp == PRIVACY._config()
        assert centre.spent == pytest.approx(first.epsilon)


class TestReceiver:
    def test_it_accepts_the_honest_and_refuses_the_tampered(self, cohort):
        fresh, fet = cohort
        senders = [Centre.fit(fresh, fet, name=f"Centre_{i}") for i in (1, 2, 3)]
        releases = [c.release(200, seed=10 + i) for i, c in enumerate(senders)]
        releases[1].cohort.loc[releases[1].cohort.index[0], "transfer_embryo_num"] += 9

        delivery = Receiver(Centre.fit(fresh, fet, name="Centre_9")).receive(releases)

        assert isinstance(delivery, Delivery)
        assert set(delivery.accepted) == {"Centre_1", "Centre_3"}
        assert set(delivery.rejected) == {"Centre_2"}

    def test_the_replayed_cohort_is_ledger_clean(self, cohort):
        fresh, fet = cohort
        senders = [Centre.fit(fresh, fet, name=f"Centre_{i}") for i in (1, 2)]
        delivery = Receiver(Centre.fit(fresh, fet, name="Centre_9"), replay_patients=200).receive(
            [c.release(150, seed=20 + i) for i, c in enumerate(senders)]
        )
        assert delivery.clean and delivery.cohort is not None

    def test_it_is_never_told_which_sender_to_distrust(self, cohort):
        """The refusal comes from recomputation, not from a hint."""
        fresh, fet = cohort
        good = Centre.fit(fresh, fet, name="Centre_1").release(150, seed=31)
        bad = Centre.fit(fresh, fet, name="Centre_2").release(150, seed=32)
        bad.cohort.loc[bad.cohort.index[0], "transfer_embryo_num"] += 9

        delivery = Receiver(Centre.fit(fresh, fet, name="Centre_9")).receive([good, bad])

        assert delivery.rejected["Centre_2"]
        assert "Centre_2" not in delivery.accepted


class TestBudgetCeiling:
    """The cap is the centre's ceiling, and it cannot move underfoot."""

    def test_a_changed_cap_is_refused_rather_than_ignored(self, cohort):
        centre = Centre.fit(*cohort, name="Centre_1", privacy=PRIVACY)
        centre.release(200, seed=5, privacy=PRIVACY)
        with pytest.raises(ValueError, match="cannot be re-ceilinged"):
            centre.release(200, seed=6, privacy=Privacy(epsilon=1.0, cap=0.001))

    def test_the_same_cap_keeps_releasing(self, cohort):
        centre = Centre.fit(*cohort, name="Centre_1", privacy=PRIVACY)
        first = centre.release(200, seed=5, privacy=PRIVACY)
        second = centre.release(200, seed=6, privacy=PRIVACY)
        assert second.epsilon > first.epsilon

    def test_a_fresh_centre_enforces_a_tight_cap_from_the_first_release(self, cohort):
        centre = Centre.fit(*cohort, name="Centre_1", privacy=Privacy())
        with pytest.raises(BudgetExhausted):
            centre.release(200, seed=5, privacy=Privacy(epsilon=1.0, cap=0.001))
        assert centre.spent == 0.0
