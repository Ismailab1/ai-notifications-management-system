"""Unit tests for router/reasoning.py's behavioral-evidence semantic
invariant: a `reason` that claims this recipient dismissed/reported/muted
something before must be grounded in a real message_events.csv row for
that recipient showing that action -- not a similar-looking row, not a
different recipient's reaction, and not an unsupported claim with no
evidence cited at all.

Run directly:  python tests/test_reasoning.py
"""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from router.context import MessageContext
from router.data_loader import Dataset
from router.reasoning import LlmDecision, _validate_behavioral_evidence
from router.safety import SafetySignals


def _dataset(message_events: dict | None = None) -> Dataset:
    return Dataset(
        messages=[], sample_messages=[], users={}, groups={}, group_members={},
        business_accounts={}, user_business_history={}, message_history=[],
        message_events=message_events or {}, images={}, voice_notes={}, daily_notification_summary=[],
    )


def _ctx(user_id: str) -> MessageContext:
    return MessageContext(
        message={"user_id": user_id}, recipient=None, group=None, group_member=None,
        business=None, business_history=None, engagement=None, extracted_media_text=None,
        safety=SafetySignals(),
    )


def _decision(reason: str, evidence_message_ids: str = "none", action: str = "mute") -> LlmDecision:
    return LlmDecision(
        message_type="spam", action=action, reason=reason, confidence=0.8,
        evidence_message_ids=evidence_message_ids,
    )


def test_claim_grounded_in_real_event_passes():
    ds = _dataset({("u1", "m1"): {"notification_dismissed": "1"}})
    decision = _decision("User dismissed this before.", evidence_message_ids="m1")
    assert _validate_behavioral_evidence(decision, _ctx("u1"), ds) is None


def test_claim_not_matching_cited_events_fails():
    # m1 exists in message_events but the dismissed flag is "0" -- the
    # recipient opened it, didn't dismiss it. Citing it to support a
    # "dismissed" claim is exactly the invariant violation to catch.
    ds = _dataset({("u1", "m1"): {"notification_dismissed": "0", "message_opened": "1"}})
    decision = _decision("User dismissed this before.", evidence_message_ids="m1")
    violation = _validate_behavioral_evidence(decision, _ctx("u1"), ds)
    assert violation is not None
    assert "notification_dismissed" in violation


def test_claim_with_no_evidence_cited_fails():
    ds = _dataset({("u1", "m1"): {"notification_dismissed": "1"}})
    decision = _decision("User dismissed this before.", evidence_message_ids="none")
    violation = _validate_behavioral_evidence(decision, _ctx("u1"), ds)
    assert violation is not None
    assert "none" in violation.lower()


def test_no_behavior_claim_always_passes():
    ds = _dataset()
    decision = LlmDecision(
        message_type="promotion", action="digest", reason="Routine promo, low priority for this user.",
        confidence=0.7, evidence_message_ids="none",
    )
    assert _validate_behavioral_evidence(decision, _ctx("u1"), ds) is None


def test_reported_claim_checked_against_correct_field():
    # "reported" must check message_reported specifically, not just any
    # negative-looking field (e.g. dismissed alone shouldn't satisfy it).
    ds = _dataset({("u1", "m1"): {"notification_dismissed": "1", "message_reported": "0"}})
    decision = _decision("User reported this content previously.", evidence_message_ids="m1")
    violation = _validate_behavioral_evidence(decision, _ctx("u1"), ds)
    assert violation is not None
    assert "message_reported" in violation


def test_non_mute_action_is_never_checked():
    # Regression: a notify decision that mentions "not muted before" as
    # context for why it ISN'T being suppressed must never be touched by
    # this check -- it's scoped to mute decisions specifically, since that's
    # the invariant it exists to enforce (an unsupported claim excusing a
    # suppression). A real doctor-appointment-reschedule message with this
    # exact shape (notify + a behavior word in the reason) was previously
    # getting incorrectly flagged and degraded to a low-confidence fallback.
    ds = _dataset()  # no events at all -- would fail if action were checked
    decision = _decision(
        "Urgent same-day request; recipient has not muted this sender before.",
        evidence_message_ids="none", action="notify",
    )
    assert _validate_behavioral_evidence(decision, _ctx("u1"), ds) is None


def test_negated_claim_is_not_treated_as_a_positive_claim():
    # "has NOT been muted" is the opposite claim from "was muted" -- must not
    # demand evidence proving the negated (unstated) claim.
    ds = _dataset()  # empty on purpose: a positive-claim check would fail here
    decision = _decision(
        "Scam pattern is strong enough to mute even though this sender has never been reported before.",
        evidence_message_ids="none",
    )
    assert _validate_behavioral_evidence(decision, _ctx("u1"), ds) is None


def test_claim_grounded_via_a_different_cited_id_still_passes():
    # Multiple evidence ids cited; only one needs to support the claimed
    # behavior for the recipient (matches how the field is actually used --
    # evidence lists can include ids that support other parts of the reason).
    ds = _dataset({
        ("u1", "m1"): {"notification_dismissed": "0", "message_opened": "1"},
        ("u1", "m2"): {"notification_dismissed": "1"},
    })
    decision = _decision("User dismissed similar messages before.", evidence_message_ids="m1;m2")
    assert _validate_behavioral_evidence(decision, _ctx("u1"), ds) is None


def _run_all():
    tests = [obj for name, obj in globals().items() if name.startswith("test_") and callable(obj)]
    passed = failed = 0
    for t in tests:
        try:
            t()
            print(f"PASS  {t.__name__}")
            passed += 1
        except AssertionError as e:
            print(f"FAIL  {t.__name__}: {e}")
            failed += 1
    print(f"\n{passed} passed, {failed} failed")
    if failed:
        sys.exit(1)


if __name__ == "__main__":
    _run_all()
