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

import anthropic
import httpx

import router.reasoning as reasoning_module
from router.context import Engagement, MessageContext
from router.data_loader import Dataset
from router.reasoning import (
    ALLOWED_ACTIONS,
    ALLOWED_TYPES,
    LlmDecision,
    _call_model,
    _fallback_decision,
    _validate_behavioral_evidence,
    reason_about_message,
)
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


def _full_ctx(user_id: str = "u1", message_id: str = "msg_x") -> MessageContext:
    return MessageContext(
        message={"message_id": message_id, "user_id": user_id, "message_text": "hello there friend"},
        recipient=None, group=None, group_member=None, business=None, business_history=None,
        engagement=Engagement(label="unknown", score=0.5, explicit_negative=False, explanation="no data"),
        extracted_media_text=None, safety=SafetySignals(),
    )


class _FakeTextBlock:
    def __init__(self, text: str) -> None:
        self.type = "text"
        self.text = text


class _FakeResponse:
    def __init__(self, text: str) -> None:
        self.content = [_FakeTextBlock(text)]


class _FakeMessages:
    """Replays a fixed script of return values / exceptions, one per call."""

    def __init__(self, script: list) -> None:
        self._script = list(script)
        self.calls = 0

    def create(self, **kwargs):
        item = self._script[self.calls]
        self.calls += 1
        if isinstance(item, BaseException):
            raise item
        return item


class _FakeClient:
    def __init__(self, script: list) -> None:
        self.messages = _FakeMessages(script)


def _connection_error() -> anthropic.APIConnectionError:
    return anthropic.APIConnectionError(request=httpx.Request("POST", "https://api.anthropic.com/v1/messages"))


def _no_sleep():
    """Context manager-ish helper: monkeypatch time.sleep so retry tests don't actually wait."""
    original = reasoning_module.time.sleep
    reasoning_module.time.sleep = lambda seconds: None
    return original


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


def test_call_model_retries_transient_error_then_succeeds():
    original_sleep = _no_sleep()
    try:
        client = _FakeClient([_connection_error(), _FakeResponse('{"action": "digest"}')])
        result = _call_model(client, [{"role": "user", "content": "hi"}])
        assert result == '{"action": "digest"}'
        assert client.messages.calls == 2
    finally:
        reasoning_module.time.sleep = original_sleep


def test_call_model_raises_after_exhausting_retries():
    original_sleep = _no_sleep()
    try:
        # llm_max_retries=3 -> 4 total attempts, all transient failures.
        script = [_connection_error() for _ in range(reasoning_module.CFG.llm_max_retries + 1)]
        client = _FakeClient(script)
        try:
            _call_model(client, [{"role": "user", "content": "hi"}])
            raised = False
        except anthropic.APIConnectionError:
            raised = True
        assert raised, "expected retries to exhaust and re-raise the last transient error"
        assert client.messages.calls == reasoning_module.CFG.llm_max_retries + 1
    finally:
        reasoning_module.time.sleep = original_sleep


def test_call_model_does_not_retry_non_transient_error():
    # A non-retryable error (e.g. bad request) must fail on the first attempt,
    # not burn through the whole backoff schedule for something that would
    # fail identically every time.
    client = _FakeClient([ValueError("not an anthropic transient error")])
    try:
        _call_model(client, [{"role": "user", "content": "hi"}])
        raised = False
    except ValueError:
        raised = True
    assert raised
    assert client.messages.calls == 1


def test_fallback_decision_is_conservative_and_safe():
    decision = _fallback_decision("some reason")
    assert decision.action == "digest"
    assert decision.message_type == "unknown"
    assert decision.evidence_message_ids == "none"
    assert decision.reason == "some reason"
    assert decision.confidence == reasoning_module.CFG.sparse_no_grounding_confidence_cap


def test_decision_schema_fields_match_llm_decision():
    schema_props = set(reasoning_module._DECISION_SCHEMA["properties"].keys())
    llm_decision_fields = {"message_type", "action", "reason", "confidence", "evidence_message_ids"}
    assert schema_props == llm_decision_fields
    assert set(reasoning_module._DECISION_SCHEMA["required"]) == llm_decision_fields
    assert set(reasoning_module._DECISION_SCHEMA["properties"]["action"]["enum"]) == ALLOWED_ACTIONS
    assert set(reasoning_module._DECISION_SCHEMA["properties"]["message_type"]["enum"]) == ALLOWED_TYPES


def test_reason_about_message_falls_back_when_model_call_exhausts_retries():
    original_sleep = _no_sleep()
    try:
        script = [_connection_error() for _ in range(reasoning_module.CFG.llm_max_retries + 1)]
        client = _FakeClient(script)
        ds = _dataset()
        decision = reason_about_message(client, _full_ctx(), [], ds)
        assert decision.action == "digest"
        assert decision.message_type == "unknown"
        assert "Model call failed" in decision.reason
    finally:
        reasoning_module.time.sleep = original_sleep


def test_reason_about_message_recovers_from_one_malformed_response():
    # First response is not valid JSON; the reformat retry succeeds. Must
    # return the retried decision, not fall back, and must have appended the
    # failed first response as the assistant turn before asking for a fix (so
    # the model sees its own bad output in context, not a stale/empty one).
    good_json = '{"message_type": "personal", "action": "digest", "reason": "routine", "confidence": 0.7, "evidence_message_ids": "none"}'
    client = _FakeClient([_FakeResponse("not json at all"), _FakeResponse(good_json)])
    ds = _dataset()
    decision = reason_about_message(client, _full_ctx(), [], ds)
    assert decision.action == "digest"
    assert decision.message_type == "personal"
    assert decision.reason == "routine"
    assert client.messages.calls == 2


def test_reason_about_message_falls_back_when_reformat_retry_also_malformed():
    client = _FakeClient([_FakeResponse("not json"), _FakeResponse("still not json")])
    ds = _dataset()
    decision = reason_about_message(client, _full_ctx(), [], ds)
    assert decision.action == "digest"
    assert decision.message_type == "unknown"
    assert "could not be parsed" in decision.reason
    assert client.messages.calls == 2


def test_reason_about_message_corrects_unsupported_behavioral_claim():
    # First decision is a mute claiming a dismissal with no supporting event
    # row -- the behavioral-evidence check should trigger one corrective
    # retry, and the corrected (evidence-free) reason should be returned.
    bad_json = '{"message_type": "spam", "action": "mute", "reason": "User dismissed this before.", "confidence": 0.8, "evidence_message_ids": "none"}'
    fixed_json = '{"message_type": "spam", "action": "mute", "reason": "Repetitive low-value spam pattern.", "confidence": 0.75, "evidence_message_ids": "none"}'
    client = _FakeClient([_FakeResponse(bad_json), _FakeResponse(fixed_json)])
    ds = _dataset()
    decision = reason_about_message(client, _full_ctx(), [], ds)
    assert decision.reason == "Repetitive low-value spam pattern."
    assert decision.action == "mute"
    assert client.messages.calls == 2


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
