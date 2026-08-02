"""Unit tests for router/context.py -- compute_engagement's small nudges on
top of the primary open/reply/dismiss counts.

Run directly:  python tests/test_context.py
"""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from router.context import compute_engagement
from router.data_loader import Dataset


def _empty_dataset(
    message_history: list[dict] | None = None,
    message_events: dict | None = None,
    user_business_history: dict | None = None,
) -> Dataset:
    return Dataset(
        messages=[],
        sample_messages=[],
        users={},
        groups={},
        group_members={},
        business_accounts={},
        user_business_history=user_business_history or {},
        message_history=message_history or [],
        message_events=message_events or {},
        images={},
        voice_notes={},
        daily_notification_summary=[],
    )


def _personal_history(sender="u_y", recipient="u_x", n=3):
    return [
        {
            "message_id": f"h{i}", "user_id": recipient, "sender_user_id": sender,
            "conversation_type": "personal", "group_id": "", "business_id": "",
            "message_text": "hi", "media_type": "", "media_id": "", "forwarded_count": "0",
        }
        for i in range(n)
    ]


def test_fast_reaction_time_nudges_engagement_score_up():
    # opened=2, dismissed=1 keeps the base score at 0.667 (below the 1.0
    # ceiling) so the +0.05 nudge is actually visible -- an all-opened mix
    # already caps at score=1.0, which would hide the nudge entirely.
    history = _personal_history()
    events = {
        ("u_x", "h0"): {"message_opened": "1", "message_replied": "0", "reaction_time_minutes": "3", "notification_dismissed": "0"},
        ("u_x", "h1"): {"message_opened": "1", "message_replied": "0", "reaction_time_minutes": "5", "notification_dismissed": "0"},
        ("u_x", "h2"): {"message_opened": "0", "message_replied": "0", "notification_dismissed": "1"},
    }
    ds = _empty_dataset(message_history=history, message_events=events)
    baseline = compute_engagement(ds, {"conversation_type": "personal", "sender_user_id": "u_y", "user_id": "u_x"})

    # Same counts, but with no reaction_time_minutes recorded at all -- isolates the nudge's effect.
    events_no_rt = {k: {kk: vv for kk, vv in v.items() if kk != "reaction_time_minutes"} for k, v in events.items()}
    ds_no_rt = _empty_dataset(message_history=history, message_events=events_no_rt)
    without_nudge = compute_engagement(ds_no_rt, {"conversation_type": "personal", "sender_user_id": "u_y", "user_id": "u_x"})

    assert baseline.score > without_nudge.score
    assert round(baseline.score - without_nudge.score, 2) == 0.05
    assert "fast" in baseline.explanation


def test_slow_reaction_time_nudges_engagement_score_down():
    history = _personal_history()
    events = {
        ("u_x", "h0"): {"message_opened": "1", "message_replied": "0", "reaction_time_minutes": "95", "notification_dismissed": "0"},
        ("u_x", "h1"): {"message_opened": "1", "message_replied": "0", "reaction_time_minutes": "110", "notification_dismissed": "0"},
        ("u_x", "h2"): {"message_opened": "1", "message_replied": "0", "reaction_time_minutes": "100", "notification_dismissed": "0"},
    }
    ds = _empty_dataset(message_history=history, message_events=events)
    result = compute_engagement(ds, {"conversation_type": "personal", "sender_user_id": "u_y", "user_id": "u_x"})

    events_no_rt = {k: {kk: vv for kk, vv in v.items() if kk != "reaction_time_minutes"} for k, v in events.items()}
    ds_no_rt = _empty_dataset(message_history=history, message_events=events_no_rt)
    without_nudge = compute_engagement(ds_no_rt, {"conversation_type": "personal", "sender_user_id": "u_y", "user_id": "u_x"})

    assert result.score < without_nudge.score
    assert round(without_nudge.score - result.score, 2) == 0.05
    assert "slow" in result.explanation


def test_no_reaction_time_data_leaves_score_unchanged():
    history = _personal_history()
    events = {
        ("u_x", "h0"): {"message_opened": "1", "message_replied": "0", "notification_dismissed": "0"},
        ("u_x", "h1"): {"message_opened": "1", "message_replied": "0", "notification_dismissed": "0"},
    }
    ds = _empty_dataset(message_history=history, message_events=events)
    result = compute_engagement(ds, {"conversation_type": "personal", "sender_user_id": "u_y", "user_id": "u_x"})
    assert "reaction time" not in result.explanation


def test_recent_business_reply_lifts_declining_engagement_to_engaged():
    # 30-day rollup alone reads as declining (low open/reply vs dismissed),
    # but a reply just 2 days before this message should lift it -- recency
    # over a diluted rollup, per the "friends at a restaurant" design.
    bh = {
        ("u_x", "biz_1"): {
            "why_user_knows_account": "past customer",
            "promotions_opted_out_at": "", "allows_promotions": "1",
            "messages_opened_30d": "1", "messages_replied_30d": "0", "messages_dismissed_30d": "4",
            "last_reply_at": "2026-07-28 10:00",
        }
    }
    ds = _empty_dataset(user_business_history=bh)
    message = {
        "conversation_type": "business", "user_id": "u_x", "business_id": "biz_1",
        "created_at": "2026-07-30 10:00",
    }
    result = compute_engagement(ds, message)
    assert result.label == "engaged"
    assert result.score >= 0.75
    assert "recency lifts" in result.explanation


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
