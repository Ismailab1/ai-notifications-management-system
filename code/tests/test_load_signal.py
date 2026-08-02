"""Unit tests for load_signal.py.

Run directly:  python tests/test_load_signal.py
"""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from router.data_loader import Dataset
from router.load_signal import LoadIndex


def _empty_dataset() -> Dataset:
    return Dataset(
        messages=[], sample_messages=[], users={}, groups={}, group_members={},
        business_accounts={}, user_business_history={},
        message_history=[], message_events={},
        images={}, voice_notes={}, daily_notification_summary=[],
    )


def _row(mid, user, conv, created_at, sender="", group="", business=""):
    return {
        "message_id": mid, "user_id": user, "conversation_type": conv,
        "created_at": created_at, "sender_user_id": sender, "group_id": group, "business_id": business,
    }


def test_burst_detected_within_window_same_category():
    batch = [
        _row("m1", "u_x", "business", "2026-07-20 09:00", business="biz_1"),
        _row("m2", "u_x", "business", "2026-07-20 09:30", business="biz_2"),
        _row("m3", "u_x", "business", "2026-07-20 10:00", business="biz_1"),  # the message being scored
    ]
    ds = _empty_dataset()
    idx = LoadIndex(ds, batch)
    signal = idx.assess(batch[2], window_hours=3.0)
    assert signal.recent_same_category_count == 2
    assert signal.is_burst  # default threshold is 2


def test_messages_outside_window_are_not_counted():
    batch = [
        _row("m1", "u_x", "business", "2026-07-20 02:00", business="biz_1"),  # 8h before -- outside a 3h window
        _row("m2", "u_x", "business", "2026-07-20 10:00", business="biz_1"),
    ]
    ds = _empty_dataset()
    idx = LoadIndex(ds, batch)
    signal = idx.assess(batch[1], window_hours=3.0)
    assert signal.recent_same_category_count == 0


def test_different_category_does_not_count_toward_burst():
    batch = [
        _row("m1", "u_x", "personal", "2026-07-20 09:00", sender="u_friend"),
        _row("m2", "u_x", "personal", "2026-07-20 09:15", sender="u_friend"),
        _row("m3", "u_x", "business", "2026-07-20 09:30", business="biz_1"),  # the message being scored
    ]
    ds = _empty_dataset()
    idx = LoadIndex(ds, batch)
    signal = idx.assess(batch[2], window_hours=3.0)
    # Two personal messages happened nearby, but this is a business message --
    # they must not count toward its same-category burst. This is the core of
    # the "narrow" design: a personal message sitting inside a business wave
    # is invisible to this signal, and vice versa.
    assert signal.recent_same_category_count == 0


def test_same_thread_activity_detected_independent_of_category_count():
    batch = [
        _row("m1", "u_x", "group", "2026-07-20 09:00", group="g1"),
        _row("m2", "u_x", "group", "2026-07-20 09:45", group="g1"),  # scored message, same group as m1
    ]
    ds = _empty_dataset()
    idx = LoadIndex(ds, batch)
    signal = idx.assess(batch[1], window_hours=3.0)
    assert signal.has_recent_thread_activity
    assert signal.recent_same_thread_count == 1


def test_load_signal_object_exposes_no_suppression_field():
    # Structural check on the "narrow" design promise: the dataclass has
    # nothing named/shaped like a suppression flag, only count/burst/thread
    # -activity fields that pipeline.py and the prompt only ever use to
    # support bundling or promotion, never to downgrade a message.
    from router.load_signal import LoadSignal
    field_names = set(LoadSignal.__dataclass_fields__.keys())
    assert not any("suppress" in f.lower() or "downgrade" in f.lower() for f in field_names), field_names


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
