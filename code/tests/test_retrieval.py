"""Unit tests for the sparse-text / thread-context additions in retrieval.py.

Run directly:  python tests/test_retrieval.py
"""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from router.data_loader import Dataset
from router.retrieval import find_thread_context, is_text_sparse


def _empty_dataset(message_history: list[dict] | None = None) -> Dataset:
    return Dataset(
        messages=[], sample_messages=[], users={}, groups={}, group_members={},
        business_accounts={}, user_business_history={},
        message_history=message_history or [], message_events={},
        images={}, voice_notes={}, daily_notification_summary=[],
    )


def test_empty_text_is_sparse():
    assert is_text_sparse("")
    assert is_text_sparse(None)


def test_emoji_only_text_is_sparse():
    # No word characters at all -- exactly the "umbrella emoji on a photo" case.
    assert is_text_sparse("☔")


def test_short_caption_is_sparse():
    assert is_text_sparse("ok cool")  # 2 words, under the default threshold of 4


def test_real_caption_is_not_sparse():
    assert not is_text_sparse("Selling a barely used kurta set, size M")


def test_thread_context_returns_most_recent_first_and_excludes_future():
    history = [
        {"message_id": "h1", "user_id": "u_x", "sender_user_id": "u_y", "conversation_type": "personal",
         "group_id": "", "business_id": "", "created_at": "2026-07-01 09:00", "message_text": "first"},
        {"message_id": "h2", "user_id": "u_x", "sender_user_id": "u_y", "conversation_type": "personal",
         "group_id": "", "business_id": "", "created_at": "2026-07-05 09:00", "message_text": "second"},
        {"message_id": "h3", "user_id": "u_x", "sender_user_id": "u_y", "conversation_type": "personal",
         "group_id": "", "business_id": "", "created_at": "2026-07-20 09:00", "message_text": "after -- should not appear"},
    ]
    ds = _empty_dataset(message_history=history)
    message = {
        "message_id": "current", "user_id": "u_x", "sender_user_id": "u_y",
        "conversation_type": "personal", "group_id": "", "business_id": "",
        "created_at": "2026-07-10 09:00", "message_text": "",
    }
    result = find_thread_context(ds, message)
    ids = [c.message_id for c in result]
    assert ids == ["h2", "h1"], f"expected most-recent-first, excluding anything after the current message, got {ids}"
    assert all(c.source == "recency" for c in result)


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
