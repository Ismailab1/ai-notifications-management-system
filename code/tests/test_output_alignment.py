"""Regression test for the row-alignment bug: dataset/messages.csv is the
organizer's original (shuffled) row order, and the grader compares
output.csv against it POSITIONALLY, not by joining on message_id. A prior
version of main.py sorted messages by message_id before writing, which
silently misaligned ~every row against the wrong message on the real
(shuffled) input file.

Run directly:  python tests/test_output_alignment.py
"""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from main import assemble_output_rows


def _msg(mid: str) -> dict:
    return {"message_id": mid}


def _row(mid: str, own_id: str | None = None) -> dict:
    # own_id lets a test deliberately construct a row whose own message_id
    # field disagrees with the dict key it's stored under, to simulate
    # upstream corruption independent of this function's own bookkeeping.
    return {"message_id": own_id if own_id is not None else mid, "action": "digest"}


def test_write_order_matches_original_file_order_not_sorted_order():
    # Deliberately NOT in sorted order -- mirrors the real dataset/messages.csv,
    # which is the organizer's shuffled row order, not msg_001, msg_002, ...
    dataset_messages = [_msg("msg_003"), _msg("msg_001"), _msg("msg_002")]
    rows_by_id = {
        "msg_001": _row("msg_001"),
        "msg_002": _row("msg_002"),
        "msg_003": _row("msg_003"),
    }
    rows = assemble_output_rows(dataset_messages, rows_by_id)
    ids = [r["message_id"] for r in rows]
    assert ids == ["msg_003", "msg_001", "msg_002"], (
        f"expected original file order, got {ids} -- looks sorted, which is exactly the bug"
    )


def test_limited_run_preserves_original_order_restricted_to_processed_ids():
    # --limit N smoke-test case: only a subset of dataset.messages was
    # actually processed. The write order must still follow the original
    # file's relative order for just those ids, not the full file.
    dataset_messages = [_msg("msg_003"), _msg("msg_001"), _msg("msg_002"), _msg("msg_004")]
    rows_by_id = {"msg_001": _row("msg_001"), "msg_004": _row("msg_004")}
    rows = assemble_output_rows(dataset_messages, rows_by_id)
    ids = [r["message_id"] for r in rows]
    assert ids == ["msg_001", "msg_004"]


def test_misaligned_row_trips_the_assertion():
    # A guardrail that never fires isn't a guardrail: construct a row stored
    # under the "msg_002" key whose own message_id field says "msg_999" --
    # simulating a row getting its identity mangled upstream, independent of
    # this function's own dict bookkeeping. This must raise, not write.
    dataset_messages = [_msg("msg_001"), _msg("msg_002"), _msg("msg_003")]
    rows_by_id = {
        "msg_001": _row("msg_001"),
        "msg_002": _row("msg_002", own_id="msg_999"),  # corrupted
        "msg_003": _row("msg_003"),
    }
    try:
        assemble_output_rows(dataset_messages, rows_by_id)
        raised = False
    except RuntimeError as e:
        raised = True
        msg = str(e)
        assert "msg_002" in msg, f"expected the expected id in the error, got: {msg}"
        assert "msg_999" in msg, f"expected the actual (corrupted) id in the error, got: {msg}"
    assert raised, "expected assemble_output_rows to raise on a misaligned row, but it didn't"


def test_full_run_shape_matches_real_dataset_ordering_pattern():
    # Same shape as the real bug report: file order is NOT sorted order.
    dataset_messages = [_msg("msg_023"), _msg("msg_091"), _msg("msg_090"), _msg("msg_048")]
    rows_by_id = {mid: _row(mid) for mid in ("msg_023", "msg_091", "msg_090", "msg_048")}
    rows = assemble_output_rows(dataset_messages, rows_by_id)
    ids = [r["message_id"] for r in rows]
    sorted_ids = sorted(ids)
    assert ids != sorted_ids, "output order should not collapse to sorted order"
    assert ids == ["msg_023", "msg_091", "msg_090", "msg_048"]


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
