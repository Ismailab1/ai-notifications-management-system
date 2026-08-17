"""Round-trip tests for evaluation/main.py's own row-order detector.

tests/test_output_alignment.py already covers main.py::assemble_output_rows
-- the in-memory logic that decides what order to WRITE output.csv in. This
file covers a different gap: evaluation/main.py::check_row_order_alignment,
the harness's own READ-BACK detector, which nothing previously exercised
against actual files on disk. A bug in the write path and a bug in the
harness's detector are independent failure modes -- the detector could
itself have an off-by-one, a wrong comparison, or silently pass a shuffled
pair -- so it needs its own round-trip test, not just a review of the write
path it's meant to catch regressions in.

Writes deliberately-shuffled synthetic messages.csv / output.csv pairs to a
temp directory and asserts the detector's verdict on each, exactly the way
`python evaluation/main.py --check-output PATH` would be used against a real
submission before it ships.

Run directly:  python tests/test_evaluation_harness.py
"""
from __future__ import annotations

import csv
import sys
import tempfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from evaluation.main import check_row_order_alignment, run_output_checks


def _write_csv(path: Path, fieldnames: list[str], rows: list[dict]) -> None:
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def _output_row(message_id: str) -> dict:
    return {
        "message_id": message_id, "action": "digest", "message_type": "personal",
        "reason": "test row", "confidence": "0.7", "evidence_message_ids": "none",
    }


def test_matching_shuffled_order_passes():
    # messages.csv in the organizer's real shuffled order; output.csv written
    # in that exact same order -- the correct, positionally-aligned case.
    shuffled = ["msg_003", "msg_001", "msg_004", "msg_002"]
    with tempfile.TemporaryDirectory() as tmp:
        tmp_path = Path(tmp)
        messages_path = tmp_path / "messages.csv"
        output_path = tmp_path / "output.csv"
        _write_csv(messages_path, ["message_id"], [{"message_id": mid} for mid in shuffled])
        _write_csv(
            output_path,
            ["message_id", "action", "message_type", "reason", "confidence", "evidence_message_ids"],
            [_output_row(mid) for mid in shuffled],
        )
        warnings = check_row_order_alignment(messages_path, output_path)
        assert warnings == [], f"expected no warnings on matching shuffled order, got: {warnings}"


def test_sorted_output_against_shuffled_input_is_caught():
    # The exact real-world regression this exists to catch: output.csv
    # written in message_id-sorted order while messages.csv (the actual
    # grading reference) stays in the organizer's shuffled order.
    shuffled = ["msg_003", "msg_001", "msg_004", "msg_002"]
    with tempfile.TemporaryDirectory() as tmp:
        tmp_path = Path(tmp)
        messages_path = tmp_path / "messages.csv"
        output_path = tmp_path / "output.csv"
        _write_csv(messages_path, ["message_id"], [{"message_id": mid} for mid in shuffled])
        _write_csv(
            output_path,
            ["message_id", "action", "message_type", "reason", "confidence", "evidence_message_ids"],
            [_output_row(mid) for mid in sorted(shuffled)],
        )
        warnings = check_row_order_alignment(messages_path, output_path)
        assert warnings, "expected a warning when output.csv is sorted but messages.csv is shuffled"
        # shuffled=[msg_003, msg_001, msg_004, msg_002], sorted=[msg_001, msg_002, msg_003, msg_004]
        # -- they diverge at the very first row.
        assert "index 0" in warnings[0], f"expected the mismatch to be caught at the first divergent index, got: {warnings}"


def test_row_count_mismatch_is_caught():
    with tempfile.TemporaryDirectory() as tmp:
        tmp_path = Path(tmp)
        messages_path = tmp_path / "messages.csv"
        output_path = tmp_path / "output.csv"
        _write_csv(messages_path, ["message_id"], [{"message_id": mid} for mid in ["msg_001", "msg_002", "msg_003"]])
        _write_csv(
            output_path,
            ["message_id", "action", "message_type", "reason", "confidence", "evidence_message_ids"],
            [_output_row(mid) for mid in ["msg_001", "msg_002"]],
        )
        warnings = check_row_order_alignment(messages_path, output_path)
        assert warnings, "expected a warning on a row-count mismatch"
        assert "row count mismatch" in warnings[0]


def test_missing_output_file_is_caught():
    with tempfile.TemporaryDirectory() as tmp:
        tmp_path = Path(tmp)
        messages_path = tmp_path / "messages.csv"
        _write_csv(messages_path, ["message_id"], [{"message_id": "msg_001"}])
        warnings = check_row_order_alignment(messages_path, tmp_path / "does_not_exist.csv")
        assert warnings and "does not exist" in warnings[0]


def test_run_output_checks_fails_only_on_row_order_not_on_advisory_collapse():
    # A message_type/action collapse is advisory (printed, doesn't fail the
    # exit code) -- only row-order misalignment is a hard failure. Build a
    # synthetic output.csv that's positionally aligned (passes) but has a
    # 100%-collapsed, non-exempt message_type (would normally warn).
    shuffled = ["msg_001", "msg_002", "msg_003", "msg_004"]
    with tempfile.TemporaryDirectory() as tmp:
        tmp_path = Path(tmp)
        _write_csv(tmp_path / "messages.csv", ["message_id"], [{"message_id": mid} for mid in shuffled])
        rows = [
            {**_output_row(mid), "message_type": "promotion", "action": "mute"}
            for mid in shuffled
        ]
        _write_csv(
            tmp_path / "output.csv",
            ["message_id", "action", "message_type", "reason", "confidence", "evidence_message_ids"],
            rows,
        )
        passed = run_output_checks(tmp_path, tmp_path / "output.csv")
        assert passed is True, "an advisory-only distributional warning must not fail the check"


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
