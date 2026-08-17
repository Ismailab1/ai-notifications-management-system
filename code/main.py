#!/usr/bin/env python3
"""Message Notification Router -- entry point.

Usage:
    python main.py                      # full run, writes ../dataset/output.csv
    python main.py --limit 5            # smoke test on the first 5 messages
    python main.py --dry-run            # wiring test, no API key / network needed
    python main.py --dataset-dir PATH --output PATH

See README.md for setup instructions.
"""
from __future__ import annotations

import argparse
import csv
import time
from collections import Counter
from pathlib import Path

from router.config import CFG
from router.data_loader import load_dataset
from router.pipeline import Router

OUTPUT_COLUMNS = ["message_id", "action", "message_type", "reason", "confidence", "evidence_message_ids"]


def assemble_output_rows(dataset_messages: list[dict], rows_by_id: dict[str, dict]) -> list[dict]:
    """Determines the row order output.csv must be written in, and enforces it.

    dataset/messages.csv is the organizer's original (shuffled) row order, and
    the grader compares output.csv against it POSITIONALLY -- not by joining
    on message_id. This function's job is exactly to guarantee the write order
    always matches that original file order (restricted to whatever ids are
    present in rows_by_id, so --limit smoke runs don't spuriously trip it).

    Raises RuntimeError -- refusing to produce rows to write -- if the
    resulting order doesn't line up, naming the first mismatched index and
    the expected vs. actual message_id there.
    """
    processed_ids = set(rows_by_id.keys())
    expected_order = [m["message_id"] for m in dataset_messages if m["message_id"] in processed_ids]
    rows = [rows_by_id[mid] for mid in expected_order]
    actual_order = [r["message_id"] for r in rows]

    if expected_order != actual_order:
        mismatch_index = next(
            i for i, (e, a) in enumerate(zip(expected_order, actual_order)) if e != a
        )
        raise RuntimeError(
            f"output row order does not match dataset/messages.csv's original file order "
            f"at index {mismatch_index}: expected message_id={expected_order[mismatch_index]!r}, "
            f"got {actual_order[mismatch_index]!r}. Refusing to write output.csv -- the grader "
            f"compares rows positionally against the original (unsorted) input file."
        )

    return rows


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--dataset-dir", type=Path, default=CFG.dataset_dir)
    p.add_argument("--output", type=Path, default=None, help="defaults to <dataset-dir>/output.csv")
    p.add_argument("--limit", type=int, default=None, help="only process the first N messages (sorted by message_id)")
    p.add_argument("--dry-run", action="store_true", help="skip real API calls; validates wiring only")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    output_path = args.output or (args.dataset_dir / "output.csv")

    if not args.dry_run:
        CFG.validate()

    print(f"Loading dataset from {args.dataset_dir} ...")
    dataset = load_dataset(args.dataset_dir)
    print(f"  {len(dataset.messages)} messages to route, {len(dataset.message_history)} historical messages loaded")

    client = None
    if not args.dry_run:
        import anthropic

        client = anthropic.Anthropic(api_key=CFG.anthropic_api_key)

    router = Router(dataset, client, dry_run=args.dry_run)

    # Internal processing order: message_id-sorted, purely for deterministic
    # caching/resumability. This must NEVER leak into the write order below --
    # dataset/messages.csv is the organizer's original (shuffled) row order,
    # and the grader compares output.csv against it POSITIONALLY, not by
    # joining on message_id. Writing in sorted order silently misaligns
    # ~every row against the wrong message on a shuffled input file.
    messages = sorted(dataset.messages, key=lambda m: m["message_id"])
    if args.limit:
        messages = messages[: args.limit]

    rows_by_id: dict[str, dict] = {}
    action_counts: Counter = Counter()
    start = time.time()
    for i, message in enumerate(messages, start=1):
        row = router.route(message)
        rows_by_id[row["message_id"]] = row
        action_counts[row["action"]] += 1
        print(f"  [{i}/{len(messages)}] {row['message_id']} -> {row['action']} / {row['message_type']} (conf={row['confidence']})")

    elapsed = time.time() - start

    # Write order must match dataset/messages.csv's original file order --
    # see assemble_output_rows's docstring. Raises rather than writing a
    # misaligned file.
    rows = assemble_output_rows(dataset.messages, rows_by_id)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=OUTPUT_COLUMNS)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)

    print(f"\nWrote {len(rows)} rows to {output_path} in {elapsed:.1f}s")
    print(f"Action breakdown: {dict(action_counts)}")


if __name__ == "__main__":
    main()
