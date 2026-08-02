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

    # Deterministic ordering: always process in message_id order.
    messages = sorted(dataset.messages, key=lambda m: m["message_id"])
    if args.limit:
        messages = messages[: args.limit]

    rows = []
    action_counts: Counter = Counter()
    start = time.time()
    for i, message in enumerate(messages, start=1):
        row = router.route(message)
        rows.append(row)
        action_counts[row["action"]] += 1
        print(f"  [{i}/{len(messages)}] {row['message_id']} -> {row['action']} / {row['message_type']} (conf={row['confidence']})")

    elapsed = time.time() - start
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
