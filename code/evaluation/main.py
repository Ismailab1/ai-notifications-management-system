#!/usr/bin/env python3
"""Evaluation harness -- run this BEFORE spending time on the full messages.csv.

Runs the router against dataset/sample_messages.csv (the 30 solved examples)
and diffs predictions against the solved action / message_type / confidence /
evidence_message_ids columns. Accuracy is broken out by media_type, since
image/voice extraction quality is the most fragile part of the pipeline and
an aggregate number can hide a bad slice.

Usage:
    python evaluation/main.py               # full sample eval
    python evaluation/main.py --dry-run      # wiring test, no API key needed
    python evaluation/main.py --verbose      # print every row's diff, not just mismatches
"""
from __future__ import annotations

import argparse
import sys
from collections import defaultdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))  # allow `import router` from evaluation/

from router.config import CFG
from router.data_loader import load_dataset
from router.pipeline import Router


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--dataset-dir", type=Path, default=CFG.dataset_dir)
    p.add_argument("--dry-run", action="store_true")
    p.add_argument("--verbose", action="store_true", help="print every row, not just mismatches")
    return p.parse_args()


def evidence_overlap(predicted: str, solved: str) -> bool:
    """evidence_message_ids match if they share at least one id, or both say 'none'."""
    pred_set = set(x for x in predicted.split(";") if x and x != "none")
    solved_set = set(x for x in solved.split(";") if x and x != "none")
    if not pred_set and not solved_set:
        return True
    return bool(pred_set & solved_set)


def main() -> None:
    args = parse_args()
    if not args.dry_run:
        CFG.validate()

    dataset = load_dataset(args.dataset_dir)
    samples = sorted(dataset.sample_messages, key=lambda m: m["message_id"])
    if not samples:
        print("No rows found in sample_messages.csv -- nothing to evaluate.")
        return

    client = None
    if not args.dry_run:
        import anthropic

        client = anthropic.Anthropic(api_key=CFG.anthropic_api_key)

    router = Router(dataset, client, dry_run=args.dry_run)

    # tallies: overall + broken out by media_type
    totals: dict[str, dict] = defaultdict(lambda: {"n": 0, "action_ok": 0, "type_ok": 0, "evidence_ok": 0})
    conf_deltas = []

    for row in samples:
        media_key = row.get("media_type") or "text"
        pred = router.route(row)

        t = totals[media_key]
        t["n"] += 1
        action_ok = pred["action"] == row["action"]
        type_ok = pred["message_type"] == row["message_type"]
        ev_ok = evidence_overlap(pred["evidence_message_ids"], row["evidence_message_ids"])
        t["action_ok"] += int(action_ok)
        t["type_ok"] += int(type_ok)
        t["evidence_ok"] += int(ev_ok)

        try:
            conf_deltas.append(abs(float(pred["confidence"]) - float(row["confidence"])))
        except (TypeError, ValueError):
            pass

        if args.verbose or not (action_ok and type_ok):
            mark = "OK" if (action_ok and type_ok) else "MISMATCH"
            print(f"[{mark}] {row['message_id']} ({media_key})")
            print(f"  predicted: action={pred['action']:8s} type={pred['message_type']:16s} conf={pred['confidence']} evidence={pred['evidence_message_ids']}")
            print(f"  solved:    action={row['action']:8s} type={row['message_type']:16s} conf={row['confidence']} evidence={row['evidence_message_ids']}")
            print(f"  predicted reason: {pred['reason']}")
            print(f"  solved reason:    {row['reason']}")
            print()

    print("=" * 70)
    print("SUMMARY (broken out by media_type)")
    print("=" * 70)
    grand_n = grand_action = grand_type = grand_ev = 0
    for media_key, t in sorted(totals.items()):
        n = t["n"]
        grand_n += n
        grand_action += t["action_ok"]
        grand_type += t["type_ok"]
        grand_ev += t["evidence_ok"]
        print(
            f"  {media_key:6s} (n={n:2d})  action={t['action_ok']}/{n} "
            f"type={t['type_ok']}/{n}  evidence_overlap={t['evidence_ok']}/{n}"
        )
    print("-" * 70)
    print(f"  {'ALL':6s} (n={grand_n:2d})  action={grand_action}/{grand_n} type={grand_type}/{grand_n} evidence_overlap={grand_ev}/{grand_n}")
    if conf_deltas:
        avg_delta = sum(conf_deltas) / len(conf_deltas)
        print(f"\n  avg |confidence - solved_confidence| = {avg_delta:.3f}")


if __name__ == "__main__":
    main()
