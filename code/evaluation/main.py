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
import csv
import sys
from collections import Counter, defaultdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))  # allow `import router` from evaluation/

from router.config import CFG
from router.data_loader import load_dataset
from router.pipeline import Router

# message_types that are ALWAYS paired with a single action by a documented
# hard override in safety.py (see SafetySignals.hard_override) -- for these,
# a 100% action collapse is the correct, intended behavior, not a bug.
EXEMPT_FROM_COLLAPSE_CHECK = {"scam"}


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--dataset-dir", type=Path, default=CFG.dataset_dir)
    p.add_argument("--dry-run", action="store_true")
    p.add_argument("--verbose", action="store_true", help="print every row, not just mismatches")
    p.add_argument(
        "--check-output", nargs="?", const="__default__", default=None, metavar="PATH",
        help=(
            "Skip the sample eval; instead audit an already-written output.csv "
            "(default: <dataset-dir>/output.csv) for row-order alignment against "
            "messages.csv and message_type/action distributional collapse. No API "
            "key or pipeline run needed -- pure static analysis of the CSV."
        ),
    )
    return p.parse_args()


def evidence_overlap(predicted: str, solved: str) -> bool:
    """evidence_message_ids match if they share at least one id, or both say 'none'."""
    pred_set = set(x for x in predicted.split(";") if x and x != "none")
    solved_set = set(x for x in solved.split(";") if x and x != "none")
    if not pred_set and not solved_set:
        return True
    return bool(pred_set & solved_set)


def check_type_action_collapse(rows: list[dict], min_examples: int = 3) -> list[str]:
    """Flags any message_type that maps to a single action 100% of the time
    across >= min_examples rows, except types produced by a documented hard
    override (EXEMPT_FROM_COLLAPSE_CHECK). A flag here means either an
    undocumented hard rule, or the model collapsing action into a mechanical
    function of type instead of reasoning per-message -- see reasoning.py's
    "Action is never a mechanical function of message_type" guidance."""
    by_type: dict[str, Counter] = defaultdict(Counter)
    for r in rows:
        by_type[r["message_type"]][r["action"]] += 1

    warnings = []
    for mtype, counts in sorted(by_type.items()):
        if mtype in EXEMPT_FROM_COLLAPSE_CHECK:
            continue
        total = sum(counts.values())
        if total >= min_examples and len(counts) == 1:
            (only_action,) = counts.keys()
            warnings.append(
                f"message_type={mtype!r} mapped to action={only_action!r} in all {total} "
                f"examples -- looks like action is being decided by type alone, not "
                f"per-message reasoning (or this deserves a documented hard override)"
            )
    return warnings


def check_row_order_alignment(messages_path: Path, output_path: Path) -> list[str]:
    """The grader compares output.csv against messages.csv POSITIONALLY, not
    by joining on message_id -- see main.py::assemble_output_rows. Round-trips
    through the actual written files (not an in-memory comparison) so a
    regression here is caught before it ever reaches a real submission."""
    if not output_path.exists():
        return [f"{output_path} does not exist -- run main.py first"]
    messages_ids = [r["message_id"] for r in csv.DictReader(messages_path.open(encoding="utf-8"))]
    output_ids = [r["message_id"] for r in csv.DictReader(output_path.open(encoding="utf-8"))]

    if len(messages_ids) != len(output_ids):
        return [
            f"row count mismatch: messages.csv has {len(messages_ids)} rows, "
            f"output.csv has {len(output_ids)}"
        ]
    if messages_ids == output_ids:
        return []
    mismatch_index = next(i for i, (a, b) in enumerate(zip(messages_ids, output_ids)) if a != b)
    return [
        f"row order mismatch at index {mismatch_index}: messages.csv has "
        f"{messages_ids[mismatch_index]!r}, output.csv has {output_ids[mismatch_index]!r} -- "
        f"output.csv is not positionally aligned with messages.csv"
    ]


def run_output_checks(dataset_dir: Path, output_path: Path) -> bool:
    """Static-analysis audit of an already-written output.csv. Returns True
    if the HARD checks pass (row-order alignment -- a certain, silent-failure
    correctness bug if it regresses). The distributional check is advisory:
    it surfaces candidates for human review (a message_type collapsing to one
    action isn't necessarily wrong -- see EXEMPT_FROM_COLLAPSE_CHECK's
    docstring -- so it's printed but doesn't affect the pass/fail exit code.
    Prints a full report either way."""
    rows = list(csv.DictReader(output_path.open(encoding="utf-8"))) if output_path.exists() else []
    print("=" * 70)
    print(f"OUTPUT CHECKS -- {output_path}")
    print("=" * 70)

    order_warnings = check_row_order_alignment(dataset_dir / "messages.csv", output_path)
    if order_warnings:
        print("\n[FAIL] row-order alignment:")
        for w in order_warnings:
            print(f"  - {w}")
    else:
        print("\n[OK] row-order alignment: output.csv matches messages.csv position-for-position")

    collapse_warnings = check_type_action_collapse(rows) if rows else []
    if collapse_warnings:
        print("\n[ADVISORY] message_type/action distribution (review, doesn't fail the check):")
        for w in collapse_warnings:
            print(f"  - {w}")
    else:
        print("\n[OK] message_type/action distribution: no type->action collapse to review")

    print()
    passed = not order_warnings
    print("PASSED" if passed else "FAILED")
    return passed


def main() -> None:
    args = parse_args()

    if args.check_output is not None:
        # Pure static CSV analysis -- no API key, no pipeline run, no dataset load needed.
        output_path = (
            args.dataset_dir / "output.csv" if args.check_output == "__default__" else Path(args.check_output)
        )
        passed = run_output_checks(args.dataset_dir, output_path)
        sys.exit(0 if passed else 1)

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
    pred_rows = []  # for the type/action collapse check below

    for row in samples:
        media_key = row.get("media_type") or "text"
        pred = router.route(row)
        pred_rows.append(pred)

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

    collapse_warnings = check_type_action_collapse(pred_rows)
    if collapse_warnings:
        print("\n  [WARN] message_type/action distribution (predicted, this run):")
        for w in collapse_warnings:
            print(f"    - {w}")


if __name__ == "__main__":
    main()
