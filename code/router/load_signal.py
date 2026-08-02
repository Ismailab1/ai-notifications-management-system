"""Situational load: "is this user currently in the middle of a burst of
similar messages" -- built from raw message timestamps, not a real-time
presence signal (this system runs as an offline batch, not a live stream).

Deliberately the NARROW version, decided after weighing it against a
"focus mode" alternative that would suppress everything (personal included)
during a detected busy period: that's a defensible product choice too, but
it requires confidently knowing the user is busy, and an inferred burst from
message timestamps is much weaker evidence than an explicit focus-mode
toggle would be. Using a weak inference to suppress something a stronger
content-level signal already flagged repeats the exact mistake the safety
layer is built to avoid elsewhere.

So this module only ever supports two things, and both are advisory context
handed to the LLM, not a rule that overrides its decision:
  - bundling: several same-category messages arriving close together is a
    reason to batch them into one digest instead of separate interruptions
  - promotion: a follow-up in a thread with recent activity and a
    historically engaged relationship is well-timed, not an interruption

It never computes anything that argues for suppressing a message outright,
and personal-message urgency is never touched by this signal -- that stays
governed entirely by the message's own content, per reasoning.py.
"""
from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timedelta

from .config import CFG
from .data_loader import Dataset

_TS_FORMAT = "%Y-%m-%d %H:%M"


def _parse_ts(value: str) -> datetime | None:
    try:
        return datetime.strptime(value, _TS_FORMAT)
    except (TypeError, ValueError):
        return None


def _category_key(row: dict) -> str:
    """What counts as 'the same category' for bundling purposes -- broad
    conversation_type, not the specific sender/group/business (that's what
    _same_thread checks separately, for the promotion case)."""
    return row.get("conversation_type", "")


def _same_thread(a: dict, b: dict) -> bool:
    if a.get("sender_user_id") and a.get("sender_user_id") == b.get("sender_user_id"):
        return True
    if a.get("business_id") and a.get("business_id") == b.get("business_id"):
        return True
    if a.get("group_id") and a.get("group_id") == b.get("group_id"):
        return True
    return False


@dataclass
class LoadSignal:
    window_hours: float
    recent_same_category_count: int = 0
    recent_same_thread_count: int = 0

    @property
    def is_burst(self) -> bool:
        return self.recent_same_category_count >= CFG.load_burst_threshold

    @property
    def has_recent_thread_activity(self) -> bool:
        return self.recent_same_thread_count > 0


class LoadIndex:
    """Built once per run from message_history + the full messages.csv batch.

    Using the full batch (not just already-processed rows) is intentional
    and doesn't leak anything unfair: this only looks at raw timestamp/type/
    sender metadata for other messages, never at another message's routing
    decision, so it's equivalent to a real system knowing what landed in the
    inbox around the same time, independent of processing order.
    """

    def __init__(self, dataset: Dataset, batch_messages: list[dict]) -> None:
        self._by_user: dict[str, list[dict]] = {}
        for row in dataset.message_history + batch_messages:
            uid = row.get("user_id", "")
            if not uid or not row.get("created_at"):
                continue
            self._by_user.setdefault(uid, []).append(row)
        for rows in self._by_user.values():
            rows.sort(key=lambda r: r.get("created_at", ""))

    def assess(self, message: dict, window_hours: float | None = None) -> LoadSignal:
        window = CFG.load_window_hours if window_hours is None else window_hours
        current_ts = _parse_ts(message.get("created_at", ""))
        user_rows = self._by_user.get(message.get("user_id", ""), [])
        signal = LoadSignal(window_hours=window)
        if current_ts is None or not user_rows:
            return signal

        window_start = current_ts - timedelta(hours=window)
        this_category = _category_key(message)
        this_id = message.get("message_id")

        for row in user_rows:
            if row.get("message_id") == this_id:
                continue
            ts = _parse_ts(row.get("created_at", ""))
            if ts is None or not (window_start <= ts < current_ts):
                continue
            if _category_key(row) == this_category:
                signal.recent_same_category_count += 1
            if _same_thread(row, message):
                signal.recent_same_thread_count += 1

        return signal
