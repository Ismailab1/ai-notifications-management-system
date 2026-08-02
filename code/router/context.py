"""Context assembly: turn one messages.csv row into everything the reasoning
stage needs, by joining the structured context files.

The one non-obvious piece here is `compute_engagement`. It implements the
"degraded relationship" design from the interview notes: engagement is a
graded signal (strong -> declining -> disengaged) that shifts the *default*
action for routine content, separate from an explicit mute/opt-out, which
is treated as a much stronger, near-hard signal. Passive disengagement
should not by itself silence urgent or content-flagged messages -- that
carve-out lives in reasoning.py, not here.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime
from typing import Optional

from .data_loader import Dataset
from .load_signal import LoadSignal
from .safety import SafetySignals

_TS_FORMAT = "%Y-%m-%d %H:%M"
_RECENT_REPLY_WINDOW_DAYS = 14
_FAST_REACTION_MINUTES = 10
_SLOW_REACTION_MINUTES = 90
_REACTION_TIME_NUDGE = 0.05


def _to_int(v: str) -> int:
    try:
        return int(v)
    except (TypeError, ValueError):
        return 0


def _parse_ts(value: str) -> Optional[datetime]:
    try:
        return datetime.strptime(value, _TS_FORMAT)
    except (TypeError, ValueError):
        return None


def compute_baseline_dismissal_rate(dataset: Dataset, user_id: str) -> Optional[float]:
    """General notification-fatigue baseline from daily_notification_summary.csv
    (2026-07-04 to 2026-07-17 -- entirely before messages.csv starts on
    2026-07-18). This is deliberately NOT a live signal: it can only speak to
    "how notification-fatigued is this person overall", never to anything
    happening around the current message -- that's what the message-density
    situational-load signal (load_signal.py) is for. Don't conflate the two."""
    rows = dataset.daily_summary_for_user(user_id)
    if not rows:
        return None
    total_sent = sum(_to_int(r.get("notifications_sent")) for r in rows)
    if total_sent == 0:
        return None
    total_dismissed = sum(_to_int(r.get("notifications_dismissed")) for r in rows)
    return round(total_dismissed / total_sent, 2)


@dataclass
class Engagement:
    label: str          # 'muted' | 'disengaged' | 'declining' | 'engaged' | 'unknown'
    score: float         # 0.0 (fully checked out) .. 1.0 (highly engaged)
    explicit_negative: bool  # user took a direct action (muted the group / opted out of promos)
    explanation: str


def _engagement_from_counts(read: int, replies: int, dismissed: int) -> tuple[str, float]:
    total_seen = read + dismissed
    if total_seen == 0 and replies == 0:
        return "unknown", 0.5
    ratio = (read + 2 * replies) / total_seen if total_seen else 1.0
    ratio = min(ratio, 1.0)
    if total_seen >= 3 and ratio < 0.25:
        return "disengaged", ratio
    if total_seen >= 3 and ratio < 0.55:
        return "declining", ratio
    return "engaged", max(ratio, 0.55)


def compute_engagement(dataset: Dataset, message: dict) -> Engagement:
    conv = message.get("conversation_type")

    if conv == "group":
        gm = dataset.group_member(message.get("group_id", ""), message.get("user_id", ""))
        if not gm:
            return Engagement("unknown", 0.5, False, "no group membership record on file")
        muted = gm.get("group_muted_by_user") == "1"
        read = _to_int(gm.get("messages_read_30d"))
        replies = _to_int(gm.get("replies_sent_30d"))
        dismissed = _to_int(gm.get("notifications_dismissed_30d"))
        if muted:
            return Engagement("muted", 0.0, True, "user has explicitly muted this group")
        label, score = _engagement_from_counts(read, replies, dismissed)
        return Engagement(label, score, False, f"group activity: read={read} replied={replies} dismissed={dismissed}")

    if conv == "business":
        bh = dataset.business_history(message.get("user_id", ""), message.get("business_id", ""))
        if not bh:
            return Engagement("unknown", 0.5, False, "no prior relationship with this business on file")
        opted_out = bool(bh.get("promotions_opted_out_at")) and bh.get("allows_promotions") == "0"
        opened = _to_int(bh.get("messages_opened_30d"))
        replies = _to_int(bh.get("messages_replied_30d"))
        dismissed = _to_int(bh.get("messages_dismissed_30d"))
        if opted_out:
            return Engagement("muted", 0.0, True, "user opted out of promotions from this business")
        label, score = _engagement_from_counts(opened, replies, dismissed)
        why = bh.get("why_user_knows_account") or "no stated relationship"
        explanation = f"business relationship: {why}; opened={opened} replied={replies} dismissed={dismissed}"
        # Recency over rollup: a reply from yesterday shouldn't get diluted into a
        # 30-day average that reads as declining/disengaged. Only ever lifts the
        # label, never lowers it -- a recent reply is positive evidence, its
        # absence isn't negative evidence (see the "friends at a restaurant" case).
        last_reply = _parse_ts(bh.get("last_reply_at", ""))
        msg_ts = _parse_ts(message.get("created_at", ""))
        if last_reply and msg_ts and label in ("declining", "disengaged"):
            days_since_reply = (msg_ts - last_reply).days
            if 0 <= days_since_reply <= _RECENT_REPLY_WINDOW_DAYS:
                label, score = "engaged", max(score, 0.75)
                explanation += f"; replied {days_since_reply}d ago -- recency lifts this above the 30d rollup"
        return Engagement(label, score, False, explanation)

    # personal: derive from actual reaction history to this specific sender
    sender = message.get("sender_user_id", "")
    recipient = message.get("user_id", "")
    prior = dataset.history_between(sender, recipient)
    if not prior:
        return Engagement("unknown", 0.5, False, "no prior message history with this sender")
    opened = replied = dismissed = 0
    reaction_times: list[int] = []
    for h in prior:
        ev = dataset.event(recipient, h["message_id"])
        if not ev:
            continue
        opened += _to_int(ev.get("message_opened"))
        replied += _to_int(ev.get("message_replied"))
        dismissed += _to_int(ev.get("notification_dismissed"))
        if ev.get("reaction_time_minutes"):
            reaction_times.append(_to_int(ev.get("reaction_time_minutes")))
    label, score = _engagement_from_counts(opened, replied, dismissed)
    explanation = f"personal history: opened={opened} replied={replied} dismissed={dismissed}"
    # Small nudge on top of the counts, not a replacement for them: distinguishes
    # "attentive and fast" from "eventually opened" without overriding what the
    # open/reply/dismiss counts already establish.
    if reaction_times:
        avg_reaction = sum(reaction_times) / len(reaction_times)
        if avg_reaction <= _FAST_REACTION_MINUTES:
            score = min(1.0, score + _REACTION_TIME_NUDGE)
            explanation += f"; avg reaction time {avg_reaction:.0f}m (fast, +{_REACTION_TIME_NUDGE:.2f})"
        elif avg_reaction >= _SLOW_REACTION_MINUTES:
            score = max(0.0, score - _REACTION_TIME_NUDGE)
            explanation += f"; avg reaction time {avg_reaction:.0f}m (slow, -{_REACTION_TIME_NUDGE:.2f})"
    return Engagement(label, score, False, explanation)


@dataclass
class MessageContext:
    message: dict
    recipient: Optional[dict]
    group: Optional[dict]
    group_member: Optional[dict]
    business: Optional[dict]
    business_history: Optional[dict]
    engagement: Engagement
    extracted_media_text: Optional[str]
    safety: SafetySignals
    evidence_candidates: list[dict] = field(default_factory=list)  # filled in by retrieval.py
    load_signal: Optional[LoadSignal] = None  # filled in by pipeline.py
    baseline_dismissal_rate: Optional[float] = None  # from daily_notification_summary.csv, general fatigue baseline


def build_context(
    dataset: Dataset,
    message: dict,
    extracted_media_text: Optional[str],
    safety: SafetySignals,
) -> MessageContext:
    return MessageContext(
        message=message,
        recipient=dataset.user(message.get("user_id", "")),
        group=dataset.group(message.get("group_id", "")),
        group_member=dataset.group_member(message.get("group_id", ""), message.get("user_id", "")),
        business=dataset.business(message.get("business_id", "")),
        business_history=dataset.business_history(message.get("user_id", ""), message.get("business_id", "")),
        engagement=compute_engagement(dataset, message),
        extracted_media_text=extracted_media_text,
        safety=safety,
        baseline_dismissal_rate=compute_baseline_dismissal_rate(dataset, message.get("user_id", "")),
    )
