"""Evidence retrieval for `evidence_message_ids`.

Deliberately not embedding-based: at ~1000 history rows, the dominant
signal is structural (same sender / same group / same business as the
current message), and lexical similarity (rapidfuzz) is enough to rank
within that pool. This keeps retrieval deterministic, needs no additional
API/account, and is easy to defend line-by-line in the interview.

The LLM reasoning stage picks its final evidence_message_ids from the
candidates this module returns -- it does not invent IDs freely, which is
what keeps evidence grounded.
"""
from __future__ import annotations

import re
from dataclasses import dataclass

from rapidfuzz import fuzz

from .config import CFG
from .data_loader import Dataset


@dataclass
class EvidenceCandidate:
    message_id: str
    score: float
    text: str
    outcome_note: str
    source: str = "similarity"  # "similarity" (lexical match) or "recency" (thread-context pull)


def _outcome_note(dataset: Dataset, recipient: str, message_id: str) -> str:
    ev = dataset.event(recipient, message_id)
    if not ev:
        return "no recorded reaction"
    flags = []
    if ev.get("message_reported") == "1":
        flags.append("reported")
    if ev.get("muted_after_message") == "1":
        flags.append("muted after")
    if ev.get("notification_dismissed") == "1":
        flags.append("dismissed")
    if ev.get("message_replied") == "1":
        flags.append("replied")
    if ev.get("message_opened") == "1" and not flags:
        flags.append("opened")
    return ", ".join(flags) if flags else "no strong reaction recorded"


def find_evidence(dataset: Dataset, message: dict, extracted_media_text: str | None) -> list[EvidenceCandidate]:
    recipient = message.get("user_id", "")
    sender = message.get("sender_user_id", "")
    group_id = message.get("group_id", "")
    business_id = message.get("business_id", "")

    pool = dataset.history_for_recipient(recipient)
    if not pool:
        return []

    # Structural filter first: same sender, same group, or same business.
    # Falling back to the full recipient history only if nothing structural matches.
    structural = [
        h for h in pool
        if (sender and h.get("sender_user_id") == sender)
        or (group_id and h.get("group_id") == group_id)
        or (business_id and h.get("business_id") == business_id)
    ]
    candidate_pool = structural or pool

    query_text = (message.get("message_text") or "")
    if extracted_media_text:
        query_text = f"{query_text} {extracted_media_text}"
    query_text = query_text.strip()

    scored: list[EvidenceCandidate] = []
    for h in candidate_pool:
        h_text = h.get("message_text", "")
        score = fuzz.token_set_ratio(query_text, h_text) / 100.0 if query_text and h_text else 0.0
        # Small structural boost so an exact-sender/group match outranks a
        # purely lexical coincidence from an unrelated thread.
        if h in structural:
            score = min(1.0, score + 0.1)
        if score >= CFG.min_retrieval_score:
            scored.append(
                EvidenceCandidate(
                    message_id=h["message_id"],
                    score=round(score, 3),
                    text=h_text,
                    outcome_note=_outcome_note(dataset, recipient, h["message_id"]),
                )
            )

    scored.sort(key=lambda c: c.score, reverse=True)
    return scored[: CFG.max_retrieval_candidates]


# ---------------------------------------------------------------------------
# Text/media balance: sparse-text detection + recency-based thread context
# ---------------------------------------------------------------------------

_WORD_RE = re.compile(r"[A-Za-z0-9']+")


def is_text_sparse(text: str, threshold_words: int | None = None) -> bool:
    """Soft check, not just an emptiness check.

    A single emoji, "ok", or a two-word caption all count as sparse: there's
    too little of the sender's own words to lean on, even though it's not
    literally an empty string. An umbrella emoji accompanying a photo is a
    real example -- almost no text, but it's doing real framing work for the
    image, which is exactly the case where thread history should help carry
    the rest of the interpretation rather than being skipped because the
    text field isn't technically blank.
    """
    threshold = CFG.sparse_text_word_threshold if threshold_words is None else threshold_words
    words = _WORD_RE.findall(text or "")
    return len(words) < threshold


def find_thread_context(dataset: Dataset, message: dict) -> list[EvidenceCandidate]:
    """Most recent same-thread messages, regardless of lexical similarity.

    Deliberately a different retrieval mode from find_evidence: find_evidence
    asks "what looks similar to this message's content" and has little to
    work with when there isn't much content to compare (an empty-captioned
    voice note, an emoji). This asks "what was already happening in this
    thread right before this message" -- recency, not similarity -- which is
    what actually supplies missing context for a sparse-text message.

    Same structural priority as find_evidence: same sender first, then same
    business, then same group.
    """
    recipient = message.get("user_id", "")
    sender = message.get("sender_user_id", "")
    group_id = message.get("group_id", "")
    business_id = message.get("business_id", "")
    current_ts = message.get("created_at", "")

    pool = dataset.history_for_recipient(recipient)
    if not pool:
        return []

    if sender:
        thread = [h for h in pool if h.get("sender_user_id") == sender]
    elif business_id:
        thread = [h for h in pool if h.get("business_id") == business_id]
    elif group_id:
        thread = [h for h in pool if h.get("group_id") == group_id]
    else:
        thread = []

    # Only messages that actually happened before this one, most recent first.
    thread = [h for h in thread if h.get("created_at", "") < current_ts]
    thread.sort(key=lambda h: h.get("created_at", ""), reverse=True)

    return [
        EvidenceCandidate(
            message_id=h["message_id"],
            score=1.0,  # recency-ranked, not similarity-scored -- the number isn't meaningful here
            text=h.get("message_text", ""),
            outcome_note=_outcome_note(dataset, recipient, h["message_id"]),
            source="recency",
        )
        for h in thread[: CFG.thread_context_k]
    ]
