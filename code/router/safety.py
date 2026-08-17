"""Deterministic safety pre-filter.

This module is pure Python -- no network calls, no LLM -- specifically so
it can be unit-tested in isolation (see tests/test_safety.py) and so its
behavior doesn't depend on how a model *happens* to respond to a prompt.

Everything in `message_text` (and OCR/ASR output) is treated as untrusted
data to be classified, never as an instruction. INJECTION_PATTERNS exists
to catch content that tries to talk the router into trusting it -- and a
hit there is itself treated as strong evidence of a scam, not a separate
concern from it.

Two of the checks here are HARD overrides: if they fire, the pipeline
never calls the LLM at all for the action/message_type/confidence -- it
writes the row directly. This is deliberate. A model can be argued with;
a fixed rule cannot, and safety-critical rows should not depend on a
model call succeeding or a clever prompt talking the model out of it.
"""
from __future__ import annotations

import re
from dataclasses import dataclass, field

from .config import CFG
from .data_loader import Dataset

# ---------------------------------------------------------------------------
# Pattern libraries
# ---------------------------------------------------------------------------

# Content that tries to instruct the router directly, wearing a "system"
# or "assistant" costume. Grounded in real rows found in this dataset
# (msg_107, msg_110, msg_095) -- all three are also OTP/payment phishing,
# which is the point: the injection attempt IS the scam signal.
INJECTION_PATTERNS = [
    re.compile(p, re.IGNORECASE)
    for p in [
        r"routing override",
        r"assistant instruction",
        r"system note (for|to) (the )?(notification )?router",
        r"set\s+action\s*=",
        r"confidence\s*=\s*1(\.0)?\b",
        r"ignore sender risk",
        r"always (mark|notify|classify) this",
        r"classify (this )?as (urgent|notify)\b",
        r"router,?\s+(please\s+)?(mark|classify|route)",
        r"ignore (all\s+)?(previous|prior)\s+(routing\s+)?(rules|instructions)",
        r"\bmark\s+(this\s+)?(message\s+)?as\s+(notify|urgent)\b",
        r"disregard (the\s+)?(above|previous)\s+(routing\s+)?(rules|instructions)",
        r"internal router metadata",
        r"verified_business\s*=\s*true",
        r"user_priority\s*=\s*high",
        r"\baction\s*=\s*(notify|digest|mute)\b",
    ]
]

# Softer content-level scam signals, grouped by category. A single hit is
# a soft signal fed to the LLM; the *same* categorization is reused by the
# compromised-sender check below (a repeated category match against a
# sender's own reported history is what makes it a hard override).
SCAM_KEYWORD_PATTERNS: dict[str, re.Pattern] = {
    "otp_request": re.compile(r"\botp\b|verification code|login code|\b6[\s-]?digit code\b", re.I),
    "payment_pressure": re.compile(
        r"pay\b.{0,20}(immediately|now|today)|complete before|clearance amount|token amount|"
        r"penalty|blocked (tomorrow|today)", re.I,
    ),
    "click_or_scan": re.compile(r"click (this|here)|scan (this|the) qr|use this link", re.I),
    "account_risk": re.compile(
        r"account (suspended|blocked)|reactivate|\bkyc\b|security patch failed|workspace suspended", re.I,
    ),
    "screenshot_ask": re.compile(r"send (a |the )?screenshot", re.I),
    "prize_or_lottery": re.compile(
        r"you.?ve won|claim your (prize|winnings|reward)|number was selected|lottery|"
        r"giveaway winner|voucher expires", re.I,
    ),
    "code_forward_request": re.compile(
        r"accidentally (sent|texted)|sent (it |that )?to (you|your number) by mistake|"
        r"forward (me |it )?(the |that )?code", re.I,
    ),
    "fake_app_or_upgrade_link": re.compile(
        r"whatsapp gold|exclusive (version|upgrade)|unlock premium features|modified whatsapp", re.I,
    ),
    "impersonated_platform_support": re.compile(
        r"whatsapp (support|team|representative)|official whatsapp (team|support)", re.I,
    ),
    "unsolicited_job_offer_fee": re.compile(
        r"registration fee|training fee|onboarding fee|pay.{0,15}(uniform|starter kit)|"
        r"hiring immediately.{0,20}no experience", re.I,
    ),
}

CHAIN_LETTER_PATTERNS = [
    re.compile(p, re.IGNORECASE)
    for p in [
        r"forward (this )?to (at least )?\d+",
        r"share in all (family )?groups",
        r"do not (break|ignore) the chain",
        r"\b(forward|fwd) as received\b",
        r"read till end and share",
        r"do not ignore.{0,20}luck",
        r"(forward|share|pls forward|please forward).{0,15}(to )?(family|whatsapp )?groups",
    ]
]

GROUP_INVITE_PATTERNS = [
    re.compile(p, re.IGNORECASE)
    for p in [
        r"chat\.whatsapp\.com",
        r"discord\.gg",
        r"join (this|our) group",
        r"click to join",
        r"add me to",
        r"join now for",
    ]
]

VAGUE_INTRO_PATTERNS = [
    re.compile(p, re.IGNORECASE)
    for p in [
        r"\bis this\b.{0,30}\bfrom\b",
        r"\bnew number\b",
        r"long time no see",
        r"do i know you",
        r"remember me\??$",
        r"(got|found) (this|your) number (from|on)",
    ]
]


def scam_signature(text: str) -> frozenset[str]:
    """Which scam-keyword categories does this text hit? Used both as a
    standalone soft signal and, comparing across two texts, as the basis
    for the compromised-sender check."""
    text = text or ""
    return frozenset(name for name, pat in SCAM_KEYWORD_PATTERNS.items() if pat.search(text))


def _matches_any(patterns: list[re.Pattern], text: str) -> bool:
    text = text or ""
    return any(p.search(text) for p in patterns)


# ---------------------------------------------------------------------------
# Result container
# ---------------------------------------------------------------------------

@dataclass
class SafetySignals:
    injection_detected: bool = False
    scam_categories: frozenset[str] = field(default_factory=frozenset)
    chain_letter: bool = False
    group_invite_signal: bool = False
    is_cold_start: bool = False  # no prior messages from this sender to this recipient
    vague_intro: bool = False
    compromised_sender: bool = False
    compromised_evidence_ids: list[str] = field(default_factory=list)
    # Same-sender, same-recipient prior messages sharing a scam-keyword category,
    # regardless of how the recipient reacted. Broader than compromised_evidence_ids
    # (which requires a reported/muted outcome) -- used to enrich the injection
    # override's evidence when a matching precedent exists but wasn't flagged.
    related_history_ids: list[str] = field(default_factory=list)
    highly_forwarded: bool = False
    domain_lookalike_risk: bool = False
    domain_mismatch_soft: bool = False
    domain_mismatch_detail: str = ""  # e.g. "official=chase.com, used=chase-secure-alert.com, age=10d"
    # Prior messages from this same business to this same recipient that were
    # actually reported or muted -- a validated repeat-offense pattern, not
    # just "this business exists in history." Empty when no such pattern
    # exists yet (e.g. a first-contact domain-lookalike), which is a
    # legitimate "none", not a gap.
    domain_lookalike_evidence_ids: list[str] = field(default_factory=list)
    notes: list[str] = field(default_factory=list)

    @property
    def hard_override(self) -> dict | None:
        """If set, the pipeline skips the LLM and writes this row directly.

        All three branches below always pair message_type="scam" with
        action="mute" -- this is the one (message_type -> action) pairing
        that's *supposed* to be 100% deterministic, because these rows never
        reach the LLM at all. It's the documented exception the
        evaluation harness's type/action-collapse check (see
        evaluation/main.py::check_type_action_collapse) is told to exempt --
        every OTHER message_type must show independently-reasoned variation,
        not a table lookup from the label.
        """
        if self.injection_detected:
            evidence = (self.compromised_evidence_ids or self.related_history_ids)[:4]
            return {
                "action": "mute",
                "message_type": "scam",
                "reason": (
                    "Message body attempts to instruct the router directly "
                    "(e.g. fake routing/system/assistant directive) alongside a "
                    "phishing-style request; treated as untrusted content, not an instruction."
                ),
                "evidence_message_ids": ";".join(evidence) if evidence else "none",
            }
        if self.compromised_sender:
            ev = ";".join(self.compromised_evidence_ids) if self.compromised_evidence_ids else "none"
            return {
                "action": "mute",
                "message_type": "scam",
                "reason": (
                    "Same sender previously sent messages with this content pattern that this "
                    "user reported or muted; identity appears trusted but is likely spoofed or compromised."
                ),
                "evidence_message_ids": ev,
            }
        if self.domain_lookalike_risk:
            has_prior_pattern = bool(self.domain_lookalike_evidence_ids)
            reason = (
                "Unverified business sending from a recently-registered domain that doesn't match "
                f"its official domain ({self.domain_mismatch_detail}) -- classic brand-impersonation "
                "phishing pattern, not a legitimate account."
            )
            if has_prior_pattern:
                reason += (
                    " This recipient has previously reported or muted messages from this same "
                    "business with the same pattern."
                )
            return {
                "action": "mute",
                "message_type": "scam",
                "reason": reason,
                "evidence_message_ids": (
                    ";".join(self.domain_lookalike_evidence_ids) if has_prior_pattern else "none"
                ),
            }
        return None


# ---------------------------------------------------------------------------
# Main entry point
# ---------------------------------------------------------------------------

def evaluate_safety(dataset: Dataset, message: dict, extracted_media_text: str | None) -> SafetySignals:
    text = (message.get("message_text") or "").strip()
    if extracted_media_text:
        text = f"{text}\n{extracted_media_text}".strip()

    sig = SafetySignals()

    if _matches_any(INJECTION_PATTERNS, text):
        sig.injection_detected = True
        sig.notes.append("injection pattern matched")

    sig.scam_categories = scam_signature(text)
    if sig.scam_categories:
        sig.notes.append(f"scam-keyword categories: {sorted(sig.scam_categories)}")

    if _matches_any(CHAIN_LETTER_PATTERNS, text):
        sig.chain_letter = True
        sig.notes.append("chain-letter / forward-for-luck pattern matched")

    if _matches_any(GROUP_INVITE_PATTERNS, text):
        sig.group_invite_signal = True
        sig.notes.append("group/community invite pattern matched")

    try:
        fwd_count = int(message.get("forwarded_count") or 0)
    except (TypeError, ValueError):
        fwd_count = 0
    if fwd_count >= CFG.high_forward_count_threshold:
        sig.highly_forwarded = True
        sig.notes.append(f"forwarded_count={fwd_count} >= mass-blast threshold")

    # Domain-lookalike / brand-impersonation: an unverified business sending
    # from a domain that doesn't match its own official_domain, registered
    # recently. Validated against 7 real messages in this dataset (all
    # 2-17 days old); the one legitimate mismatch on file (a verified
    # business using a link-shortener domain) is 3368 days old, which is why
    # this needs an age gate, not just a mismatch check.
    business = dataset.business(message.get("business_id", ""))
    if business:
        official = business.get("official_domain", "")
        used = business.get("domain_used_by_sender", "")
        try:
            domain_age = int(business.get("domain_used_by_sender_age_days") or 0)
        except (TypeError, ValueError):
            domain_age = 0
        if official and used and official != used:
            detail = f"official={official}, used={used}, age={domain_age}d"
            if business.get("verified") != "1" and domain_age < CFG.domain_lookalike_max_age_days:
                sig.domain_lookalike_risk = True
                sig.domain_mismatch_detail = detail
                sig.notes.append(f"domain-lookalike hard override: {detail}")
                # Thread through real supporting evidence when this recipient has
                # a validated (reported/muted) prior pattern with this exact
                # business -- never invent it, and "none" stays correct when no
                # such pattern exists (e.g. a first-contact domain-lookalike).
                recipient = message.get("user_id", "")
                business_id = message.get("business_id", "")
                prior_from_business = [
                    h for h in dataset.history_for_recipient(recipient)
                    if h.get("business_id") == business_id
                ]
                reported_or_muted = []
                for h in prior_from_business:
                    ev = dataset.event(recipient, h["message_id"])
                    if ev and (ev.get("message_reported") == "1" or ev.get("muted_after_message") == "1"):
                        reported_or_muted.append(h["message_id"])
                sig.domain_lookalike_evidence_ids = sorted(reported_or_muted)
                if reported_or_muted:
                    sig.notes.append(f"domain-lookalike + prior reported/muted pattern: {reported_or_muted}")
            else:
                sig.domain_mismatch_soft = True
                sig.domain_mismatch_detail = detail
                sig.notes.append(f"domain mismatch present but below hard-override bar: {detail}")
        elif not official and used:
            # No official_domain on file at all to compare against, so this can't
            # be scored as a "mismatch" -- but an unverified business sending from
            # a domain registered within the same window as the validated
            # lookalike cases (no real messages.csv example exists yet, so this
            # stays soft, not a hard override) is worth flagging rather than
            # silently ignoring.
            if business.get("verified") != "1" and domain_age < CFG.domain_lookalike_max_age_days:
                sig.domain_mismatch_soft = True
                sig.domain_mismatch_detail = f"official=missing, used={used}, age={domain_age}d"
                sig.notes.append(f"no official_domain on file, unverified young domain: {sig.domain_mismatch_detail}")

    sender = message.get("sender_user_id", "")
    recipient = message.get("user_id", "")
    prior_between = dataset.history_between(sender, recipient) if sender else []
    sig.is_cold_start = sender != "" and len(prior_between) == 0

    if sig.is_cold_start and not sig.scam_categories and _matches_any(VAGUE_INTRO_PATTERNS, text):
        sig.vague_intro = True
        sig.notes.append("vague first-contact opener from a sender with no prior history")

    # Compromised / impersonated trusted sender: does this exact sender have a
    # history (with this recipient) of the same scam-keyword category? Two
    # tiers: `related` is any content-pattern match (used just as supporting
    # evidence); `compromised_evidence_ids` additionally requires the
    # recipient having reported/muted it, which is what makes it strong
    # enough to be a hard override on its own.
    if sender and sig.scam_categories:
        related: list[str] = []
        reported_or_muted: list[str] = []
        for h in prior_between:
            h_sig = scam_signature(h.get("message_text", ""))
            if h_sig & sig.scam_categories:
                related.append(h["message_id"])
                ev = dataset.event(recipient, h["message_id"])
                if ev and (ev.get("message_reported") == "1" or ev.get("muted_after_message") == "1"):
                    reported_or_muted.append(h["message_id"])
        sig.related_history_ids = sorted(related)
        if reported_or_muted:
            sig.compromised_sender = True
            sig.compromised_evidence_ids = sorted(reported_or_muted)
            sig.notes.append(f"compromised-sender pattern: prior reported/muted messages {reported_or_muted}")
        elif related:
            sig.notes.append(f"same sender sent similar content before (outcome not flagged): {related}")

    return sig
