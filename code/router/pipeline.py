"""Orchestrates one message through the full pipeline:

  media extraction -> safety pre-filter -> [hard override? skip to output]
  -> context assembly -> evidence retrieval (+ thread context if text is
  sparse) -> LLM reasoning -> calibration -> output row

Three things worth flagging for the interview:
  1. Hard overrides never reach the LLM. This is deterministic and cheap,
     and it means a safety-critical row's correctness doesn't depend on a
     model call succeeding.
  2. LLM decisions are cached to disk keyed by message_id + a hash of the
     assembled context (and tagged by dry-run vs. real, so a wiring check
     never poisons a real run's cache), so a rerun with unchanged inputs
     never re-calls the API.
  3. Confidence is only capped for a genuinely under-informed message: text
     is sparse AND the media extraction came back thin AND there's no
     thread history to lean on either. Any one of those being substantive
     is enough to let the LLM's own confidence stand.
"""
from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Optional

from .config import CFG
from .context import build_context
from .data_loader import Dataset
from .load_signal import LoadIndex
from .media import MediaExtractor, extracted_text_for_message, is_extraction_thin
from .reasoning import LlmDecision, reason_about_message
from .retrieval import find_evidence, find_thread_context, is_text_sparse
from .safety import evaluate_safety

DECISION_CACHE_FILE = CFG.cache_dir / "llm_decisions.json"


def _load_decision_cache() -> dict:
    if DECISION_CACHE_FILE.exists():
        return json.loads(DECISION_CACHE_FILE.read_text(encoding="utf-8"))
    return {}


def _save_decision_cache(cache: dict) -> None:
    DECISION_CACHE_FILE.parent.mkdir(parents=True, exist_ok=True)
    DECISION_CACHE_FILE.write_text(json.dumps(cache, indent=2, ensure_ascii=False), encoding="utf-8")


def _context_hash(message: dict, extracted_media_text: Optional[str]) -> str:
    payload = json.dumps({"m": message, "media": extracted_media_text}, sort_keys=True)
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()[:16]


class Router:
    def __init__(self, dataset: Dataset, client, dry_run: bool = False) -> None:
        self.dataset = dataset
        self.client = client
        self.dry_run = dry_run
        self.extractor = MediaExtractor(dataset, client=None if dry_run else client, dry_run=dry_run)
        # Built from the full batch (not just rows processed so far) -- this only
        # uses raw timestamp/type/sender metadata, never another row's routing
        # decision, so it's independent of processing order. See load_signal.py.
        self.load_index = LoadIndex(dataset, dataset.messages)
        self._decision_cache = _load_decision_cache()

    def route(self, message: dict) -> dict:
        message_id = message["message_id"]

        extracted_media_text = extracted_text_for_message(self.extractor, message)
        safety = evaluate_safety(self.dataset, message, extracted_media_text)

        override = safety.hard_override
        if override is not None:
            return {
                "message_id": message_id,
                "action": override["action"],
                "message_type": override["message_type"],
                "reason": override["reason"],
                "confidence": CFG.hard_override_confidence,
                "evidence_message_ids": override["evidence_message_ids"],
            }

        ctx = build_context(self.dataset, message, extracted_media_text, safety)
        evidence = find_evidence(self.dataset, message, extracted_media_text)

        sparse_text = is_text_sparse(message.get("message_text", ""))
        thread_context = find_thread_context(self.dataset, message) if sparse_text else []
        if thread_context:
            # Merge without duplicating a message that both retrieval modes happened to surface.
            existing_ids = {e.message_id for e in evidence}
            evidence = evidence + [e for e in thread_context if e.message_id not in existing_ids]

        ctx.evidence_candidates = [
            {"message_id": e.message_id, "score": e.score} for e in evidence
        ]
        ctx.load_signal = self.load_index.assess(message)

        decision = self._decide(message_id, ctx, evidence)

        confidence = decision.confidence
        if safety.vague_intro:
            # Never let the model claim more certainty than the evidence supports
            # on a cold-start opener with nothing to confirm it either way.
            confidence = min(confidence, CFG.vague_intro_confidence_cap)
        elif sparse_text and is_extraction_thin(extracted_media_text) and not thread_context:
            # The narrower case: sparse text AND thin media extraction AND no
            # thread history to lean on either -- genuinely nothing to go on,
            # as opposed to just "not text" (a clear QR code with no caption
            # is not under-informed, and shouldn't be capped).
            confidence = min(confidence, CFG.sparse_no_grounding_confidence_cap)

        return {
            "message_id": message_id,
            "action": decision.action,
            "message_type": decision.message_type,
            "reason": decision.reason,
            "confidence": round(confidence, 2),
            "evidence_message_ids": decision.evidence_message_ids,
        }

    def _decide(self, message_id: str, ctx, evidence) -> LlmDecision:
        cache_key = message_id
        ctx_hash = _context_hash(ctx.message, ctx.extracted_media_text)
        cached = self._decision_cache.get(cache_key)
        # Dry-run placeholders and real LLM decisions are never interchangeable:
        # a dry-run pass must not poison a later real run's cache, and vice versa.
        if cached and cached.get("_ctx_hash") == ctx_hash and cached.get("_dry_run") == self.dry_run:
            d = cached["decision"]
            return LlmDecision(**d)

        if self.dry_run:
            decision = LlmDecision(
                message_type="unknown",
                action="digest",
                reason="Dry-run mode: no API key configured, LLM reasoning skipped.",
                confidence=0.5,
                evidence_message_ids="none",
            )
        else:
            decision = reason_about_message(self.client, ctx, evidence)

        self._decision_cache[cache_key] = {
            "_ctx_hash": ctx_hash,
            "_dry_run": self.dry_run,
            "decision": decision.__dict__,
        }
        _save_decision_cache(self._decision_cache)
        return decision
