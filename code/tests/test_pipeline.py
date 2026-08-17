"""Unit tests for router/pipeline.py's decision cache.

Run directly:  python tests/test_pipeline.py
"""
from __future__ import annotations

import sys
import tempfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import router.pipeline as pipeline_module
from router.context import build_context
from router.data_loader import Dataset
from router.pipeline import Router, _context_hash
from router.safety import evaluate_safety


def _empty_dataset() -> Dataset:
    return Dataset(
        messages=[], sample_messages=[], users={}, groups={}, group_members={},
        business_accounts={}, user_business_history={}, message_history=[],
        message_events={}, images={}, voice_notes={}, daily_notification_summary=[],
    )


def test_dry_run_cache_write_does_not_clobber_a_real_cached_decision():
    # Regression: cache writes used to key on message_id alone, so a dry-run
    # pass touching an id that already had a real cached decision would
    # silently overwrite it with the dry-run placeholder -- even though the
    # READ side already correctly refused to serve a dry-run entry back to a
    # real run (and vice versa). The write path needed the same separation.
    #
    # Isolated from the project's real cache file: DECISION_CACHE_FILE is
    # monkeypatched to a temp path for the duration of this test so nothing
    # here touches cache/llm_decisions.json.
    original_cache_file = pipeline_module.DECISION_CACHE_FILE
    with tempfile.TemporaryDirectory() as tmp:
        pipeline_module.DECISION_CACHE_FILE = Path(tmp) / "llm_decisions.json"
        try:
            dataset = _empty_dataset()
            message = {
                "message_id": "msg_001", "user_id": "u_x", "sender_user_id": "u_y",
                "conversation_type": "personal", "group_id": "", "business_id": "",
                "message_text": "hello there, how are you doing today friend", "media_type": "",
                "media_id": "", "forwarded_count": "0", "created_at": "2026-07-20 09:00",
            }

            router = Router(dataset, client=None, dry_run=False)
            # Seed a fake "real" cached decision directly, as a prior real run would have.
            safety = evaluate_safety(dataset, message, None)
            ctx = build_context(dataset, message, None, safety)
            ctx_hash = _context_hash(ctx.message, ctx.extracted_media_text)
            router._decision_cache["msg_001"] = {
                "_ctx_hash": ctx_hash,
                "_dry_run": False,
                "decision": {
                    "message_type": "personal", "action": "digest", "reason": "real cached decision",
                    "confidence": 0.75, "evidence_message_ids": "none",
                },
            }

            # Same Router instance (same in-memory cache dict, as a shared cache
            # FILE would give two separate runs), now switched into dry-run for
            # the same message_id.
            router.dry_run = True
            router.route(message)

            assert router._decision_cache["msg_001"]["decision"]["reason"] == "real cached decision", (
                "dry-run write clobbered the previously cached real decision"
            )
            assert "msg_001#dry_run" in router._decision_cache, "dry-run entry should live under its own key"
        finally:
            pipeline_module.DECISION_CACHE_FILE = original_cache_file


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
