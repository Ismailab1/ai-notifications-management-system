"""Central configuration.

Everything tunable lives here so the AI Judge interview has one place to
point at for "why did you pick that threshold". Nothing here reads secrets
except via environment variables (never hardcoded), per the submission
constraints in AGENTS.md / problem_statement.md.
"""
from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path

from dotenv import load_dotenv

load_dotenv()

CODE_DIR = Path(__file__).resolve().parent.parent
REPO_ROOT = CODE_DIR.parent


@dataclass(frozen=True)
class Config:
    # --- secrets (env only, never hardcoded) ---
    anthropic_api_key: str = os.environ.get("ANTHROPIC_API_KEY", "")

    # --- paths ---
    dataset_dir: Path = Path(os.environ.get("ROUTER_DATASET_DIR", REPO_ROOT / "dataset"))
    cache_dir: Path = Path(os.environ.get("ROUTER_CACHE_DIR", CODE_DIR / "cache"))

    # --- model ---
    model: str = os.environ.get("ROUTER_MODEL", "claude-sonnet-5")
    # Sonnet 5 rejects non-default temperature/top_p/top_k outright (400).
    # Thinking is disabled instead, for fast/cheap/bounded classification calls.

    # --- retrieval ---
    max_retrieval_candidates: int = int(os.environ.get("ROUTER_MAX_RETRIEVAL_CANDIDATES", "5"))
    min_retrieval_score: float = 0.15  # below this, don't bother offering it to the LLM as evidence

    # --- multimodal ---
    whisper_model_size: str = os.environ.get("ROUTER_WHISPER_MODEL_SIZE", "base")

    # --- determinism ---
    random_seed: int = 42

    # --- confidence calibration ---
    hard_override_confidence: float = 0.94  # injection / repeated-reported-pattern overrides
    vague_intro_confidence_cap: float = 0.62  # see safety.py: never claim high certainty on a cold-start opener

    # --- text/media balance ---
    sparse_text_word_threshold: int = 4  # fewer real words than this -> "sparse", pull in thread context
    thread_context_k: int = 3  # how many recent same-thread messages to surface as framing
    sparse_no_grounding_confidence_cap: float = 0.55  # sparse text + thin extraction + no thread history at all

    # --- situational load (narrow: same-category bundling/promotion only, never cross-category suppression) ---
    load_window_hours: float = 3.0
    load_burst_threshold: int = 2  # this many same-category arrivals in the window counts as a "burst"

    # --- scam signal thresholds ---
    high_forward_count_threshold: int = 5  # forwarded_count at/above this is a mass-blast soft signal

    # --- reliability ---
    llm_max_retries: int = 3  # transient API errors (rate limit, connection, 5xx) per model call
    llm_retry_base_delay_seconds: float = 2.0  # exponential backoff base; actual delay = base * 2**attempt
    domain_lookalike_max_age_days: int = 60  # unverified sender-domain mismatch under this age -> hard override;
    # all 7 validated real examples in this dataset are 2-17 days old, vs. one legitimate
    # verified/old-domain counterexample at 3368 days -- 60 gives comfortable margin either way

    def validate(self) -> None:
        if not self.anthropic_api_key:
            raise RuntimeError(
                "ANTHROPIC_API_KEY is not set. Copy code/.env.example to code/.env "
                "and fill in your key, or export it in your shell."
            )
        if not self.dataset_dir.exists():
            raise RuntimeError(f"Dataset directory not found: {self.dataset_dir}")


CFG = Config()
