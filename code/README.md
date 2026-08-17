# Message Notification Router

Solution for HackerRank Orchestrate (August 2026) -- Message Notification Router.

## Setup

```bash
cd code
python3 -m venv .venv && source .venv/bin/activate   # optional but recommended
pip install -r requirements.txt

cp .env.example .env
# edit .env and set ANTHROPIC_API_KEY (from https://console.anthropic.com)
```

You'll also need `ffmpeg` on your PATH for voice-note transcription:
`apt install ffmpeg` (Linux) or `brew install ffmpeg` (macOS).

## Run

Always run the evaluation harness first -- it's fast, uses only the 30 solved
rows in `dataset/sample_messages.csv`, and tells you whether the system is
actually working before you spend time/tokens on the full 110-row run.

```bash
# 1. Wiring check, no API key or network needed:
python evaluation/main.py --dry-run

# 2. Real evaluation against the solved samples:
python evaluation/main.py
# add --verbose to see every row's diff, not just mismatches

# 3. Full run -- writes ../dataset/output.csv:
python main.py

# Useful flags on main.py:
python main.py --limit 10          # smoke test on the first 10 messages
python main.py --dry-run           # wiring check, no API key needed
python main.py --output somewhere.csv --dataset-dir /path/to/dataset
```

## Tests

```bash
python tests/test_safety.py
python tests/test_retrieval.py
python tests/test_load_signal.py
python tests/test_context.py
python tests/test_output_alignment.py
python tests/test_reasoning.py
python tests/test_pipeline.py
python tests/test_safety_invariants.py
# or: pytest tests/
```

62 pure-logic unit tests total (19 safety + 5 retrieval + 5 situational-load
+ 4 context/engagement + 4 output-alignment + 17 reasoning + 1 pipeline
cache + 7 safety-invariant spot-checks) -- no API key, no network,
sub-second to run (the reasoning retry tests fake the Anthropic client
entirely, so backoff/retry logic is exercised without a real network call
or a real delay).
`test_safety.py` is what to point at in the AI Judge interview for "how do
you know the injection defense (and now the domain-lookalike override)
works independent of what the model would have done"; several cases are run
directly against real dataset rows and field values (`msg_018`, `msg_108`,
`business_062`/Chase, `business_106`/AWS -- the one legitimate
brand-name-mismatch counter-example on file). `test_load_signal.py` includes
a structural test asserting the `LoadSignal` dataclass has no field shaped
like a suppression flag -- it documents the "narrow" design promise (see
Pipeline step 4 below), not just describes it. `test_context.py` covers the
engagement-scoring nudges (reaction-time, last-reply-at recency) in
isolation from the LLM call. `test_output_alignment.py` guards against
`output.csv` ever being written in a different row order than
`messages.csv` -- the grader compares positionally, not by `message_id`
join, so a sorted or reordered write would silently tank the score; it
deliberately corrupts the write order in one case and asserts the
pre-write guardrail in `main.py` fires. `test_reasoning.py` covers both the
behavioral-evidence invariant (with a negation-awareness regression test)
and the retry/fallback machinery -- transient-error backoff, malformed-
response reformatting, and the conservative fallback decision, all against
a scripted fake client. `test_pipeline.py` guards the decision cache: a
dry-run wiring check must never overwrite a previously cached real
decision for the same message_id, so real and dry-run entries are stored
under distinct cache keys.

## How it's built

```
code/
├── main.py                  entry point -- routes dataset/messages.csv -> output.csv
├── evaluation/main.py       harness -- scores predictions against sample_messages.csv
├── tests/
│   ├── test_safety.py       unit tests for the deterministic safety rules
│   ├── test_retrieval.py    unit tests for sparse-text detection + thread-context retrieval
│   ├── test_load_signal.py  unit tests for the situational-load signal
│   ├── test_context.py      unit tests for engagement-scoring nudges (reaction-time, recency)
│   ├── test_output_alignment.py  regression test for output.csv row-order alignment
│   ├── test_reasoning.py    behavioral-evidence invariant + retry/fallback/structured-output tests
│   ├── test_pipeline.py     regression test for the decision-cache dry-run/real key separation
│   ├── test_safety_invariants.py    adversarial spot-checks for SAFETY_INVARIANTS.md's HARD rules
│   └── adversarial_llm_spotcheck.py  real-API spot-checks for the PROMPT-level rules (manual, not in the fast suite)
├── SAFETY_INVARIANTS.md     the small number of rules that hold on every message, and how each is enforced
└── router/
    ├── config.py            all tunable constants + env loading, one place
    ├── data_loader.py        loads every dataset CSV, exposes join/lookup helpers
    ├── safety.py             deterministic pre-filter: injection, scam-keyword categories,
    │                         chain-letter, compromised-sender, vague-intro, group-invite,
    │                         mass-forward detection
    ├── context.py             per-message context assembly + engagement scoring
    ├── media.py               OCR (Claude vision) + ASR (local Whisper), cached per media_id
    ├── retrieval.py           evidence candidates: structural filter + lexical similarity,
    │                         plus recency-ranked thread context for sparse-text messages
    ├── load_signal.py         situational-load signal: same-category bundling / same-thread
    │                         promotion only, never a suppression signal (see below)
    ├── reasoning.py           the one LLM call: classify, write reason, pick evidence
    └── pipeline.py            wires all of the above together, caches LLM decisions
```

### Pipeline per message

1. **Media extraction** (`media.py`) -- image/voice messages get OCR'd or
   transcribed once per unique `media_id` and cached, since several media
   files are reused across many messages and recipients. Every image file
   in `dataset/images.csv` is named `*.jpg` regardless of its actual
   encoding (the real mix includes JPEG, PNG, WebP, and AVIF) -- the real
   format is sniffed via Pillow rather than trusted from the extension, and
   AVIF (not accepted by Claude's vision input) is transparently re-encoded
   to PNG before the API call.
2. **Safety pre-filter** (`safety.py`) -- pure regex/lookup rules, no LLM.
   Three conditions are *hard overrides* that skip the LLM entirely and
   write `mute`/`scam` directly:
   - **Injection detection**: the message body tries to instruct the router
     directly (fake "routing override" / "system note" / "assistant
     instruction" language, or fake structured fields like
     `verified_business=true` / `action=notify`). Content is always treated
     as data, never as a command to the model.
   - **Compromised/impersonated sender**: this exact sender has previously
     sent this recipient a message with an overlapping scam-content
     signature that got reported or muted. A trusted role (admin, verified
     business) does not override this.
   - **Domain-lookalike / brand impersonation**: an *unverified* business
     sending from a domain that doesn't match its own `official_domain`,
     registered under `domain_lookalike_max_age_days` (60) ago. Validated
     against 7 real brand-impersonation rows in this dataset (fake Chase,
     HDFC, Amazon, Talabat, RazorpayX -- all 2-17 days old); the age gate
     exists because the one legitimate domain mismatch on file (a verified
     business using a link-shortener) is 3368 days old. A mismatch that
     doesn't clear this bar, or a business with no `official_domain` on
     file at all but a young unverified sending domain, sets a softer
     `domain_mismatch_soft` signal instead of silently dropping it.
   Softer signals feed the LLM as context rather than overriding it: 10
   scam-keyword categories (OTP requests, payment pressure, click/scan asks,
   account-risk language, screenshot asks, prize/lottery claims, "forward me
   the code you got by mistake" requests, fake WhatsApp-upgrade links,
   impersonated-platform-support language, and upfront-fee job offers),
   chain-letter phrasing, group-invite phrasing, a vague cold-start opener
   with no explicit ask, mass-forward detection (`forwarded_count` at or
   above a configurable threshold), and the `domain_mismatch_soft` signal
   above. Notably, a business's `brand_name` differing from its
   `display_name` is surfaced as plain context but deliberately does *not*
   get its own flag: 26 of 110 businesses have this mismatch, but a clean
   verified counter-example (a cloud-security vendor operating under a
   shortened brand name, matching domain, verified) proves brand-name
   mismatch alone isn't a reliable signal on its own.

   See [`SAFETY_INVARIANTS.md`](SAFETY_INVARIANTS.md) for the short list of
   rules that hold on every message regardless of other signals (content is
   never an instruction, situational load can never move anything into or
   out of mute, a mute decision can't claim unsupported recipient behavior,
   etc.), each marked by whether it's enforced in code or in the prompt, and
   spot-checked adversarially in `tests/test_safety_invariants.py` (fast,
   no API key) and `tests/adversarial_llm_spotcheck.py` (real API calls,
   run manually).
3. **Context assembly** (`context.py`) -- joins user/group/business/history
   context and computes an engagement score per relationship (explicit
   mute/opt-out is a strong negative signal distinct from passive
   disengagement, which only shifts the *default*, not a hard rule). Two
   small, deliberately narrow nudges sit on top of the primary open/reply/
   dismiss counts, never replacing them: a business reply within 14 days
   lifts a declining/disengaged 30-day rollup up to "engaged" (recency over
   a diluted average), and an average `reaction_time_minutes` of 10 or under
   / 90 or over shifts the personal-engagement score by +/-0.05. A general
   notification-fatigue baseline (`recipient_baseline_dismissal_rate`, from
   `daily_notification_summary.csv`) is also surfaced to the LLM -- that
   file's date range ends before `messages.csv` starts, so it can only speak
   to "how fatigued is this person overall," never to anything happening
   around the current message (that's what situational-load, below, is for).
   A handful of fields (`groups.csv`'s `admin_count`/`created_at`/
   `messages_30d`, the recipient's own `joined_at` in that group,
   `business_accounts.csv`'s `brand_name`/`category`/`messages_sent_30d`)
   are surfaced to the LLM as descriptive context without a dedicated rule
   attached -- checked each against the real data first (e.g. newer groups
   don't correlate with anything risky here) and didn't find a validated
   case to build a hard signal around, so they stay contextual rather than
   invented.
4. **Evidence retrieval** (`retrieval.py`, `load_signal.py`) -- structural
   filter (same sender/group/business) + lexical similarity (rapidfuzz), no
   embeddings API required. When a message's own text is sparse (a bare
   emoji caption, under ~4 words), a second recency-ranked pass
   (`find_thread_context`) pulls in the last few same-thread messages so the
   LLM has something to interpret the media/caption against; evidence
   candidates are tagged `similarity` or `recency` so the model knows which
   kind of match it's looking at. A separate situational-load signal
   (`load_signal.py`) tracks how many same-category messages arrived for
   this user recently -- deliberately narrow: it can only support batching
   routine content into a digest or promoting an already-engaged thread's
   timely follow-up, and is structurally incapable of suppressing anything
   (no field on it is shaped like a suppression flag, and the prompt
   explicitly forbids using it to downgrade personal/urgent content or to
   excuse a scam/spam classification into `notify`). The LLM only ever
   picks evidence ids from the retrieved shortlist.
5. **LLM reasoning** (`reasoning.py`) -- one bounded Claude API call per
   message when no hard override applies: message content and OCR/ASR
   output are placed in an explicitly labeled, untrusted data block. The
   response shape is enforced at the API level via structured outputs
   (`output_config` with a JSON schema), with a second, independent parsing
   layer that never assumes the schema is the only thing standing between it
   and a malformed response. Transient API errors (rate limit, connection,
   5xx) retry with exponential backoff (`llm_max_retries`,
   `llm_retry_base_delay_seconds`); a response that still fails to parse
   gets one reformat retry; a decision that mutes while making an
   unsupported behavioral claim gets one corrective retry. If a row
   exhausts all of that, it falls back to a conservative, clearly-labeled
   `digest`/`unknown` decision rather than crashing the batch or silently
   guessing -- one bad row never costs the other 109.
6. **Calibration + output** (`pipeline.py`) -- confidence is capped in two
   narrow cases, never a fixed value: a vague cold-start opener with no
   ask caps at `vague_intro_confidence_cap`, and a message that's sparse in
   *both* its own text and its media extraction *and* has no thread history
   to lean on caps at `sparse_no_grounding_confidence_cap` (any one of those
   three being substantive is enough to leave the model's own confidence
   alone -- a clear QR code with no caption isn't penalized just for being
   media-only). LLM decisions are cached to disk keyed by message_id + a
   hash of the assembled context (and tagged dry-run vs. real, so a wiring
   check never poisons a real run's cache), so reruns with unchanged input
   never re-call the API.

### Known simplifications (worth naming up front, not hiding)

- Retrieval is lexical, not embedding-based -- reasonable at this dataset's
  size (~400 historical messages), where the structural filter (same
  sender/group/business) already does most of the real work.
- Voice transcription runs locally (faster-whisper) since the Claude
  Messages API doesn't accept raw audio input -- no second API account
  needed, but transcription quality depends on the local model size
  (`ROUTER_WHISPER_MODEL_SIZE`, default `base`).
- The safety pre-filter's keyword matching is regex-based, so it can
  occasionally pull in a defensive/warning message that happens to share
  vocabulary with a scam (e.g. "we never ask for your OTP") as evidence,
  since regex doesn't reason about negation. This is a direct tradeoff of
  keeping the injection/compromised-sender rule deterministic and
  independent of a model call -- worth naming in the interview rather than
  claiming perfect precision.
- Engagement scoring uses the 30-day rollup fields in the dataset, which
  only approximates a true trend; a per-day time series per relationship
  would support the "relationship repair" case more precisely than a
  single rollup snapshot does.
