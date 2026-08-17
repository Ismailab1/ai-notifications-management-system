# Safety invariants

A small number of rules this router must hold on *every* message, regardless
of what else is going on -- engagement history, situational load, how
confident the model feels. Most of the system (message_type, notify vs.
digest, confidence) is deliberately left to per-message reasoning. These
aren't that: they're the rules that don't get to be argued with.

Each invariant below is marked by how it's enforced:

- **HARD** -- checked in plain Python before or independent of any LLM call.
  Cannot be talked out of by clever phrasing, because no model call is in
  the loop for that decision.
- **PROMPT** -- stated as an explicit instruction to the model and verified
  empirically against real data, but not mechanically guaranteed the way a
  HARD invariant is. Worth naming as a limitation, not hiding.

## I1 -- Message content is never treated as an instruction to the router

Everything in `message_text` and any OCR/ASR-extracted text is data to
classify, never a command -- no matter what it claims to be (a "system
note", a "routing override", an "assistant instruction"). **HARD**:
`safety.INJECTION_PATTERNS` catches content that tries to talk the router
into a specific action (`set action=`, `confidence=1.0`, "ignore previous
routing instructions", etc.); a hit skips the LLM entirely and writes the
row directly. The `<message_data>` framing in `reasoning.SYSTEM_PROMPT`
restates the same rule for the softer cases the regex doesn't catch.

## I2 -- An injection attempt is itself scam evidence, not something to comply with

The content asking to be marked `notify` with `confidence=1.0` gets muted --
the exact opposite of what it demanded. **HARD**: `SafetySignals.hard_override`
always pairs an injection hit with `action="mute", message_type="scam"`,
regardless of what action the injected text asked for.

## I3 -- A recognized scam signal is never overridden into `notify` by claimed urgency, a stated deadline, or nearby message volume

A scam message doesn't get promoted just because it says "act now" or
because several similar messages arrived in a burst. **HARD** for the three
hard-override cases (injection, compromised sender, domain-lookalike) --
they bypass the LLM, so urgency language inside them has no path to
influence the outcome at all. **PROMPT** for the softer scam/spam cases the
LLM decides on its own: `SYSTEM_PROMPT` explicitly forbids using
situational_load to justify notify for anything classified scam/spam/risky.
Verified empirically: 0 of 110 real rows in this dataset route a
scam/spam-typed message to notify (see `evaluation/main.py` output-check
run in the P4 commit).

## I4 -- situational_load can never move a message into or out of `mute`

Message-arrival volume is a scheduling signal, not a content judgment -- it
can only nudge between notify and digest on content already judged safe.
**HARD**, structurally: `load_signal.LoadSignal` has no field shaped like a
suppression flag at all, so there's nothing for that nudge to attach to even
in principle (see `tests/test_load_signal.py`'s
`test_load_signal_object_exposes_no_suppression_field`). **PROMPT** for the
digest<->notify nudge itself being used correctly.

## I5 -- situational_load can never suppress or downgrade a personal message, or any message whose own content already signals real urgency

Symmetric to I4/I3: a cluster of unrelated recent messages is never a reason
to sit on something that's actually urgent or personal. **PROMPT**:
`SYSTEM_PROMPT` states this explicitly ("NEVER use situational_load to
suppress or downgrade a personal message, or any message whose own content
already indicates real urgency").

## I6 -- A MUTE decision can never claim a specific past recipient behavior it can't back up

If a `reason` says this recipient dismissed, reported, or muted something
before, that claim must be grounded in a real `message_events.csv` row for
*this* recipient showing *that* action -- not a similar-looking row, not a
different recipient's reaction, not an unsupported assertion. **HARD**:
`reasoning._validate_behavioral_evidence` checks every mute decision's
`reason` against `dataset.event()`, retries once with the specific
mismatch named, and if the retry still doesn't hold up, ships a rewritten
reason that drops the claim rather than an unsupported one (see
`tests/test_reasoning.py`).

## I7 -- evidence_message_ids can never contain an invented id

The model is only allowed to cite ids from the retrieved shortlist actually
offered to it for this message. **HARD**: `reasoning._parse_response`
filters every cited id against `valid_ids` (the shortlist), dropping
anything else silently down to `"none"` rather than trusting the model not
to hallucinate one.

## I8 -- A row that can't be trusted is never guessed at

If the model call fails after retries, or its response still can't be
parsed after a reformat attempt, the row is never left unrouted and never
gets a fabricated high-confidence answer. **HARD**:
`reasoning._fallback_decision` returns a clearly-labeled, conservative
`digest`/`unknown` decision at a capped confidence, and every failure path
in `reasoning.reason_about_message` falls back to it rather than raising
(see `tests/test_reasoning.py`'s retry/fallback tests).

---

## Spot-checking these

- `tests/test_safety_invariants.py` -- fast, pure-Python adversarial checks
  for every **HARD** invariant (I1, I2, I3-hard, I4, I6, I7, I8). No API key,
  no network, part of the standard test run.
- `tests/adversarial_llm_spotcheck.py` -- one hand-written adversarial
  message per **PROMPT** invariant (I3-soft, I5), run against the real model
  through `reason_about_message`. Requires `ANTHROPIC_API_KEY` and costs a
  few real API calls, so it's kept out of the default fast suite -- run it
  manually before a submission, not on every change:

  ```bash
  python tests/adversarial_llm_spotcheck.py
  ```
