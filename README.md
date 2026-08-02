# AI Notifications Management System

A personal project built for **HackerRank Orchestrate (August 2026)**, a 24-hour solo hackathon. This README covers the problem, what I built, the decisions and tradeoffs behind it, and how it performed. For setup/run instructions and a full architectural breakdown, see [`code/README.md`](./code/README.md).

> **Attribution**: the challenge, problem statement, and dataset are © HackerRank / [Interview Street](https://github.com/interviewstreet), from their starter repository **[interviewstreet/hackerrank-orchestrate-august26](https://github.com/interviewstreet/hackerrank-orchestrate-august26)**. This repo is my personal fork of that starter, submitted solo as my hackathon entry. The `code/` implementation is my own original work; `dataset/`, `problem_statement.md`, and `AGENTS.md` are organizer-provided material, kept here for reproducibility rather than presented as mine. See [Attribution & License](#attribution--license) below.

---

## The Problem

WhatsApp is noisy. A single user's message stream mixes family chats, school and society notices, work pings, business promotions, image posters, voice notes — and scams. Treating every message the same produces two failure modes: real emergencies get buried in noise, and unwanted or dangerous messages get through anyway.

The challenge: build a system that reads every incoming message in `dataset/messages.csv` — text, image, or voice note — and, using the provided user/group/business/history context, decides for **that specific recipient**:

- `notify` — interrupt them now
- `digest` — useful, but it can wait
- `mute` — low-value, repetitive, unwanted, or unsafe

Output one row per message: `message_id, action, message_type, reason, confidence, evidence_message_ids`. Full spec in [`problem_statement.md`](./problem_statement.md).

The dataset was deliberately adversarial in places — messages designed to look like routing instructions to the system itself, senders with mixed trust histories, and the same image poster producing opposite correct outcomes for different recipients. Getting a high score meant actually reasoning about context, not pattern-matching keywords.

---

## What I Built

A pipeline that runs each message through five stages before producing a row:

```text
media extraction → safety pre-filter → context assembly → evidence retrieval → LLM reasoning → calibration
```

1. **Media extraction** — image posters go through Claude's vision input directly (real file format sniffed via Pillow, since every image in the dataset is misleadingly named `.jpg` regardless of actual encoding); voice notes are transcribed locally with `faster-whisper`, so no second API account is needed. Both are cached per `media_id`, not per message, since the same poster or voice note is often reused across many recipients.
2. **Deterministic safety pre-filter** — pure regex/lookup rules, no LLM, unit-tested in isolation. Three conditions are *hard overrides* that skip the LLM entirely: prompt-injection attempts (a message trying to instruct the router directly — treated as evidence of a scam, not something to comply with), a sender with a track record of reported/muted scam-pattern messages to this exact recipient, and brand-impersonation domain lookalikes (an unverified business sending from a recently-registered domain that doesn't match its own official one). Softer signals — scam-keyword categories, chain-letter phrasing, mass-forwarding, vague cold-start openers — feed the LLM as context instead of overriding it.
3. **Context assembly** — joins user, group, business, and historical-interaction data, and computes a graded engagement score per relationship (explicit mute/opt-out is a much stronger signal than passive disengagement, which only shifts the *default* for routine content).
4. **Evidence retrieval** — structural filtering (same sender/group/business) plus lexical similarity, with a recency-ranked fallback for messages whose own text is too sparse to compare (a bare emoji caption on a photo, for instance). A narrow "situational load" signal tracks message density without ever being allowed to suppress content on its own.
5. **LLM reasoning** — one bounded Claude call per message when no hard override applies, with all message content and OCR/ASR output placed in an explicitly untrusted data block.
6. **Calibration** — confidence gets capped, not the action, when a message is genuinely under-informed (no text, thin media extraction, no thread history to lean on).

---

## Key Decisions & Tradeoffs

**Deterministic hard overrides, not LLM judgment, for the highest-stakes calls.** Prompt injection and confirmed compromised senders skip the model entirely and write the row directly. A model can be argued with by a sufficiently clever message; a fixed rule can't. This meant a chunk of the engineering effort went into finding *real* gaps in those rules against the actual data (e.g. a lottery-scam message with zero keyword-category matches, or an injection attempt phrased as fake structured metadata) rather than trusting the rules by inspection.

**Lexical retrieval over embeddings.** At this dataset's scale (~400 historical messages), structural filtering by sender/group/business already does most of the real work; adding an embeddings API would have meant a second account and dependency for marginal gain. Documented as a deliberate scope tradeoff, not an oversight.

**A narrow situational-load signal, not a "focus mode."** I considered inferring a general "busy period" from message timestamps and using it to suppress notifications outright, including personal ones. Rejected: an inferred burst from timestamps is much weaker evidence than an explicit toggle would be, and letting a weak inference override a strong content-level signal (a message that's actually urgent) repeats the exact mistake the safety layer exists to avoid. The shipped version can only support batching routine content into a digest or timing an already-engaged thread's follow-up — it is structurally incapable of suppressing anything, which is enforced with a unit test asserting the signal object has no field shaped like a suppression flag, not just a docstring promise.

**Recency over rollups for engagement.** The dataset only provides 30-day rollup counts, which dilute a real signal (a reply yesterday, a sender someone stopped responding to two weeks ago) into a flat average. Small, explicitly bounded nudges — a recent reply lifts a declining relationship back to "engaged," fast vs. slow historical reaction time shifts the score ±0.05 — sit on top of the primary counts rather than replacing them.

**Every rule earned its place against real rows, not hypotheticals.** Several late additions (a fourth and fifth injection pattern, a domain-lookalike age threshold, a soft signal for businesses with no domain on file at all) came from deliberately auditing whether every provided data field was actually being used, then checking each candidate rule against real message rows before writing it — including finding and preserving a legitimate counter-example (a verified vendor operating under a shortened brand name) that would have been a false positive under a naive version of the rule.

---

## Results

- **32 unit tests** (safety rules, retrieval helpers, the situational-load signal, engagement-scoring nudges) run independent of any LLM call — deterministic, sub-second, no API key required.
- **Sample evaluation** (30 hand-solved rows): **29/30 action accuracy**, 21/30 exact message-type match (most misses are defensible label choices — e.g. `event` vs. `urgent` — not wrong outcomes), 23/30 evidence overlap, average confidence deviation 0.089 from the solved values.
- **Full run** (110 messages): 55 muted, 37 notified, 18 digested — with **zero** scam/spam-classified messages ever routed to `notify`, verified as an explicit invariant check on every run.

A good share of the real engineering here was finding and fixing bugs that only running the *full* dataset — not just the 30 sample rows — surfaced: an LLM decision cache that didn't distinguish dry-run placeholders from real API output, a Windows-only crash writing OCR text containing a ₹ symbol, an image pipeline trusting a misleading `.jpg` extension on files that were actually PNG/WebP/AVIF, and a prompt-guidance regression where the situational-load signal was briefly (and wrongly) letting a busy thread override a scam classification into `notify` — caught specifically because two affected rows weren't in the 30-row sample set.

---

## Process

Built solo, with Claude Code (Anthropic's coding agent) as a pair-programming tool for implementation — the challenge explicitly evaluates the system built, not the mechanics of how the code was typed. The full working transcript is logged per the challenge's own `AGENTS.md` requirement. The process was deliberately evidence-first throughout: every safety rule, threshold, and scoring nudge was checked against real dataset rows before being implemented, and several were revised or scoped narrower after that check contradicted the initial hypothesis (see [Key Decisions & Tradeoffs](#key-decisions--tradeoffs)).

---

## Repository Layout

```text
.
├── code/                    my solution (see code/README.md for setup + architecture)
├── dataset/                 organizer-provided challenge data (messages, users, groups, media, ...)
├── problem_statement.md     original challenge spec, as provided
└── AGENTS.md                organizer-provided AI-tool rules + transcript-logging requirement
```

---

## Attribution & License

This project was built for and submitted to **HackerRank Orchestrate (August 2026)**, a hackathon organized by [HackerRank / Interview Street](https://github.com/interviewstreet). The challenge brief, dataset, and starter scaffold originate from their repository:

**[interviewstreet/hackerrank-orchestrate-august26](https://github.com/interviewstreet/hackerrank-orchestrate-august26)**

That upstream repository carries no explicit license (verified via the GitHub API at the time of writing); this fork is provided for the purpose the organizers built it for — a public hackathon starter template participants build their own solution on top of and may keep as their submission record. `problem_statement.md`, `AGENTS.md`, and the contents of `dataset/` are reproduced here unmodified (aside from the generated `dataset/output.csv` predictions) as organizer-authored material, not original work of mine.

Everything under `code/` — the routing pipeline, safety rules, retrieval logic, prompts, and tests — is my own original implementation, built solo during the 24-hour window.
