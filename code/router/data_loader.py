"""Loads dataset/*.csv and exposes lookup helpers.

Design choice: every CSV is read with dtype=str and na_filter=False, so a
blank cell is always '' rather than NaN. This keeps every downstream
consumer (safety rules, context assembly, retrieval) free of NaN-checking
boilerplate -- we convert to int/float only at the point a field is
actually used numerically.

Only reads from dataset/, per the submission constraint that organizer-only
files must never be used.
"""
from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

import pandas as pd

from .config import CFG


def _read(path: Path) -> list[dict]:
    if not path.exists():
        raise FileNotFoundError(f"Expected dataset file missing: {path}")
    df = pd.read_csv(path, dtype=str, na_filter=False, keep_default_na=False)
    return df.to_dict(orient="records")


@dataclass
class Dataset:
    # IMPORTANT: must stay in dataset/messages.csv's original file order.
    # `_read()` uses pandas.read_csv().to_dict(orient="records"), which does
    # not reorder rows -- this list is exactly the file's row order as read.
    # The grader compares output.csv against this file POSITIONALLY, not by
    # joining on message_id, so nothing upstream (main.py included) may
    # re-sort this list before it's used to determine the write order.
    messages: list[dict]
    sample_messages: list[dict]
    users: dict[str, dict]
    groups: dict[str, dict]
    group_members: dict[tuple[str, str], dict]          # (group_id, user_id) -> row
    business_accounts: dict[str, dict]
    user_business_history: dict[tuple[str, str], dict]  # (user_id, business_id) -> row
    message_history: list[dict]
    message_events: dict[tuple[str, str], dict]          # (user_id, message_id) -> row
    images: dict[str, str]                                # image_id -> file_path
    voice_notes: dict[str, str]                           # voice_note_id -> file_path
    daily_notification_summary: list[dict]

    # derived indices, built in __post_init__
    _history_by_user: dict[str, list[dict]] = field(default_factory=dict, repr=False)
    _history_by_sender: dict[str, list[dict]] = field(default_factory=dict, repr=False)
    _history_by_id: dict[str, dict] = field(default_factory=dict, repr=False)
    _daily_summary_by_user: dict[str, list[dict]] = field(default_factory=dict, repr=False)

    def __post_init__(self) -> None:
        by_user: dict[str, list[dict]] = defaultdict(list)
        by_sender: dict[str, list[dict]] = defaultdict(list)
        by_id: dict[str, dict] = {}
        for row in self.message_history:
            by_id[row["message_id"]] = row
            if row.get("user_id"):
                by_user[row["user_id"]].append(row)
            if row.get("sender_user_id"):
                by_sender[row["sender_user_id"]].append(row)
        self._history_by_user = dict(by_user)
        self._history_by_sender = dict(by_sender)
        self._history_by_id = by_id

        by_daily_user: dict[str, list[dict]] = defaultdict(list)
        for row in self.daily_notification_summary:
            if row.get("user_id"):
                by_daily_user[row["user_id"]].append(row)
        self._daily_summary_by_user = dict(by_daily_user)

    # ---- lookups -------------------------------------------------------

    def user(self, user_id: str) -> Optional[dict]:
        return self.users.get(user_id)

    def group(self, group_id: str) -> Optional[dict]:
        return self.groups.get(group_id) if group_id else None

    def group_member(self, group_id: str, user_id: str) -> Optional[dict]:
        if not group_id:
            return None
        return self.group_members.get((group_id, user_id))

    def business(self, business_id: str) -> Optional[dict]:
        return self.business_accounts.get(business_id) if business_id else None

    def business_history(self, user_id: str, business_id: str) -> Optional[dict]:
        if not business_id:
            return None
        return self.user_business_history.get((user_id, business_id))

    def event(self, user_id: str, message_id: str) -> Optional[dict]:
        return self.message_events.get((user_id, message_id))

    def history_for_recipient(self, user_id: str) -> list[dict]:
        """All past messages this specific user received (any sender)."""
        return self._history_by_user.get(user_id, [])

    def history_for_sender(self, sender_user_id: str) -> list[dict]:
        """All past messages this specific sender has sent (any recipient).

        Used by the compromised/impersonated-sender check: we need this
        sender's own track record, not the general corpus.
        """
        if not sender_user_id:
            return []
        return self._history_by_sender.get(sender_user_id, [])

    def history_between(self, sender_user_id: str, user_id: str) -> list[dict]:
        """Past messages from this exact sender to this exact recipient."""
        return [h for h in self.history_for_sender(sender_user_id) if h.get("user_id") == user_id]

    def daily_summary_for_user(self, user_id: str) -> list[dict]:
        """This user's daily_notification_summary.csv rows (2026-07-04 to
        2026-07-17 -- entirely before messages.csv starts on 2026-07-18).
        A general notification-fatigue baseline, not a live signal; see
        context.py::compute_baseline_dismissal_rate."""
        return self._daily_summary_by_user.get(user_id, [])

    def media_path(self, media_type: str, media_id: str) -> Optional[Path]:
        if not media_id:
            return None
        rel = self.images.get(media_id) if media_type == "image" else self.voice_notes.get(media_id)
        if not rel:
            return None
        return CFG.dataset_dir / rel


def load_dataset(dataset_dir: Optional[Path | str] = None) -> Dataset:
    d = Path(dataset_dir) if dataset_dir else CFG.dataset_dir

    users = {r["user_id"]: r for r in _read(d / "users.csv")}
    groups = {r["group_id"]: r for r in _read(d / "groups.csv")}
    group_members = {(r["group_id"], r["user_id"]): r for r in _read(d / "group_members.csv")}
    business_accounts = {r["business_id"]: r for r in _read(d / "business_accounts.csv")}
    user_business_history = {(r["user_id"], r["business_id"]): r for r in _read(d / "user_business_history.csv")}
    message_events = {(r["user_id"], r["message_id"]): r for r in _read(d / "message_events.csv")}
    images = {r["image_id"]: r["file_path"] for r in _read(d / "images.csv")}
    voice_notes = {r["voice_note_id"]: r["file_path"] for r in _read(d / "voice_notes.csv")}

    return Dataset(
        messages=_read(d / "messages.csv"),
        sample_messages=_read(d / "sample_messages.csv"),
        users=users,
        groups=groups,
        group_members=group_members,
        business_accounts=business_accounts,
        user_business_history=user_business_history,
        message_history=_read(d / "message_history.csv"),
        message_events=message_events,
        images=images,
        voice_notes=voice_notes,
        daily_notification_summary=_read(d / "daily_notification_summary.csv"),
    )
