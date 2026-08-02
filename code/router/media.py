"""Multimodal extraction for image and voice-note messages.

Key design decision (see conversation history / interview notes): several
media_ids are reused across many messages and many recipients -- e.g. one
poster image sent to a dozen different users. Extraction is keyed and
cached by media_id, never by message_id, so a given image or voice note is
only ever OCR'd / transcribed once, no matter how many rows reference it.
This is what keeps the pipeline cheap and deterministic on rerun.

Images go through Claude's vision input (no separate OCR service).
Voice notes go through a local, open-source ASR model (faster-whisper) --
the Claude Messages API does not accept raw audio input, and this also
means no second API key / vendor account is required.
"""
from __future__ import annotations

import base64
import io
import json
from pathlib import Path
from typing import Optional

from PIL import Image

from .config import CFG
from .data_loader import Dataset

# Every file in dataset/images.csv is named *.jpg regardless of its actual
# encoding -- the underlying bytes are a mix of real JPEG, PNG, WebP, and
# AVIF. The Claude vision API rejects a mismatched media_type outright (and
# doesn't accept AVIF at all), so the real format has to be sniffed from the
# file content, never assumed from the extension.
_CLAUDE_SUPPORTED_FORMATS = {"JPEG": "image/jpeg", "PNG": "image/png", "WEBP": "image/webp", "GIF": "image/gif"}


def _prepare_image_for_vision(path: Path) -> tuple[str, str]:
    """Returns (media_type, base64_data). Re-encodes to PNG if the source
    format isn't one Claude's vision input accepts (e.g. AVIF)."""
    with Image.open(path) as im:
        fmt = im.format
        if fmt in _CLAUDE_SUPPORTED_FORMATS:
            im.load()
            raw = path.read_bytes()
            return _CLAUDE_SUPPORTED_FORMATS[fmt], base64.standard_b64encode(raw).decode("utf-8")
        buf = io.BytesIO()
        im.convert("RGB").save(buf, format="PNG")
        return "image/png", base64.standard_b64encode(buf.getvalue()).decode("utf-8")

IMAGE_CACHE_FILE = CFG.cache_dir / "image_extractions.json"
VOICE_CACHE_FILE = CFG.cache_dir / "voice_extractions.json"

_IMAGE_PROMPT = (
    "This image was sent as a WhatsApp message. Extract two things only:\n"
    "1. OCR: any visible text in the image, verbatim, as plain text.\n"
    "2. DESCRIPTION: one short sentence on what the image depicts "
    "(e.g. 'sale poster for a clothing store', 'screenshot of a payment app error', "
    "'QR code with a payment amount').\n\n"
    "Do not judge whether the message is safe or interpret intent -- just describe "
    "what is visibly present. Respond in exactly this format:\n"
    "OCR: <text or 'none visible'>\n"
    "DESCRIPTION: <one sentence>"
)


def _load_cache(path: Path) -> dict:
    if path.exists():
        return json.loads(path.read_text(encoding="utf-8"))
    return {}


def _save_cache(path: Path, cache: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(cache, indent=2, ensure_ascii=False), encoding="utf-8")


class MediaExtractor:
    def __init__(self, dataset: Dataset, client, dry_run: bool = False) -> None:
        self.dataset = dataset
        self.client = client  # anthropic.Anthropic instance, or None for dry-run
        self.dry_run = dry_run or client is None
        self._image_cache = _load_cache(IMAGE_CACHE_FILE)
        self._voice_cache = _load_cache(VOICE_CACHE_FILE)
        self._whisper_model = None  # lazy-loaded, only if a voice note is actually hit

    # ---- images ---------------------------------------------------------

    def extract_image(self, media_id: str) -> str:
        if not media_id:
            return ""
        if media_id in self._image_cache:
            return self._image_cache[media_id]

        path = self.dataset.media_path("image", media_id)
        if path is None or not path.exists():
            result = "[image: file not found]"
        elif self.client is None:
            # Dry-run placeholder: never persisted, so it can't poison a later real run's cache.
            return "[image: extraction skipped, dry-run mode]"
        else:
            result = self._call_vision(path)

        self._image_cache[media_id] = result
        _save_cache(IMAGE_CACHE_FILE, self._image_cache)
        return result

    def _call_vision(self, path: Path) -> str:
        media_type, data = _prepare_image_for_vision(path)
        response = self.client.messages.create(
            model=CFG.model,
            max_tokens=300,
            thinking={"type": "disabled"},
            messages=[
                {
                    "role": "user",
                    "content": [
                        {"type": "image", "source": {"type": "base64", "media_type": media_type, "data": data}},
                        {"type": "text", "text": _IMAGE_PROMPT},
                    ],
                }
            ],
        )
        text_blocks = [b.text for b in response.content if getattr(b, "type", None) == "text"]
        return "\n".join(text_blocks).strip()

    # ---- voice notes ------------------------------------------------------

    def extract_voice(self, media_id: str) -> str:
        if not media_id:
            return ""
        if media_id in self._voice_cache:
            return self._voice_cache[media_id]

        path = self.dataset.media_path("voice", media_id)
        if path is None or not path.exists():
            result = "[voice note: file not found]"
        elif self.dry_run:
            # Dry-run placeholder: never persisted, so it can't poison a later real run's cache.
            return "[voice note: extraction skipped, dry-run mode]"
        else:
            result = self._transcribe(path)

        self._voice_cache[media_id] = result
        _save_cache(VOICE_CACHE_FILE, self._voice_cache)
        return result

    def _transcribe(self, path: Path) -> str:
        if self._whisper_model is None:
            from faster_whisper import WhisperModel  # lazy import: slow, only pay for it if needed

            self._whisper_model = WhisperModel(CFG.whisper_model_size, device="cpu", compute_type="int8")

        segments, _info = self._whisper_model.transcribe(str(path), language=None)
        transcript = " ".join(seg.text.strip() for seg in segments).strip()
        return transcript or "[voice note: no speech detected]"


_THIN_MARKERS = (
    "file not found",
    "no speech detected",
    "extraction skipped",
    "none visible",
)


def is_extraction_thin(extracted_text: str | None, min_words: int = 3) -> bool:
    """True if OCR/ASR came back essentially empty -- a placeholder, a couple
    of words, or nothing usable. Used alongside is_text_sparse: a message is
    only genuinely under-informed (and should get a capped confidence) when
    BOTH the sender's own text AND the media extraction are thin -- if either
    one is substantive, there's enough to reason from."""
    if not extracted_text:
        return True
    lowered = extracted_text.lower()
    if any(marker in lowered for marker in _THIN_MARKERS):
        return True
    return len(extracted_text.split()) < min_words


def extracted_text_for_message(extractor: MediaExtractor, message: dict) -> Optional[str]:
    """Returns a normalized text block for a media message, or None for text messages.

    The returned text is deliberately shaped like message_text so downstream
    stages (safety pre-filter, retrieval, LLM reasoning) never need to branch
    on media_type -- an injection attempt hidden inside a poster image is
    caught by the same rules that catch one typed directly.
    """
    media_type = message.get("media_type", "")
    media_id = message.get("media_id", "")
    if media_type == "image":
        return extractor.extract_image(media_id)
    if media_type == "voice":
        return extractor.extract_voice(media_id)
    return None
