"""Text normalization for *scoring only* (agreement and WER/CER).

Training labels are never passed through this; it only removes spelling
variation that should not count as an error when comparing two transcripts.
"""
from __future__ import annotations

import re
import unicodedata

_CHAR_MAP = str.maketrans({
    "ى": "ي", "ی": "ي",        # alef maqsura / Farsi yeh -> yeh
    "ة": "ه",                  # ta marbuta -> ha
    "ک": "ك",                  # Farsi kaf
    "گ": "ڭ",                  # both /g/ letters -> Moroccan gaf
    "ـ": "",                   # tatweel
    **{chr(0x0660 + i): str(i) for i in range(10)},   # Arabic-Indic digits
    **{chr(0x06F0 + i): str(i) for i in range(10)},   # Extended Arabic-Indic digits
})

_SPACES = re.compile(r"\s+")


def normalize(text: str | None) -> str:
    if not isinstance(text, str) or not text:  # None, or NaN from a missing label
        return ""
    # NFKD splits hamza/madda carriers (أ إ آ ؤ ئ) and Latin accents into a base
    # letter plus a combining mark; dropping all marks (Mn) also drops tashkeel.
    text = unicodedata.normalize("NFKD", text)
    text = "".join(ch for ch in text if unicodedata.category(ch) != "Mn")
    text = text.translate(_CHAR_MAP).lower()
    # Punctuation and symbols (incl. ، ؛ ؟) become spaces.
    text = "".join(" " if unicodedata.category(ch)[0] in "PS" else ch for ch in text)
    return _SPACES.sub(" ", text).strip()


def latin_ratio(text: str | None) -> float:
    """Share of letters in Latin script; shows code-switching / script mismatches."""
    letters = [ch for ch in (text if isinstance(text, str) else "") if ch.isalpha()]
    if not letters:
        return 0.0
    return sum("LATIN" in unicodedata.name(ch, "") for ch in letters) / len(letters)
