"""Prompt for the LLM judge.

The system prompt (instructions + style guide + optional lexicon) is identical
for every clip so it is served from the prompt cache; only the candidates vary.
Candidates get neutral letters in a per-clip shuffled order so the judge cannot
learn a position or source bias, unless `reveal_sources` is on.
"""
from __future__ import annotations

import hashlib
import random
from pathlib import Path

INSTRUCTIONS = """\
You reconstruct the transcript of one short Moroccan Darija audio clip. You do not hear the audio. \
You receive several transcripts of the same clip, each produced independently by a different \
speech-recognition system; each can contain errors (misheard words, dropped or invented words, \
words wrongly turned into Modern Standard Arabic, French words wrongly written in Arabic script, \
or the reverse).

Produce the single most likely exact transcript of what was said:
- Compare the candidates word by word. Where they agree, keep that. Where they disagree, choose \
the reading best supported by the other candidates and by what a Darija speaker would plausibly say.
- Use only words that are supported by at least one candidate. You may change the spelling or the \
script of a supported word to follow the style guide, but never add content no candidate supports, \
never translate Darija into MSA, never fix the speaker's grammar, and never complete a cut-off sentence.
- Follow the style guide below for script, spelling, numbers and punctuation.
- If the candidates are so different that you cannot tell what was said, or none of them looks like \
real speech, set "reject" to true.

Return "text" (the transcript, or "" when rejecting), "reject", "confidence" \
("high" when the candidates mostly agree, "medium" when you resolved real disagreements, "low" \
when you had to guess), and "notes": at most one short sentence on what you resolved."""

OUTPUT_SCHEMA = {
    "type": "object",
    "properties": {
        "text": {"type": "string"},
        "reject": {"type": "boolean"},
        "confidence": {"type": "string", "enum": ["high", "medium", "low"]},
        "notes": {"type": "string"},
    },
    "required": ["text", "reject", "confidence", "notes"],
    "additionalProperties": False,
}


def build_system_prompt(style_guide_path: str | None, lexicon_path: str | None = None) -> str:
    parts = [INSTRUCTIONS]
    if style_guide_path:
        parts.append("<style_guide>\n" + Path(style_guide_path).read_text(encoding="utf-8").strip()
                     + "\n</style_guide>")
    if lexicon_path:
        # TSV: word<TAB>note. Reference spellings, e.g. from the DODa dictionary.
        lines = [ln for ln in Path(lexicon_path).read_text(encoding="utf-8").splitlines()[1:] if ln.strip()]
        parts.append("<reference_spellings>\n" + "\n".join(lines) + "\n</reference_spellings>")
    return "\n\n".join(parts)


def build_user_message(clip_id: str, candidates: dict[str, str], reveal_sources: bool = False) -> str:
    """candidates: source name -> raw transcript (missing/empty sources are skipped)."""
    present = sorted((k, v.strip()) for k, v in candidates.items() if v and v.strip())
    seed = int(hashlib.md5(clip_id.encode()).hexdigest(), 16)
    random.Random(seed).shuffle(present)
    blocks = []
    for i, (source, text) in enumerate(present):
        label = source if reveal_sources else chr(ord("A") + i)
        blocks.append(f'<candidate id="{label}">\n{text}\n</candidate>')
    return "\n".join(blocks)
