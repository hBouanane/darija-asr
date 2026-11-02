from __future__ import annotations

import jiwer

from ..normalize import normalize


def score(refs: list[str], hyps: list[str], normalized: bool = True) -> dict:
    """Corpus WER/CER. Missing hypotheses count as empty (all deletions)."""
    pairs = [(r, h or "") for r, h in zip(refs, hyps)]
    if normalized:
        pairs = [(normalize(r), normalize(h)) for r, h in pairs]
    pairs = [(r, h) for r, h in pairs if r.strip()]
    refs_, hyps_ = [p[0] for p in pairs], [p[1] for p in pairs]
    return {"wer": jiwer.wer(refs_, hyps_), "cer": jiwer.cer(refs_, hyps_), "n": len(pairs)}
