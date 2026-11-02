from __future__ import annotations

from .base import ASRBackend

BACKENDS = {
    "cohere": "darija_asr.asr.cohere:CohereBackend",
    "voxtral": "darija_asr.asr.voxtral:VoxtralBackend",
    "moulsot": "darija_asr.asr.moulsot:MoulSotBackend",
}


def load_backend(name: str, model_cfg: dict) -> ASRBackend:
    """Import lazily so each Colab session only needs its own model's deps."""
    import importlib

    if name not in BACKENDS:
        raise SystemExit(f"Unknown ASR model {name!r}; known: {', '.join(BACKENDS)}")
    module, cls = BACKENDS[name].split(":")
    return getattr(importlib.import_module(module), cls)(model_cfg)
