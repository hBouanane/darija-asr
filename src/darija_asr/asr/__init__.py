from __future__ import annotations

from .base import ASRBackend

BACKENDS = {
    "cohere": "darija_asr.asr.cohere:CohereBackend",
    "voxtral": "darija_asr.asr.voxtral:VoxtralBackend",
    "moulsot": "darija_asr.asr.moulsot:MoulSotBackend",
    "voxtral:vllm": "darija_asr.asr.voxtral_vllm:VoxtralVllmBackend",
    "moulsot:vllm": "darija_asr.asr.vllm_transcribe:VllmTranscribeBackend",
    "cohere:vllm": "darija_asr.asr.vllm_transcribe:VllmTranscribeBackend",
}


def load_backend(name: str, model_cfg: dict) -> ASRBackend:
    """Import lazily so each Colab session only needs its own model's deps."""
    import importlib

    # `backend:` in a model's config picks another runtime for the same model,
    # e.g. voxtral with backend: vllm. Its transcripts still go to asr/<name>/.
    key = name if model_cfg.get("backend", "transformers") == "transformers" else f"{name}:{model_cfg['backend']}"
    if key not in BACKENDS:
        raise SystemExit(f"Unknown ASR model/backend {key!r}; known: {', '.join(BACKENDS)}")
    module, cls = BACKENDS[key].rsplit(":", 1)
    return getattr(importlib.import_module(module), cls)(model_cfg)
