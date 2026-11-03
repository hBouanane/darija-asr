"""CohereLabs/cohere-transcribe-arabic-07-2026 (gated: accept the terms on the Hub first)."""
from __future__ import annotations

import numpy as np
import torch

from ..audio import SAMPLE_RATE
from .base import ASRBackend, resolve_dtype


def clean(text: str) -> str:
    """Drop the `#` markers the model emits around unclear stretches (`###أنا كنت#`)."""
    return " ".join(text.replace("#", " ").split())


class CohereBackend(ASRBackend):
    def __init__(self, cfg: dict):
        super().__init__(cfg)
        from transformers import AutoProcessor, CohereAsrForConditionalGeneration

        self.language = cfg.get("language", "ar")
        self.max_new_tokens = int(cfg.get("max_new_tokens", 256))
        self.processor = AutoProcessor.from_pretrained(cfg["repo"])
        self.model = CohereAsrForConditionalGeneration.from_pretrained(
            cfg["repo"], device_map="auto", dtype=resolve_dtype(cfg.get("dtype", "auto")))
        self.model.eval()
        self._batching_ok = self.batch_size > 1

    @torch.inference_mode()
    def _run(self, audio) -> list[str]:
        inputs = self.processor(audio, sampling_rate=SAMPLE_RATE, return_tensors="pt", language=self.language)
        chunk_index = inputs.get("audio_chunk_index")
        inputs.to(self.model.device, dtype=self.model.dtype)
        outputs = self.model.generate(**inputs, max_new_tokens=self.max_new_tokens)
        texts = self.processor.decode(outputs, skip_special_tokens=True,
                                      audio_chunk_index=chunk_index, language=self.language)
        texts = [texts] if isinstance(texts, str) else list(texts)
        return [clean(t) for t in texts]

    def transcribe(self, audios: list[np.ndarray]) -> list[str]:
        # The model card only documents single-clip calls. Try a batch, and fall
        # back to one-by-one for the rest of the run if batching misbehaves.
        if self._batching_ok and len(audios) > 1:
            try:
                texts = self._run(list(audios))
                if len(texts) == len(audios):
                    return texts
            except Exception as exc:  # noqa: BLE001 - any failure means "don't batch"
                print(f"[cohere] batched call failed ({exc!r}); switching to one clip at a time")
            else:
                print("[cohere] batched output count mismatch; switching to one clip at a time")
            self._batching_ok = False
        return [self._run(a)[0] for a in audios]
