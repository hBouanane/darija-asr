"""mistralai/Voxtral-Mini-4B-Realtime-Arabic, offline file transcription via transformers."""
from __future__ import annotations

import numpy as np
import torch

from .base import ASRBackend, resolve_dtype


class VoxtralBackend(ASRBackend):
    def __init__(self, cfg: dict):
        super().__init__(cfg)
        from transformers import AutoProcessor, VoxtralRealtimeForConditionalGeneration

        self.generate_kwargs = {"do_sample": False, **(cfg.get("generate_kwargs") or {})}
        self.processor = AutoProcessor.from_pretrained(cfg["repo"])
        self.model = VoxtralRealtimeForConditionalGeneration.from_pretrained(
            cfg["repo"], dtype=resolve_dtype(cfg.get("dtype", "auto")), device_map="auto")
        self.model.eval()

    @torch.inference_mode()
    def transcribe(self, audios: list[np.ndarray]) -> list[str]:
        texts = []
        for audio in audios:
            inputs = self.processor(audio, return_tensors="pt").to(self.model.device, dtype=self.model.dtype)
            outputs = self.model.generate(**inputs, **self.generate_kwargs)
            texts.append(self.processor.batch_decode(outputs, skip_special_tokens=True)[0])
        return texts
