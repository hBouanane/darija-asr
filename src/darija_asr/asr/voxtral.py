"""mistralai/Voxtral-Mini-4B-Realtime-Arabic, offline file transcription via transformers."""
from __future__ import annotations

import warnings

import numpy as np
import torch

from .base import ASRBackend, resolve_dtype

# The generation length follows the audio length (one step per 80 ms frame), so
# transformers' "default max_length" warning is expected on every call.
warnings.filterwarnings("ignore", message="Using the model-agnostic default `max_length`")


class VoxtralBackend(ASRBackend):
    def __init__(self, cfg: dict):
        super().__init__(cfg)
        from transformers import AutoProcessor, VoxtralRealtimeForConditionalGeneration

        self.generate_kwargs = {"do_sample": False, **(cfg.get("generate_kwargs") or {})}
        self.processor = AutoProcessor.from_pretrained(cfg["repo"])
        self.model = VoxtralRealtimeForConditionalGeneration.from_pretrained(
            cfg["repo"], dtype=resolve_dtype(cfg.get("dtype", "auto")), device_map="auto")
        self.model.eval()
        self._batching_ok = self.batch_size > 1

    @torch.inference_mode()
    def _run(self, audio) -> list[str]:
        # One decoder step reads all 8.8 GB of weights whatever the batch size,
        # so batching is close to free throughput on a memory-bound T4.
        inputs = self.processor(audio, return_tensors="pt").to(self.model.device, dtype=self.model.dtype)
        outputs = self.model.generate(**inputs, **self.generate_kwargs)
        return self.processor.batch_decode(outputs, skip_special_tokens=True)

    def transcribe(self, audios: list[np.ndarray]) -> list[str]:
        if self._batching_ok and len(audios) > 1:
            try:
                texts = self._run(list(audios))
                if len(texts) == len(audios):
                    return texts
            except Exception as exc:  # noqa: BLE001 - any failure means "don't batch"
                print(f"[voxtral] batched call failed ({exc!r}); switching to one clip at a time")
            else:
                print("[voxtral] batched output count mismatch; switching to one clip at a time")
            self._batching_ok = False
        return [self._run([a])[0] for a in audios]
