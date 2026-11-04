"""atlasia/moulsot.v0.3 (Qwen3-ASR-1.7B fine-tuned on MoulSot 100-gt-2.5).

Only use it as a voter on clips it was not trained on (extra-gt-2.5, the test
split); on 100-gt-2.5 train it would just echo its training labels.
"""
from __future__ import annotations

import numpy as np

from ..audio import SAMPLE_RATE
from .base import ASRBackend


class MoulSotBackend(ASRBackend):
    def __init__(self, cfg: dict):
        super().__init__(cfg)
        import torch
        from qwen_asr import Qwen3ASRModel

        dtype = cfg.get("dtype", "auto")
        if dtype == "auto":
            dtype = "bfloat16" if torch.cuda.is_bf16_supported() else "float16"
        self.language = cfg.get("language", "Arabic")
        self.model = Qwen3ASRModel.from_pretrained(
            cfg["repo"], dtype=getattr(torch, dtype), device_map="cuda",
            max_inference_batch_size=self.batch_size, max_new_tokens=int(cfg.get("max_new_tokens", 256)))

    def transcribe(self, audios: list[np.ndarray]) -> list[str]:
        # qwen_asr batches a list of (array, sample_rate) tuples itself.
        results = self.model.transcribe(audio=[(a, SAMPLE_RATE) for a in audios],
                                        language=[self.language] * len(audios))
        if not isinstance(results, list):
            results = [results]
        return [r.text for r in results]
