"""atlasia/moulsot.v0.3 (Qwen3-ASR-1.7B fine-tuned on MoulSot 100-gt-2.5).

Only use it as a voter on clips it was not trained on (extra-gt-2.5, the test
split); on 100-gt-2.5 train it would just echo its training labels.
"""
from __future__ import annotations

import os
import tempfile

import numpy as np
import soundfile as sf

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
        self.model = Qwen3ASRModel.from_pretrained(cfg["repo"], dtype=dtype, device_map="cuda")

    def transcribe(self, audios: list[np.ndarray]) -> list[str]:
        # The documented API takes a file path, so round-trip through a temp wav.
        texts = []
        for audio in audios:
            fd, path = tempfile.mkstemp(suffix=".wav")
            os.close(fd)
            try:
                sf.write(path, audio, SAMPLE_RATE)
                result = self.model.transcribe(audio=path, language=self.language)
                if isinstance(result, list):
                    result = result[0]
                texts.append(result.text)
            finally:
                os.remove(path)
        return texts
