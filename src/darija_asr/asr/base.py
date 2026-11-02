from __future__ import annotations

import numpy as np


def resolve_dtype(name: str):
    import torch

    if name == "auto":
        if torch.cuda.is_available() and torch.cuda.is_bf16_supported():
            return torch.bfloat16
        # T4 has no native bf16; fp16 halves memory but can overflow on some
        # models, so check the smoke-test output before a long run.
        return torch.float16 if torch.cuda.is_available() else torch.float32
    return {"bf16": torch.bfloat16, "bfloat16": torch.bfloat16, "fp16": torch.float16,
            "float16": torch.float16, "fp32": torch.float32, "float32": torch.float32}[name]


class ASRBackend:
    """One model behind a common interface: 16 kHz float32 mono arrays -> texts."""

    def __init__(self, cfg: dict):
        self.cfg = cfg
        self.batch_size = int(cfg.get("batch_size", 1))

    def transcribe(self, audios: list[np.ndarray]) -> list[str]:
        raise NotImplementedError
