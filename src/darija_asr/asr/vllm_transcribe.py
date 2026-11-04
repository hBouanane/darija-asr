"""Any model a vLLM server exposes on the OpenAI-style /v1/audio/transcriptions route.

Used for MoulSot (Qwen3-ASR) and Cohere with `backend: vllm`. Each clip is one
HTTP request; `concurrency` requests are in flight and vLLM batches them on the
GPU. Start the server first, in its own environment (see README, "vLLM").
"""
from __future__ import annotations

import asyncio
import io

import numpy as np
import soundfile as sf

from ..audio import SAMPLE_RATE
from .base import ASRBackend

ASR_TEXT_TAG = "<asr_text>"  # Qwen3-ASR prefixes its output with "language X<asr_text>"


class VllmTranscribeBackend(ASRBackend):
    def __init__(self, cfg: dict):
        super().__init__(cfg)
        import httpx

        self.base_url = cfg.get("url", "http://localhost:8000/v1").rstrip("/")
        self.model = cfg.get("served_name", cfg["repo"])
        self.language = cfg.get("iso_language", "ar")
        self.concurrency = int(cfg.get("concurrency", 64))
        self.timeout = float(cfg.get("timeout", 120))
        try:
            httpx.get(f"{self.base_url}/models", timeout=10).raise_for_status()
        except httpx.HTTPError as exc:
            raise SystemExit(f"No vLLM server at {self.base_url} ({exc}). Start it first: see README, 'vLLM'.") from exc

    async def _one(self, client, audio: np.ndarray, sem: asyncio.Semaphore) -> str:
        buf = io.BytesIO()
        sf.write(buf, audio, SAMPLE_RATE, format="WAV", subtype="PCM_16")
        async with sem:
            resp = await client.post(
                f"{self.base_url}/audio/transcriptions",
                files={"file": ("clip.wav", buf.getvalue(), "audio/wav")},
                data={"model": self.model, "language": self.language, "temperature": "0"},
            )
        resp.raise_for_status()
        text = resp.json()["text"]
        return text.split(ASR_TEXT_TAG, 1)[1] if ASR_TEXT_TAG in text else text

    async def _all(self, audios: list[np.ndarray]) -> list[str]:
        import httpx

        sem = asyncio.Semaphore(self.concurrency)
        async with httpx.AsyncClient(timeout=self.timeout) as client:
            results = await asyncio.gather(*(self._one(client, a, sem) for a in audios), return_exceptions=True)
            for i, res in enumerate(results):  # one retry per failed clip
                if isinstance(res, BaseException):
                    results[i] = await self._one(client, audios[i], sem)
        return [r.strip() for r in results]

    def transcribe(self, audios: list[np.ndarray]) -> list[str]:
        return asyncio.run(self._all(audios))
