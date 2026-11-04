"""Voxtral Realtime through a vLLM server's /v1/realtime websocket.

vLLM batches every open session together (continuous batching) and replays each
decoder step as a CUDA graph, which removes the per-step Python overhead that
keeps the transformers backend near real time on a T4. Each clip is one
websocket session; `concurrency` sessions run at once.

Start the server first, in its own environment (see README, "Voxtral with vLLM").
"""
from __future__ import annotations

import asyncio
import base64
import json

import numpy as np

from .base import ASRBackend

CHUNK_BYTES = 32000  # 1 s of 16 kHz PCM16 per append message


class VoxtralVllmBackend(ASRBackend):
    def __init__(self, cfg: dict):
        super().__init__(cfg)
        self.url = cfg.get("url", "ws://localhost:8000/v1/realtime")
        self.model = cfg["repo"]
        self.concurrency = int(cfg.get("concurrency", 12))
        self.timeout = float(cfg.get("timeout", 120))
        try:
            asyncio.run(self._ping())
        except OSError as exc:
            raise SystemExit(f"No vLLM server at {self.url} ({exc}). Start it first: see README, "
                             "'Voxtral with vLLM'.") from exc

    async def _ping(self) -> None:
        import websockets

        async with websockets.connect(self.url, open_timeout=10):
            pass

    async def _one(self, audio: np.ndarray, sem: asyncio.Semaphore) -> str:
        import websockets

        pcm = (np.clip(audio, -1.0, 1.0) * 32767).astype(np.int16).tobytes()
        async with sem, websockets.connect(self.url, max_size=None) as ws:
            created = json.loads(await ws.recv())
            if created.get("type") != "session.created":
                raise RuntimeError(f"unexpected first event: {created}")
            await ws.send(json.dumps({"type": "session.update", "model": self.model}))
            await ws.send(json.dumps({"type": "input_audio_buffer.commit"}))
            for start in range(0, len(pcm), CHUNK_BYTES):
                chunk = base64.b64encode(pcm[start:start + CHUNK_BYTES]).decode()
                await ws.send(json.dumps({"type": "input_audio_buffer.append", "audio": chunk}))
            await ws.send(json.dumps({"type": "input_audio_buffer.commit", "final": True}))
            while True:
                event = json.loads(await asyncio.wait_for(ws.recv(), self.timeout))
                if event["type"] == "transcription.done":
                    return event["text"]
                if event["type"] == "error":
                    raise RuntimeError(f"vLLM error: {event.get('error')}")

    async def _all(self, audios: list[np.ndarray]) -> list[str]:
        sem = asyncio.Semaphore(self.concurrency)
        results = await asyncio.gather(*(self._one(a, sem) for a in audios), return_exceptions=True)
        # One retry for clips whose session failed (vLLM has known multi-session hiccups).
        for i, res in enumerate(results):
            if isinstance(res, BaseException):
                results[i] = await self._one(audios[i], sem)
        return list(results)

    def transcribe(self, audios: list[np.ndarray]) -> list[str]:
        return asyncio.run(self._all(audios))
