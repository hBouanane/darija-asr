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


class ServerDown(SystemExit):
    """The vLLM server stopped answering. A SystemExit, so `transcribe` saves its
    rows and stops instead of timing out clip after clip; restart the server and
    rerun (only the unfinished clips are redone)."""


class VoxtralVllmBackend(ASRBackend):
    def __init__(self, cfg: dict):
        super().__init__(cfg)
        self.url = cfg.get("url", "ws://localhost:8000/v1/realtime")
        self.model = cfg["repo"]
        self.concurrency = int(cfg.get("concurrency", 12))
        self.timeout = float(cfg.get("timeout", 120))
        self.clip_timeout = float(cfg.get("clip_timeout", 180))  # whole session, per clip
        # http://host:port/health of the same server, to tell "slow" from "frozen"
        self.health_url = self.url.replace("ws://", "http://").replace("wss://", "https://").split("/v1/")[0] + "/health"
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

    def _healthy(self) -> bool:
        import httpx

        try:
            return httpx.get(self.health_url, timeout=10).status_code == 200
        except httpx.HTTPError:
            return False

    async def _timed(self, audio: np.ndarray, sem: asyncio.Semaphore) -> str:
        return await asyncio.wait_for(self._one(audio, sem), self.clip_timeout)

    async def _all(self, audios: list[np.ndarray]) -> list[str]:
        sem = asyncio.Semaphore(self.concurrency)
        results = await asyncio.gather(*(self._timed(a, sem) for a in audios), return_exceptions=True)
        if any(isinstance(r, BaseException) for r in results) and not self._healthy():
            raise ServerDown(f"vLLM server at {self.health_url} is not answering; restart it and rerun")
        # One retry for clips whose session failed (vLLM has known multi-session hiccups).
        for i, res in enumerate(results):
            if isinstance(res, BaseException):
                results[i] = await self._timed(audios[i], sem)
        return list(results)

    def transcribe(self, audios: list[np.ndarray]) -> list[str]:
        if not self._healthy():
            raise ServerDown(f"vLLM server at {self.health_url} is not answering; restart it and rerun")
        return asyncio.run(self._all(audios))
