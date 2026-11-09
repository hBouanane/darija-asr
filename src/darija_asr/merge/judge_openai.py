"""The judge on any OpenAI-compatible chat API: Gemini (Google's OpenAI endpoint),
or an open model such as Gemma served by `vllm serve` on a rented GPU.

Same interface as merge.judge.Judge: call(user_message) -> result row. One clip
per request; run many at once (merge.workers) and let the server batch them.
With vLLM the system prompt is identical on every request, so its automatic
prefix caching computes it only once.
"""
from __future__ import annotations

import json
import os
import re
import time

import httpx

from .judge import error_row
from .prompt import OUTPUT_SCHEMA

FENCE = re.compile(r"^```(?:json)?\s*|\s*```$")


class OpenAIJudge:
    def __init__(self, mcfg: dict, system_prompt: str):
        self.model = mcfg["model"]
        self.base_url = mcfg["base_url"].rstrip("/")
        key_env = mcfg.get("api_key_env")
        self.api_key = os.environ.get(key_env, "") if key_env else "EMPTY"
        if key_env and not self.api_key:
            raise SystemExit(f"Set {key_env} (merge.api_key_env) for {self.base_url}")
        self.max_tokens = int(mcfg.get("max_tokens", 2048))
        self.temperature = float(mcfg.get("temperature", 0.0))
        self.extra_body = mcfg.get("extra_body") or {}
        self.system_prompt = system_prompt
        self.client = httpx.Client(timeout=float(mcfg.get("timeout", 300)),
                                   headers={"Authorization": f"Bearer {self.api_key}"})

    def payload(self, user_message: str) -> dict:
        return {
            "model": self.model,
            "messages": [{"role": "system", "content": self.system_prompt},
                         {"role": "user", "content": user_message}],
            "max_tokens": self.max_tokens,
            "temperature": self.temperature,
            "response_format": {"type": "json_schema",
                                "json_schema": {"name": "verdict", "schema": OUTPUT_SCHEMA, "strict": True}},
            **self.extra_body,
        }

    def call(self, user_message: str, retries: int = 6) -> dict:
        for attempt in range(retries):
            try:
                resp = self.client.post(f"{self.base_url}/chat/completions", json=self.payload(user_message))
            except httpx.HTTPError as exc:
                err = f"connection: {exc!r}"
            else:
                if resp.status_code == 200:
                    return parse_response(resp.json())
                err = f"api_{resp.status_code}: {resp.text[:300]}"
                if resp.status_code not in (408, 409, 429, 500, 502, 503, 504):
                    return error_row(err)
            time.sleep(min(60, 2 ** attempt))
        return error_row(err)


def parse_response(body: dict) -> dict:
    usage = body.get("usage") or {}
    tokens = {"input_tokens": usage.get("prompt_tokens", 0), "output_tokens": usage.get("completion_tokens", 0),
              "cache_read_tokens": (usage.get("prompt_tokens_details") or {}).get("cached_tokens", 0) or 0}
    choice = (body.get("choices") or [{}])[0]
    if choice.get("finish_reason") == "length":
        return {**error_row("max_tokens"), **tokens}
    text = FENCE.sub("", (choice.get("message") or {}).get("content") or "").strip()
    try:
        data = json.loads(text)
        row = {"text": data["text"], "reject": bool(data["reject"]), "confidence": data["confidence"],
               "notes": data.get("notes", "")}
    except (json.JSONDecodeError, KeyError, TypeError):
        return {**error_row(f"unparseable: {text[:200]}"), **tokens}
    return {**row, "error": None, "model": body.get("model"), **tokens}
