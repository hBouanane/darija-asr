"""Claude as the transcript judge: one request per clip, structured JSON output.

Two ways to run it:
- sync:  parallel requests, results as they come (good for gold/calibration runs);
- batch: Message Batches API, 50% cheaper, results within ~1h (good for big runs).
"""
from __future__ import annotations

import hashlib
import json

import anthropic

from .prompt import OUTPUT_SCHEMA


class Judge:
    def __init__(self, mcfg: dict, system_prompt: str):
        self.model = mcfg["model"]
        self.effort = mcfg.get("effort", "medium")
        self.max_tokens = int(mcfg.get("max_tokens", 16000))
        self.system_prompt = system_prompt
        self.client = anthropic.Anthropic(max_retries=6)

    def params(self, user_message: str) -> dict:
        return {
            "model": self.model,
            "max_tokens": self.max_tokens,
            # Same system prompt on every request -> served from cache once it is
            # past the model's minimum cacheable length (a long style guide/lexicon).
            "system": [{"type": "text", "text": self.system_prompt, "cache_control": {"type": "ephemeral"}}],
            "messages": [{"role": "user", "content": user_message}],
            "output_config": {"effort": self.effort,
                              "format": {"type": "json_schema", "schema": OUTPUT_SCHEMA}},
        }

    def call(self, user_message: str) -> dict:
        """Sync request. Returns a result row (without the clip id)."""
        try:
            message = self.client.beta.messages.create(
                **self.params(user_message),
                # On a safety decline the API retries on a fallback model in the same call.
                betas=["server-side-fallback-2026-07-01"],
                fallbacks="default",
            )
        except anthropic.BadRequestError as exc:
            return error_row(f"bad_request: {exc.message}")
        except anthropic.RateLimitError as exc:
            return error_row(f"rate_limited: {exc.message}")
        except anthropic.APIStatusError as exc:
            return error_row(f"api_{exc.status_code}: {exc.message}")
        except anthropic.APIConnectionError as exc:
            return error_row(f"connection: {exc}")
        return parse_message(message)


def parse_message(message) -> dict:
    usage = {"input_tokens": message.usage.input_tokens,
             "output_tokens": message.usage.output_tokens,
             "cache_read_tokens": getattr(message.usage, "cache_read_input_tokens", 0) or 0}
    if message.stop_reason == "refusal":
        return {**error_row("refusal"), **usage}
    if message.stop_reason == "max_tokens":
        return {**error_row("max_tokens"), **usage}
    text = next((b.text for b in message.content if b.type == "text"), None)
    try:
        data = json.loads(text)
    except (TypeError, json.JSONDecodeError):
        return {**error_row(f"unparseable: {str(text)[:200]}"), **usage}
    return {"text": data["text"], "reject": bool(data["reject"]), "confidence": data["confidence"],
            "notes": data["notes"], "error": None, "model": message.model, **usage}


def error_row(error: str) -> dict:
    return {"text": None, "reject": None, "confidence": None, "notes": None, "error": error[:500]}


def custom_id(clip_id: str) -> str:
    """Batch custom_ids must match ^[a-zA-Z0-9_-]{1,64}$; clip ids contain '@'."""
    return "c" + hashlib.sha1(clip_id.encode()).hexdigest()[:32]
