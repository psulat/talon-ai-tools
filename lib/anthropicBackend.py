"""
Claude backend for talon-ai-tools: a minimal Anthropic Messages API client.

Local addition on the `claude-backend` branch (2026-11-20), not upstream.
Selected with `user.model_endpoint = "anthropic"`; see
personal/ai_plan.md (T1.1) for the wider plan.

Why raw HTTP instead of the official `anthropic` SDK: the SDK installs fine
into Talon's venv, but cannot be imported inside Talon 1.0. Talon hides
`frame.f_code`, and pydantic (an SDK dependency) reads it while building its
models (`AttributeError: f_code` in pydantic/_internal/_typing_extra.py).
Talon's bundled `requests` works, and we only need one endpoint.

This module deliberately imports nothing from talon-ai-tools, so
modelHelpers.py can import it without an import cycle.
"""

import json
import os
import time
from typing import Any, Optional

import requests

API_URL = "https://api.anthropic.com/v1/messages"
API_VERSION = "2026-10-21"
FALLBACK_BETA = "server-side-fallback-2026-07-01"
# Models that accept the server-side refusal fallback (`fallbacks: "default"`):
# if Claude's safety classifier declines, the API reruns the request on a
# suitable model inside the same call. Haiku 5.5 has no server-side fallback.
FALLBACK_MODEL_PREFIXES = ("claude-opus-5-5", "claude-sonnet-5-5")
CONNECT_TIMEOUT_S = 5


class AnthropicError(Exception):
    """A failed Claude call.

    `retryable` marks transient failures (network, timeout, 429, 5xx). The
    OpenAI fallback (T1.3) keys off it as well as off auth failures.
    """

    def __init__(
        self, message: str, status: Optional[int] = None, retryable: bool = False
    ):
        super().__init__(message)
        self.status = status
        self.retryable = retryable


def to_anthropic_content(items: list[dict]) -> list[dict]:
    """Convert talon-ai-tools message items (OpenAI-style) to Anthropic content blocks."""
    blocks: list[dict] = []
    for item in items:
        kind = item.get("type")
        if kind == "text":
            blocks.append({"type": "text", "text": item.get("text", "")})
        elif kind == "image_url":
            url = item["image_url"]["url"]
            if url.startswith("data:"):
                header, data = url.split(",", 1)
                media_type = header[len("data:") :].split(";")[0]
                # format_clipboard() writes "data:image/;base64,..." with no
                # subtype; Talon's clipboard images encode as PNG.
                if media_type in ("", "image/"):
                    media_type = "image/png"
                blocks.append(
                    {
                        "type": "image",
                        "source": {
                            "type": "base64",
                            "media_type": media_type,
                            "data": data,
                        },
                    }
                )
            else:
                blocks.append({"type": "image", "source": {"type": "url", "url": url}})
    return blocks


def create_message(
    *,
    model: str,
    messages: list[dict],
    system: str = "",
    max_tokens: int = 16000,
    effort: str = "",
    tools: Optional[list[dict]] = None,
    timeout_s: float = 20,
    retries: int = 1,
) -> dict:
    """POST /v1/messages and return the parsed JSON response.

    Raises AnthropicError on any failure. Retries once on network errors,
    429 and 5xx, but not on timeouts: a second wait would double an
    already long pause while the user is waiting to keep talking.
    """
    key = os.environ.get("ANTHROPIC_API_KEY")
    if not key:
        raise AnthropicError(
            "ANTHROPIC_API_KEY is not set (Windows user environment variable; "
            "restart Talon after setting it)"
        )
    headers = {
        "content-type": "application/json",
        "x-api-key": key,
        "anthropic-version": API_VERSION,
    }
    body: dict[str, Any] = {
        "model": model,
        "max_tokens": max_tokens,
        "messages": messages,
    }
    if system:
        body["system"] = system
    if effort:
        body["output_config"] = {"effort": effort}
    if tools:
        body["tools"] = tools
    if model.startswith(FALLBACK_MODEL_PREFIXES):
        headers["anthropic-beta"] = FALLBACK_BETA
        body["fallbacks"] = "default"

    payload = json.dumps(body)
    attempt = 0
    while True:
        retry_after = None
        try:
            response = requests.post(
                API_URL,
                headers=headers,
                data=payload,
                timeout=(CONNECT_TIMEOUT_S, timeout_s),
            )
        except requests.Timeout:
            raise AnthropicError(
                f"Claude did not answer within {timeout_s:g}s", retryable=True
            )
        except requests.RequestException as e:
            error = AnthropicError(f"network error reaching Claude: {e}", retryable=True)
        else:
            if response.status_code == 200:
                return response.json()
            error = AnthropicError(
                f"Claude API error {response.status_code}: {_error_message(response)}",
                status=response.status_code,
                retryable=response.status_code in (408, 409, 429)
                or response.status_code >= 500,
            )
            retry_after = response.headers.get("retry-after")
        if not error.retryable or attempt >= retries:
            raise error
        attempt += 1
        time.sleep(_retry_delay(retry_after, attempt))


def extract_text(message: dict) -> str:
    """Join the text blocks of a Messages API response (skipping thinking/fallback blocks).

    Raises AnthropicError if the whole fallback chain refused or no text came back.
    """
    stop_reason = message.get("stop_reason")
    if stop_reason == "refusal":
        details = message.get("stop_details") or {}
        category = details.get("category") or "no category given"
        raise AnthropicError(f"Claude declined this request ({category})")
    text = "".join(
        block.get("text", "")
        for block in message.get("content", [])
        if block.get("type") == "text"
    ).strip()
    if not text:
        raise AnthropicError(f"Claude returned no text (stop_reason={stop_reason})")
    return text


def _error_message(response: requests.Response) -> str:
    try:
        return response.json().get("error", {}).get("message") or response.text[:300]
    except ValueError:
        return response.text[:300]


def _retry_delay(retry_after: Optional[str], attempt: int) -> float:
    try:
        return min(float(retry_after), 5.0) if retry_after else 0.5 * 2**attempt
    except ValueError:
        return 0.5 * 2**attempt
