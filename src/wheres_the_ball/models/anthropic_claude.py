"""Anthropic (Claude, multimodal) adapter for ball localization.

Reads ANTHROPIC_API_KEY from the environment. Mirrors the auth handling of the
llm-language-limits client: subscription OAuth tokens (sk-ant-oat*) authenticate via
Bearer (auth_token), normal keys (sk-ant-api*) via x-api-key.
"""
from __future__ import annotations

import base64
import mimetypes
import os

import anthropic

# Claude models are vision-capable. Default to the latest Sonnet; override per call.
DEFAULT_MODEL = "claude-sonnet-4-6"


def _client() -> anthropic.Anthropic:
    key = os.environ.get("ANTHROPIC_API_KEY")
    if not key:
        raise RuntimeError(
            "ANTHROPIC_API_KEY not set. Source it, e.g.:\n"
            "  export ANTHROPIC_API_KEY=$(grep '^ANTHROPIC_API_KEY=' "
            "../llm-language-limits/.env | cut -d= -f2-)"
        )
    if key.startswith("sk-ant-oat"):
        return anthropic.Anthropic(auth_token=key, timeout=90.0, max_retries=2)
    return anthropic.Anthropic(api_key=key, timeout=90.0, max_retries=2)


def _image_block(image_path: str) -> dict:
    media_type = mimetypes.guess_type(image_path)[0] or "image/jpeg"
    with open(image_path, "rb") as fh:
        data = base64.b64encode(fh.read()).decode()
    return {
        "type": "image",
        "source": {"type": "base64", "media_type": media_type, "data": data},
    }


def localize(image_path: str, prompt: str, model: str = DEFAULT_MODEL) -> str:
    """Send one masked image + prompt, return Claude's raw text answer."""
    client = _client()
    resp = client.messages.create(
        model=model,
        max_tokens=1024,
        messages=[{
            "role": "user",
            "content": [{"type": "text", "text": prompt}, _image_block(image_path)],
        }],
    )
    return "".join(b.text for b in resp.content if b.type == "text")
