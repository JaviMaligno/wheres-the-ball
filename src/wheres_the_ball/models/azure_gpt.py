"""Azure OpenAI (GPT, multimodal) adapter for ball localization.

Reads AZURE_API_BASE / AZURE_API_KEY / AZURE_API_VERSION (exported by
../CooperBench/azure_env.sh). The deployment name is the model id, e.g. "gpt-5.4".
"""
from __future__ import annotations

import base64
import mimetypes
import os

from openai import AzureOpenAI


def _client() -> AzureOpenAI:
    base = os.environ.get("AZURE_API_BASE")
    key = os.environ.get("AZURE_API_KEY")
    version = os.environ.get("AZURE_API_VERSION")
    missing = [k for k, v in {
        "AZURE_API_BASE": base, "AZURE_API_KEY": key, "AZURE_API_VERSION": version
    }.items() if not v]
    if missing:
        raise RuntimeError(
            f"Missing Azure env vars: {missing}. Run `source ../CooperBench/azure_env.sh`."
        )
    return AzureOpenAI(azure_endpoint=base, api_key=key, api_version=version)


def _data_url(image_path: str) -> str:
    mime = mimetypes.guess_type(image_path)[0] or "image/jpeg"
    with open(image_path, "rb") as fh:
        b64 = base64.b64encode(fh.read()).decode()
    return f"data:{mime};base64,{b64}"


def _run(client, deployment, content) -> str:
    kwargs = {"model": deployment, "messages": [{"role": "user", "content": content}]}
    try:
        resp = client.chat.completions.create(
            response_format={"type": "json_object"}, **kwargs
        )
    except Exception:
        resp = client.chat.completions.create(**kwargs)
    return resp.choices[0].message.content or ""


def localize(image_path: str, prompt: str, deployment: str = "gpt-5.4") -> str:
    """Send one masked image + prompt, return the model's raw text answer."""
    client = _client()
    content = [
        {"type": "text", "text": prompt},
        {"type": "image_url", "image_url": {"url": _data_url(image_path)}},
    ]
    return _run(client, deployment, content)


def localize_sequence(image_paths: list[str], prompt: str, deployment: str = "gpt-5.4") -> str:
    """Send an ordered sequence of masked frames (oldest→newest) + prompt."""
    client = _client()
    content = [{"type": "text", "text": prompt}]
    for i, p in enumerate(image_paths):
        tag = "LAST frame (most recent)" if i == len(image_paths) - 1 else f"Frame {i+1}"
        content.append({"type": "text", "text": tag + ":"})
        content.append({"type": "image_url", "image_url": {"url": _data_url(p)}})
    return _run(client, deployment, content)
