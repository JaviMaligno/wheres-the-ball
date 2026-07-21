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


def localize(image_path: str, prompt: str, deployment: str = "gpt-5.4") -> str:
    """Send one masked image + prompt, return the model's raw text answer."""
    client = _client()
    messages = [
        {
            "role": "user",
            "content": [
                {"type": "text", "text": prompt},
                {"type": "image_url", "image_url": {"url": _data_url(image_path)}},
            ],
        }
    ]
    kwargs = {"model": deployment, "messages": messages}
    try:
        resp = client.chat.completions.create(
            response_format={"type": "json_object"}, **kwargs
        )
    except Exception:
        # Some deployments reject response_format; retry without it.
        resp = client.chat.completions.create(**kwargs)
    return resp.choices[0].message.content or ""
