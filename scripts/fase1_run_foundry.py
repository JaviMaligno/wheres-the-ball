"""Run an Azure AI Foundry catalog model (Meta/xAI/Mistral, vision) over the fase1 set.

Uses the Azure AI Model Inference route (/models/chat/completions) — the AzureOpenAI
/openai/deployments route does not serve non-OpenAI models. Writes predictions
incrementally to a per-model file so it composes with the GPT/Claude runs.

Env: WTB_FOUNDRY_KEY (api key for the AIServices resource).

Usage:
  WTB_FOUNDRY_KEY=$(az cognitiveservices account keys list -n wtb-foundry -g rg-sandbox-poc --query key1 -o tsv) \
  uv run python scripts/fase1_run_foundry.py --deployment llama4-maverick --key llama4 \
    --pred-path results/fase1/predictions_llama4.json
"""
from __future__ import annotations

import argparse
import base64
import json
import mimetypes
import os
import pathlib
import time

import requests

from wheres_the_ball.baselines.geometric import center_of_frame, centroid
from wheres_the_ball.models.base import parse_prediction
from wheres_the_ball.prompts.localize import PROMPTS

OUT = pathlib.Path("results/fase1")
ENDPOINT = "https://australiaeast.api.cognitive.microsoft.com/models/chat/completions"
API_VERSION = "2024-05-01-preview"


def data_url(p):
    mime = mimetypes.guess_type(p)[0] or "image/png"
    return f"data:{mime};base64," + base64.b64encode(pathlib.Path(p).read_bytes()).decode()


def call(deployment, prompt, image_path, key):
    body = {"model": deployment, "max_tokens": 400, "messages": [{"role": "user", "content": [
        {"type": "text", "text": prompt},
        {"type": "image_url", "image_url": {"url": data_url(image_path)}}]}]}
    delay = 5
    for attempt in range(6):
        r = requests.post(f"{ENDPOINT}?api-version={API_VERSION}",
                          headers={"Content-Type": "application/json", "api-key": key},
                          json=body, timeout=120)
        if r.status_code == 429:  # throttled — back off and retry
            wait = int(r.headers.get("Retry-After", delay))
            time.sleep(min(wait, 30)); delay = min(delay * 2, 30); continue
        r.raise_for_status()
        return r.json()["choices"][0]["message"]["content"] or ""
    r.raise_for_status()
    return ""


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--deployment", required=True)
    ap.add_argument("--key", required=True, help="prediction key in the output json")
    ap.add_argument("--pred-path", required=True)
    ap.add_argument("--prompt", default="neutral", choices=["neutral", "informed"])
    ap.add_argument("--limit", type=int, default=0)
    args = ap.parse_args()
    api_key = os.environ["WTB_FOUNDRY_KEY"]
    pred_path = pathlib.Path(args.pred_path)

    items = json.loads((OUT / "manifest.json").read_text())["items"]
    if args.limit:
        items = items[: args.limit]
    preds = json.loads(pred_path.read_text()) if pred_path.exists() else {}
    prompt = PROMPTS[args.prompt]

    for i, it in enumerate(items):
        rec = preds.get(it["id"], {})
        rec.setdefault("center", dict(zip("xy", center_of_frame())))
        cen = centroid(it["players"])
        if cen:
            rec.setdefault("centroid", {"x": round(cen[0], 4), "y": round(cen[1], 4)})
        need = args.key not in rec or (isinstance(rec.get(args.key), dict) and "error" in rec[args.key])
        if need:
            try:
                rec[args.key] = parse_prediction(call(args.deployment, prompt, it["masked_path"], api_key)).model_dump()
            except Exception as e:  # noqa: BLE001
                rec[args.key] = {"error": f"{type(e).__name__}: {e}"}
        preds[it["id"]] = rec
        pred_path.write_text(json.dumps(preds, indent=2))
        v = rec.get(args.key, {})
        x = v.get("x") if isinstance(v, dict) else None
        print(f"[{i+1}/{len(items)}] {it['id']} {args.key}={'%.2f'%x if x is not None else v.get('error','ERR')[:30]}", flush=True)

    print(f"\nWrote {pred_path} ({len(preds)} items)")


if __name__ == "__main__":
    main()
