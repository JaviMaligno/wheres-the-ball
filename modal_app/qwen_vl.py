"""Qwen2.5-VL-7B on Modal GPU as an open-VLM reference for the ball task.

Mirrors the llm-language-limits Modal pattern (GPU cls + @modal.enter). The 7B-VL model
fits an A10G (24GB); weights cache in a Volume so only the first run downloads (~16GB).

Deploy-free run (reads the local manifest, writes results/fase1/qwen_preds.json):
  cd wheres-the-ball
  uv run modal run modal_app/qwen_vl.py                      # neutral prompt
  uv run modal run modal_app/qwen_vl.py --prompt-variant informed
"""
from __future__ import annotations

import base64
import io
import json
import pathlib

import modal

MODEL = "Qwen/Qwen2.5-VL-7B-Instruct"
app = modal.App("wheres-the-ball-qwenvl")
hf_cache = modal.Volume.from_name("wtb-hf-cache", create_if_missing=True)

image = (
    modal.Image.debian_slim(python_version="3.11")
    .pip_install("torch", "torchvision", "transformers>=4.49", "accelerate>=0.33",
                 "qwen-vl-utils", "pillow", "numpy")
    .env({"HF_HOME": "/cache"})
)


@app.cls(image=image, gpu="A10G", volumes={"/cache": hf_cache}, timeout=1800)
class QwenVL:
    @modal.enter()
    def load(self):
        import torch
        from transformers import AutoProcessor, Qwen2_5_VLForConditionalGeneration
        self.model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
            MODEL, torch_dtype=torch.bfloat16, device_map="auto")
        self.processor = AutoProcessor.from_pretrained(MODEL)

    @modal.method()
    def localize(self, image_b64: str, prompt: str) -> str:
        from PIL import Image
        from qwen_vl_utils import process_vision_info
        img = Image.open(io.BytesIO(base64.b64decode(image_b64))).convert("RGB")
        messages = [{"role": "user", "content": [
            {"type": "image", "image": img}, {"type": "text", "text": prompt}]}]
        text = self.processor.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True)
        image_inputs, video_inputs = process_vision_info(messages)
        inputs = self.processor(text=[text], images=image_inputs, videos=video_inputs,
                                padding=True, return_tensors="pt").to(self.model.device)
        gen = self.model.generate(**inputs, max_new_tokens=256, do_sample=False)
        trimmed = [o[len(i):] for i, o in zip(inputs.input_ids, gen)]
        return self.processor.batch_decode(trimmed, skip_special_tokens=True)[0]


@app.local_entrypoint()
def main(prompt_variant: str = "neutral"):
    from wheres_the_ball.models.base import parse_prediction
    from wheres_the_ball.prompts.localize import PROMPTS

    out = pathlib.Path("results/fase1")
    manifest = json.loads((out / "manifest.json").read_text())
    prompt = PROMPTS[prompt_variant]
    qwen = QwenVL()

    preds = {}
    items = manifest["items"]
    for i, it in enumerate(items):
        b64 = base64.b64encode(pathlib.Path(it["masked_path"]).read_bytes()).decode()
        try:
            raw = qwen.localize.remote(b64, prompt)
            preds[it["id"]] = parse_prediction(raw).model_dump()
            g = preds[it["id"]]
            print(f"[{i+1}/{len(items)}] {it['id']}: ({g['x']:.2f},{g['y']:.2f})")
        except Exception as e:  # noqa: BLE001
            preds[it["id"]] = {"error": f"{type(e).__name__}: {e}"}
            print(f"[{i+1}/{len(items)}] {it['id']}: ERR {e}")

    fname = "qwen_preds.json" if prompt_variant == "neutral" else f"qwen_preds_{prompt_variant}.json"
    (out / fname).write_text(json.dumps(preds, indent=2))
    print(f"\nWrote {out/fname} ({len(preds)} items)")
