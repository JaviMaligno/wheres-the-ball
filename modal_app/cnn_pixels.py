"""4th cell of the grid: a small vision net TRAINED on masked pixels (Modal GPU).

Pipeline (all on an A10G, data in the `wtb-cnn-data` Volume):
  1. mask_frames : LaMa-inpaint the ~8.5k training frames (ball removed) via iopaint.
  2. train_eval  : fine-tune a torchvision ResNet-18 (regression head, sigmoid xy)
                   on the masked frames, then predict the 92 Level-1 eval items
                   (already LaMa-masked locally) and return their predictions.

Upload first:
  modal volume put wtb-cnn-data data/cnn_train /cnn_train
  modal volume put wtb-cnn-data results/fase1/masked /eval_masked
Run:
  uv run modal run modal_app/cnn_pixels.py --step mask     # ~30-40 min
  uv run modal run modal_app/cnn_pixels.py --step train    # ~30 min, prints + saves preds
"""
from __future__ import annotations

import json
import pathlib

import modal

app = modal.App("wtb-cnn-pixels")
vol = modal.Volume.from_name("wtb-cnn-data", create_if_missing=True)

lama_image = (
    modal.Image.debian_slim(python_version="3.11")
    .apt_install("libgl1", "libglib2.0-0")
    .pip_install("iopaint", "torch", "torchvision")
    .env({"HF_HOME": "/cache", "XDG_CACHE_HOME": "/cache"})
)
train_image = (
    modal.Image.debian_slim(python_version="3.11")
    .apt_install("libgl1", "libglib2.0-0")
    .pip_install("torch", "torchvision", "pillow", "numpy")
)


@app.function(image=lama_image, gpu="A10G", volumes={"/data": vol}, timeout=5400)
def mask_frames():
    import shutil
    import subprocess
    # Volume I/O is FUSE (slow per-file): copy to container-local disk, process
    # there at full speed, then copy results back and commit once.
    print("copying inputs to local disk…")
    shutil.copytree("/data/cnn_train/original", "/tmp/original")
    shutil.copytree("/data/cnn_train/masks", "/tmp/masks")
    total = len(list(pathlib.Path("/tmp/original").glob("*.jpg")))
    print(f"masking {total} frames on local disk…")
    subprocess.run(
        ["iopaint", "run", "--model", "lama", "--device", "cuda",
         "--image", "/tmp/original", "--mask", "/tmp/masks",
         "--output", "/tmp/masked"], check=True)
    n = len(list(pathlib.Path("/tmp/masked").glob("*")))
    print(f"copying {n} masked frames back to the volume…")
    out = pathlib.Path("/data/cnn_train/masked")
    if out.exists():
        shutil.rmtree(out)
    shutil.copytree("/tmp/masked", out)
    vol.commit()
    return n


@app.function(image=train_image, gpu="A10G", volumes={"/data": vol}, timeout=5400)
def train_eval(epochs: int = 8):
    import numpy as np
    import torch
    import torch.nn as nn
    from PIL import Image
    from torchvision import models, transforms

    labels = json.loads(pathlib.Path("/data/cnn_train/labels.json").read_text())
    frames = sorted(pathlib.Path("/data/cnn_train/masked").glob("*"))
    frames = [f for f in frames if f.stem in labels]
    print(f"{len(frames)} masked training frames")
    # split by CLIP to avoid leakage: last 6 clips -> val
    clips = sorted({f.stem.rsplit("_", 1)[0] for f in frames})
    val_clips = set(clips[-6:])
    train_f = [f for f in frames if f.stem.rsplit("_", 1)[0] not in val_clips]
    val_f = [f for f in frames if f.stem.rsplit("_", 1)[0] in val_clips]

    tf = transforms.Compose([
        transforms.Resize((288, 512)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
    ])

    class DS(torch.utils.data.Dataset):
        def __init__(self, files):
            self.files = files
        def __len__(self):
            return len(self.files)
        def __getitem__(self, i):
            f = self.files[i]
            x = tf(Image.open(f).convert("RGB"))
            y = torch.tensor(labels[f.stem], dtype=torch.float32)
            return x, y

    model = models.resnet18(weights=models.ResNet18_Weights.DEFAULT)
    model.fc = nn.Sequential(nn.Linear(512, 2), nn.Sigmoid())
    model = model.cuda()
    opt = torch.optim.AdamW(model.parameters(), lr=2e-4)
    tl = torch.utils.data.DataLoader(DS(train_f), batch_size=64, shuffle=True,
                                     num_workers=4, pin_memory=True)
    vl = torch.utils.data.DataLoader(DS(val_f), batch_size=64, num_workers=4)

    for ep in range(epochs):
        model.train(); tot = 0.0
        for x, y in tl:
            x, y = x.cuda(), y.cuda()
            loss = nn.functional.mse_loss(model(x), y)
            opt.zero_grad(); loss.backward(); opt.step()
            tot += float(loss) * len(x)
        model.eval(); errs = []
        with torch.no_grad():
            for x, y in vl:
                p = model(x.cuda()).cpu().numpy()
                errs += np.linalg.norm(p - y.numpy(), axis=1).tolist()
        print(f"epoch {ep+1}/{epochs} train_mse={tot/len(train_f):.5f} "
              f"val_med={np.median(errs):.4f}")

    # eval items
    preds = {}
    model.eval()
    for f in sorted(pathlib.Path("/data/eval_masked").glob("*")):
        x = tf(Image.open(f).convert("RGB")).unsqueeze(0).cuda()
        with torch.no_grad():
            p = model(x).cpu().numpy()[0]
        preds[f.stem] = {"x": round(float(p[0]), 4), "y": round(float(p[1]), 4)}
    print(f"predicted {len(preds)} eval items")
    return preds


@app.function(volumes={"/data": vol}, timeout=9000)
def pipeline(epochs: int = 8):
    """Cloud-side orchestrator: mask -> train -> save preds to the volume.
    Launch with `modal run --detach` so it survives the laptop sleeping."""
    n = mask_frames.remote()
    print(f"pipeline: {n} frames masked")
    preds = train_eval.remote(epochs=epochs)
    pathlib.Path("/data/cnn_preds.json").write_text(json.dumps(preds, indent=2))
    vol.commit()
    print(f"pipeline: saved {len(preds)} eval predictions to /data/cnn_preds.json")
    return len(preds)


@app.local_entrypoint()
def main(step: str = "train", epochs: int = 8):
    if step == "pipeline":
        call = pipeline.spawn(epochs=epochs)
        print(f"pipeline SPAWNED (detached from this client): {call.object_id}")
        print("results will land in the volume as /cnn_preds.json")
    elif step == "mask":
        n = mask_frames.remote()
        print(f"masked frames in volume: {n}")
    else:
        preds = train_eval.remote(epochs=epochs)
        out = pathlib.Path("results/fase1/cnn_preds.json")
        out.write_text(json.dumps(preds, indent=2))
        print(f"wrote {out} ({len(preds)} items)")
