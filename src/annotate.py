"""Region annotation — locate + circle/label a structure or lesion on an image.

Primary method: GEMINI vision bounding-box detection (accurate — the model
actually sees the image and returns coordinates for the asked object).
Fallback: Grad-CAM model-attention over BiomedCLIP (indicative only; used when
no Gemini key is set). Works on retrieved, uploaded, and generated images.

Honest: Gemini localization is strong but still NOT a validated medical detector;
the UI labels every result as indicative, never a diagnosis.
"""
from __future__ import annotations

import json
import re

import numpy as np
from PIL import Image, ImageDraw


# --------------------------------------------------------------------------- Gemini
def _gemini_boxes(pil_image, target):
    """Ask Gemini for bounding boxes of `target`. Returns list of (x0,y0,x1,y1)
    normalized 0..1, [] if none found, or None on error/unavailable."""
    from src import cloudllm
    if not cloudllm.available():
        return None
    prompt = (
        f"You are analysing a medical image. Detect the location of: {target}. "
        "Return ONLY a compact JSON array of bounding boxes, each as "
        "[ymin, xmin, ymax, xmax] with integer coordinates normalized to 0-1000 "
        "(top-left origin). Return at most 3 boxes for the most likely region(s). "
        "If it is not visible, return []. Output JSON only, no explanation."
    )
    try:
        txt = cloudllm.vision(pil_image, prompt, max_tokens=200)
    except Exception:
        return None
    m = re.search(r"\[.*\]", txt, re.S)
    if not m:
        return []
    try:
        arr = json.loads(m.group(0))
    except Exception:
        return []
    boxes = []
    for b in arr:
        if isinstance(b, (list, tuple)) and len(b) >= 4:
            ymin, xmin, ymax, xmax = [float(v) / 1000.0 for v in b[:4]]
            if xmax > xmin and ymax > ymin:
                boxes.append((xmin, ymin, xmax, ymax))
    return boxes


def _draw_boxes(base, boxes, label):
    img = base.convert("RGB")
    W, H = img.size
    draw = ImageDraw.Draw(img)
    red = (255, 60, 60)
    w = max(3, W // 160)
    for (x0, y0, x1, y1) in boxes:
        px = [x0 * W, y0 * H, x1 * W, y1 * H]
        draw.rectangle(px, outline=red, width=w)
        draw.ellipse(px, outline=(255, 190, 60), width=max(2, w - 1))
        ly = max(0, px[1] - 16)
        draw.rectangle([px[0], ly, px[0] + 7 * len(label) + 8, ly + 15], fill=red)
        draw.text((px[0] + 4, ly + 2), label, fill=(255, 255, 255))
    return img


# --------------------------------------------------------------------------- Grad-CAM (fallback)
def _gradcam_map(query, pil_image):
    import torch
    from src.biomedclip import load_biomedclip
    from src.common import load_config
    cfg = load_config()
    model, preprocess, tokenizer, device = load_biomedclip(cfg["retrieval"]["clip_model"])
    trunk = getattr(getattr(model, "visual", None), "trunk", None)
    blocks = getattr(trunk, "blocks", None)
    if blocks is None:
        return None
    acts = {}

    def hook(_m, _i, out):
        acts["a"] = out
        out.retain_grad()

    h = blocks[-1].register_forward_hook(hook)
    try:
        x = preprocess(pil_image.convert("RGB")).unsqueeze(0).to(device)
        f = model.encode_image(x); f = f / f.norm(dim=-1, keepdim=True)
        with torch.no_grad():
            t = tokenizer([query]).to(device)
            tf = model.encode_text(t); tf = tf / tf.norm(dim=-1, keepdim=True)
        (f * tf).sum().backward()
        a, g = acts["a"][0], acts["a"].grad[0]
        cam = torch.relu((a * g.mean(0)).sum(-1))
        cam = cam[1:] if cam.shape[0] % 2 == 1 else cam
        s = int(cam.shape[0] ** 0.5)
        if s * s != cam.shape[0]:
            return None
        cam = cam.reshape(s, s).detach().cpu().numpy()
    finally:
        h.remove()
    cam[0, :] = cam[-1, :] = cam[:, 0] = cam[:, -1] = 0
    if cam.max() <= cam.min():
        return None
    cam = (cam - cam.min()) / (cam.max() - cam.min())
    W, H = pil_image.size
    return np.array(Image.fromarray((cam * 255).astype("uint8")).resize((W, H))) / 255.0


def _gradcam(base, target):
    cam = _gradcam_map(target, base)
    if cam is None:
        return base, False
    W, H = base.size
    ov = np.zeros((H, W, 4), "uint8"); ov[..., 0] = 255; ov[..., 3] = (cam * 140).astype("uint8")
    out = Image.alpha_composite(base.convert("RGBA"), Image.fromarray(ov, "RGBA")).convert("RGB")
    cy, cx = np.unravel_index(int(np.argmax(cam)), cam.shape)
    r = max(W, H) // 8
    d = ImageDraw.Draw(out)
    d.ellipse([cx - r, cy - r, cx + r, cy + r], outline=(255, 60, 60), width=max(2, W // 180))
    d.text((cx - r + 4, cy - r - 14), "peak attention", fill=(255, 200, 60))
    return out, True


# --------------------------------------------------------------------------- public
def annotate(pil_image, target: str):
    """Return (annotated_image, found, method)."""
    base = pil_image.convert("RGB")
    boxes = _gemini_boxes(base, target)          # accurate path
    if boxes:
        return _draw_boxes(base, boxes, target[:22]), True, "gemini"
    if boxes == []:                              # Gemini answered: not visible
        return base, False, "gemini_none"
    img, ok = _gradcam(base, target)             # boxes is None -> no Gemini; fallback
    return img, ok, "gradcam"
