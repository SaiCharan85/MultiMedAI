"""Vision-Language Model (moondream2) — detailed image narration + Q&A.

Unlike the text LLM (Qwen, which never sees the image), a VLM ingests the
PIXELS and produces a detailed description or answers a question about the image
— the "how a normal multimodal LLM answers" capability.

moondream2 (~1.8B, Apache-2.0) runs on CPU: ~100s first load, ~60s/answer.
HONEST: it is GENERAL-DOMAIN, not medical-validated — it can be imprecise or
wrong on scans. Always shown as educational, paired with the trained finding
classifier + retrieved real cases, never as a diagnosis.
"""
from __future__ import annotations

import functools

_REV = "2024-08-26"   # pinned revision with the encode_image/answer_question API


@functools.lru_cache(maxsize=1)
def _load():
    from transformers import AutoModelForCausalLM, AutoTokenizer
    model = AutoModelForCausalLM.from_pretrained(
        "vikhyatk/moondream2", revision=_REV, trust_remote_code=True).eval()
    tok = AutoTokenizer.from_pretrained("vikhyatk/moondream2", revision=_REV)
    return model, tok


DEFAULT_Q = (
    "You are describing a medical image for clinicians. Give a technical "
    "radiological/pathological description, not a lay one. State: (1) the imaging "
    "modality and plane/view; (2) the anatomy or tissue shown; (3) any lesion or "
    "abnormality characterized by location, size, shape, margins, and "
    "signal/density/echogenicity; (4) a brief impression. Use precise medical "
    "terminology and be specific about location and morphology."
)


def describe(pil_image, question: str | None = None) -> str:
    model, tok = _load()
    enc = model.encode_image(pil_image.convert("RGB"))
    q = (question or "").strip()
    if not q:
        q = DEFAULT_Q
    else:
        # steer a user's free-form question toward a technical clinical answer
        q = (q + " Answer technically, using precise medical/radiological "
             "terminology and specifying location and morphology.")
    return model.answer_question(enc, q, tok).strip()
