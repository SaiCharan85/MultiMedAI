"""Chat engine routing — turns a user message (+ optional image) into a
structured response drawing on the four capabilities:

  retrieve  : text query -> top-k REAL pathology images from the bank
  vqa       : image + question -> answer (trained head, if available)
  report    : image -> short textual description/report (BLIP)
  generate  : prompt -> synthesized image (SD v1.5, slow on CPU; LoRA if present)

This is a RETRIEVAL/ROUTING assistant, not an LLM chatbot — feasible on CPU.
"""
from __future__ import annotations

from src.common import resolve


def detect_intent(message: str, has_image: bool) -> str:
    """Route a message to: generate | retrieve | vqa | report | ask.

    Key distinction the user asked for:
      - a REQUEST to see images ("show me X")      -> retrieve (return images)
      - a QUESTION ("what does this represent?")   -> vqa/ask (return TEXT)
    """
    m = (message or "").lower().strip()

    # 1) explicit image generation
    if any(w in m for w in ("generate", "synthesize", "synthesise",
                            "create an image", "make an image", "draw")):
        return "generate"

    # 2) EXPLICIT request to see ONLY images -> pure gallery
    show_triggers = ("show me", "show images", "show me images", "find images",
                     "images of", "image of", "pictures of", "picture of",
                     "see examples", "display images", "just images", "only images",
                     "give me images", "give me pictures")
    if any(t in m for t in show_triggers):
        return "retrieve"

    # 3) an uploaded image present -> answer about THAT image (text)
    report_words = ("describe", "caption", "report", "findings", "summary")
    if has_image:
        if any(w in m for w in report_words):
            return "report"
        return "vqa"

    # 4) DEFAULT = text-first, multimodal: give a TEXT answer + a few illustrative
    #    images. Covers questions ("what is X"), info requests ("tell me about X",
    #    "info on X", "explain X"), and bare concepts ("adenocarcinoma").
    return "ask"


def vqa_available(cfg) -> bool:
    return resolve(cfg["paths"]["weights"], "vqa", "vqa_head.pt").exists()


def bank_available(cfg) -> bool:
    return resolve(cfg["paths"]["weights"], "retrieval_bank.npz").exists()


def lora_path(cfg):
    p = resolve(cfg["paths"]["weights"], "lora")
    if p.exists():
        files = list(p.glob("*.safetensors"))
        return files[0] if files else None
    return None
