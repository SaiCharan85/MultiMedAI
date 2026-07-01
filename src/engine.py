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
    m = (message or "").lower()
    gen_words = ("generate", "synthesize", "synthesise", "create an image",
                 "make an image", "draw")
    if any(w in m for w in gen_words):
        return "generate"
    if has_image:
        report_words = ("describe", "caption", "report", "findings", "summary")
        if any(w in m for w in report_words):
            return "report"
        return "vqa"          # default for an uploaded image
    return "retrieve"          # text-only -> find matching images


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
