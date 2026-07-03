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


def is_medical_advice_request(message: str) -> bool:
    """Detect personal-diagnosis / treatment-advice requests we must NOT answer
    as if authoritative (safety boundary). Educational queries are NOT flagged."""
    m = (message or "").lower()
    patterns = ("diagnose me", "diagnose my", "do i have", "am i sick",
                "should i take", "what should i take", "what medication should i",
                "treat my", "cure my", "what's wrong with me", "whats wrong with me",
                "is my", "prescribe", "dosage should i", "how much should i take")
    return any(p in m for p in patterns)


def is_report_request(message: str) -> bool:
    """User wants a structured report/summary generated (from an active document)."""
    m = (message or "").lower()
    return any(p in m for p in ("generate a report", "generate report", "make a report",
               "write a report", "summarize the", "summary of the", "summarise the",
               "give me a report", "report on this", "overview of the document",
               "structured report", "full summary"))


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

    # 2) an ACTIVE image (uploaded OR grabbed from results) + text -> analyze it
    report_words = ("describe", "caption", "report", "findings", "summary")
    if has_image:
        if any(w in m for w in report_words):
            return "report"
        return "vqa"          # "explain this", "is it a tumor?", "what diagnosis?"

    # 3) KEYWORD-GATED images: only show a gallery when the user explicitly asks
    #    to SEE images. Otherwise we answer with TEXT (no image dump).
    show_kw = ("show", "display", "images", "image of", "picture", "photos",
               "photo of", "scans", "scan of", "examples", "gallery", "see image",
               "find image", "give me image", "give me picture", "give me scan")
    if any(k in m for k in show_kw):
        return "retrieve"

    # 4) DEFAULT = TEXT answer only (no images unless keywords above).
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
