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
    locate_words = ("where is", "where's", "where are", "locate", "circle", "mark",
                    "highlight", "point to", "point out", "label", "which part",
                    "show the location", "annotate")
    if has_image:
        if any(w in m for w in locate_words):
            return "locate"   # circle + label a region ("where is the tumor?")
        if any(w in m for w in report_words):
            return "report"
        return "vqa"          # "explain this", "is it a tumor?", "what diagnosis?"

    # 3) RESEARCH: explicit request for papers/evidence/literature
    research_kw = ("research paper", "research papers", "find papers", "research on",
                   "studies on", "literature", "evidence for", "evidence on",
                   "papers about", "papers on", "citations", "references for",
                   "google scholar", "pubmed", "recent research", "latest studies",
                   "find research")
    if any(k in m for k in research_kw):
        return "research"

    # 3b) "labelled diagram / anatomy of X" (without 'generate') -> retrieve REAL
    #     images, since diffusion can't render real labels.
    if any(k in m for k in ("labelled diagram", "labeled diagram", "diagram of",
                            "anatomy of", "atlas of", "labelled image", "labeled image",
                            "labelled scan", "anatomy diagram")):
        return "retrieve"

    # 4) KEYWORD-GATED images: only show a gallery when the user explicitly asks
    #    to SEE images. Otherwise we answer with TEXT (no image dump).
    show_kw = ("show", "display", "images", "image of", "picture", "photos",
               "photo of", "scans", "scan of", "examples", "gallery", "see image",
               "find image", "give me image", "give me picture", "give me scan",
               # "how/what does an X-ray of Y look like" — appearance = show images
               "look like", "looks like", "look alike", "appearance", "what it looks",
               "how it looks", "what does it look", "how does it look",
               "x-ray of", "xray of", "ct of", "ct scan of", "mri of", "mri scan of",
               "radiograph of", "ultrasound of", "sonograph")
    if any(k in m for k in show_kw):
        return "retrieve"

    # 4) DEFAULT = TEXT answer only (no images unless keywords above).
    return "ask"


_TYPO = {
    "iamge": "image", "imge": "image", "imgae": "image", "reserach": "research",
    "reasearch": "research", "resarch": "research", "paers": "papers",
    "papres": "papers", "brian": "brain", "scna": "scan", "scaan": "scan",
    "pnuemonia": "pneumonia", "pnemonia": "pneumonia", "tumour": "tumour",
    "detials": "details", "detailled": "detailed", "diagnositics": "diagnostics",
    "diagnsis": "diagnosis", "radioligy": "radiology", "pathlogy": "pathology",
    "summry": "summary", "summrize": "summarize", "analyses": "analysis",
    "hemmorrhage": "hemorrhage", "hemmorhage": "hemorrhage", "carsinoma": "carcinoma",
    "tuberculsis": "tuberculosis", "fracutre": "fracture", "leison": "lesion",
    "wht": "what", "wheer": "where", "reprot": "report", "docmuent": "document",
}


def autocorrect(message: str) -> str:
    """Light, safe typo correction for common misspellings + 'form YYYY'->'from YYYY'.
    Only fixes clearly-wrong tokens; leaves valid words untouched."""
    import re

    def fix_word(m):
        w = m.group(0)
        low = w.lower()
        if low in _TYPO:
            rep = _TYPO[low]
            return rep.capitalize() if w[:1].isupper() else rep
        return w

    out = re.sub(r"[A-Za-z]+", fix_word, message)
    out = re.sub(r"\bform\s+((?:19|20)\d{2})\b", r"from \1", out, flags=re.I)
    return out


def extract_target(message: str) -> str:
    """Pull the thing to locate from a message ('where is the tumor?' -> 'tumor')."""
    m = (message or "").lower().strip(" ?.!")
    triggers = ("where is the", "where's the", "where are the", "where is",
                "point out the", "point to the", "highlight the", "circle the",
                "mark the", "label the", "locate the", "locate", "annotate the",
                "which part is the", "show the location of", "highlight", "circle",
                "mark", "label")
    for kw in triggers:
        if kw in m:
            tail = m.split(kw, 1)[1].strip()
            for art in ("the ", "a ", "an "):
                if tail.startswith(art):
                    tail = tail[len(art):]
            tail = tail.replace("in this image", "").replace("on this image", "").strip()
            if tail:
                return tail[:40]
    return "abnormality"


def extract_research_topic(message: str) -> str:
    """Pull the topic from a research request, dropping trigger phrases."""
    m = (message or "").strip()
    low = m.lower()
    for kw in ("research paper on", "research papers on", "research paper about",
               "research papers about", "research paper", "research papers",
               "find papers on", "find papers about", "find research on",
               "research on", "studies on", "papers about", "papers on",
               "literature on", "evidence for", "evidence on", "references for",
               "citations for", "recent research on", "latest studies on",
               "find papers", "google scholar", "pubmed", "literature"):
        i = low.find(kw)
        if i != -1:
            return (m[i + len(kw):].strip(" :?.") or m).strip()
    return m


def wants_scholar(message: str) -> bool:
    """Only spend SerpApi (Scholar) quota when the user explicitly asks for it."""
    return "scholar" in (message or "").lower()


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
