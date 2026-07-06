"""Content safety — restrict explicit (genital/pubic) and pediatric images.

Two layers, because many bank captions are generic ("what does this image show?"):
  1) caption keyword filter (fast, free)
  2) an NSFW image classifier (Falconsai/nsfw_image_detection) run on the small
     set of retrieved candidates to catch visually-explicit images with no
     revealing caption.

Safe mode is ON by default; a professional/authorized mode can disable it.
"""
from __future__ import annotations

import functools

import re

# Restricted caption cues. WORD-BOUNDARY matched so "kid" won't hit "kidney",
# "anal" won't hit "analysis/canal", "breast" won't hit "breastbone", etc.
# Regex fragments (already allow common suffixes where safe).
SENSITIVE = [
    # genital / pubic / intimate
    r"genital\w*", r"penis", r"penile", r"vulva\w*", r"vagina\w*", r"scrotu\w*",
    r"scrotal", r"perine\w*", r"pubic", r"pubis", r"groin", r"buttock\w*",
    r"glute\w*", r"anus", r"anal", r"perianal", r"labia\w*", r"clitor\w*",
    r"foreskin", r"testic\w*", r"phallu\w*", r"inguinal",
    # breasts (per request — mammography still requestable via access).
    # "breasts?" (not breast\\w*) so "breastbone"/sternum is NOT flagged.
    r"breasts?", r"mammar\w*", r"nipple\w*", r"areola\w*",
    # minors / pediatric (under 18)
    r"infant\w*", r"bab(y|ies)", r"neonat\w*", r"newborn\w*", r"toddler\w*",
    r"p(a)?ediatric\w*", r"child\w*", r"fetus", r"foetus", r"fetal", r"foetal",
    r"minor", r"minors", r"adolescen\w*", r"juvenile", r"teenager\w*", r"teen",
    r"\bboy\b", r"\bgirl\b", r"\bkid\b", r"\bkids\b", r"school[- ]?age\w*",
    r"under\s?18", r"under the age",
]
_SENS_RE = re.compile(r"\b(?:" + "|".join(SENSITIVE) + r")\b", re.I)


def caption_restricted(caption: str) -> bool:
    return bool(_SENS_RE.search(caption or ""))


@functools.lru_cache(maxsize=1)
def _nsfw():
    from transformers import pipeline
    return pipeline("image-classification",
                    model="Falconsai/nsfw_image_detection", device=-1)


def nsfw_score(pil_image) -> float:
    try:
        for x in _nsfw()(pil_image.convert("RGB")):
            if str(x["label"]).lower() == "nsfw":
                return float(x["score"])
    except Exception:
        return 0.0
    return 0.0


def is_restricted(meta, pil_image=None, nsfw_thresh: float = 0.6) -> bool:
    """True if this image should be hidden in safe mode."""
    if caption_restricted(meta.get("question", "") if isinstance(meta, dict) else str(meta)):
        return True
    if pil_image is not None and nsfw_score(pil_image) >= nsfw_thresh:
        return True
    return False


def query_wants_restricted(query: str) -> bool:
    """True if the user's QUERY explicitly asks for restricted content."""
    return caption_restricted(query)


# request-based access: grant only with a clinical/professional justification
_JUSTIFY = ("doctor", "physician", "clinician", "clinical", "dermatolog", "patholog",
            "radiolog", "pediatric", "paediatric", "surgeon", "nurse", "medical",
            "research", "diagnos", "patient care", "professional", "hospital",
            "resident", "study", "education", "teaching", "urolog", "gynaec", "gynec")


def justification_ok(reason: str) -> bool:
    """Lightweight context check: grant access if the reason states a genuine
    clinical/professional/educational purpose. (Heuristic — not identity proof.)"""
    r = (reason or "").lower()
    return len(r.strip()) >= 15 and any(k in r for k in _JUSTIFY)
