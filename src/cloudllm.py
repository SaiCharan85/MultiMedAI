"""Optional cloud LLM (Google Gemini free tier) for fast, context-aware responses.

If a Gemini key is present (env GOOGLE_API_KEY or .keys.json {"gemini": "..."}),
text + image analysis route through Gemini — near-instant vs the slow local CPU
models. If no key, everything falls back to the local Qwen/moondream.

Each Gemini model has its OWN free-tier quota bucket, so we keep a FALLBACK CHAIN:
when the primary model returns a rate/quota error (429 / ResourceExhausted) we
automatically try the next model. This keeps the app working after one model's
daily/'minute' quota is used up. flash-lite / 2.0-flash-lite return COMPLETE
answers (2.5-flash spends the budget on hidden "thinking" and truncates).
"""
from __future__ import annotations

import functools
import json
import os

from src.common import resolve

# Fallback order: confirmed-working + complete-answer models first, each a separate
# quota bucket. On a 429/quota error _generate() rotates to the next one. (1.5-flash
# is 404 for this key and 2.0-* were already quota-exhausted, so they're last/omitted.)
MODELS = [
    "gemini-3.1-flash-lite",      # primary (requested) — newest lite, own quota bucket
    "gemini-2.5-flash-lite",      # verified working, fast, COMPLETE answers
    "gemini-flash-lite-latest",   # newest lite alias, fresh quota bucket
    "gemini-2.5-flash",           # works (may truncate) — fallback
    "gemini-flash-latest",        # newest flash alias
    "gemini-2.0-flash-lite",      # last resort
    "gemini-2.0-flash",
]
MODEL = MODELS[0]   # primary (shown in the UI status line)


def _key():
    if os.environ.get("GOOGLE_API_KEY"):
        return os.environ["GOOGLE_API_KEY"]
    kf = resolve(".keys.json")
    if kf.is_file():
        try:
            d = json.loads(kf.read_text(encoding="utf-8"))
            return d.get("gemini") or d.get("google")
        except Exception:
            return None
    return None


def available() -> bool:
    return bool(_key())


def set_key(key: str):
    """Persist a Gemini key into .keys.json (gitignored) and reset the client."""
    kf = resolve(".keys.json")
    d = {}
    if kf.is_file():
        try:
            d = json.loads(kf.read_text(encoding="utf-8"))
        except Exception:
            d = {}
    d["gemini"] = key.strip()
    kf.write_text(json.dumps(d), encoding="utf-8")
    _client.cache_clear()


@functools.lru_cache(maxsize=8)
def _client(model: str):
    import google.generativeai as genai
    genai.configure(api_key=_key())
    return genai.GenerativeModel(model)


def _is_quota(err) -> bool:
    """True if the error is a rate-limit / quota-exhausted condition (try next model)."""
    s = str(err).lower()
    return any(k in s for k in ("429", "quota", "resourceexhausted", "resource exhausted",
                                "rate limit", "rate-limit", "exceeded", "too many requests"))


def _generate(parts, max_tokens: int):
    """Run generation across the fallback chain: on a quota/rate error move to the
    next model; on other errors also try the next, remembering the last error."""
    last = None
    tried = []
    for name in MODELS:
        try:
            m = _client(name)
            r = m.generate_content(
                parts,
                generation_config={"max_output_tokens": max(max_tokens, 700),
                                   "temperature": 0.3})
            txt = (r.text or "").strip()
            if txt:
                return txt
            last = RuntimeError(f"{name}: empty response")
        except Exception as e:  # noqa: BLE001
            last = e
            tried.append(f"{name}({'quota' if _is_quota(e) else 'err'})")
            continue
    raise RuntimeError("All Gemini models failed: " + ", ".join(tried)) from last


def text(system: str, user: str, max_tokens: int = 500) -> str:
    return _generate(f"{system}\n\n{user}", max_tokens)


def vision(pil_image, prompt: str, max_tokens: int = 500) -> str:
    return _generate([prompt, pil_image.convert("RGB")], max_tokens)


def vision_multi(images, prompt: str, max_tokens: int = 900) -> str:
    """Analyze MULTIPLE images together in one call (Gemini supports this)."""
    return _generate([prompt] + [im.convert("RGB") for im in images], max_tokens)


def check_key(key: str):
    """Validate a key with a tiny live call. Returns (ok: bool, message: str)."""
    try:
        import google.generativeai as genai
        genai.configure(api_key=key.strip())
        m = genai.GenerativeModel(MODELS[0])
        r = m.generate_content("Reply with the single word: OK",
                               generation_config={"max_output_tokens": 5})
        return True, (r.text or "").strip()
    except Exception as e:  # noqa: BLE001
        return False, str(e)
