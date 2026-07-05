"""Optional cloud LLM (Google Gemini free tier) for fast, context-aware responses.

If a Gemini key is present (env GOOGLE_API_KEY or .keys.json {"gemini": "..."}),
text + image analysis route through Gemini 1.5 Flash — near-instant vs the slow
local CPU models. If no key, everything falls back to the local Qwen/moondream.
Claude/ChatGPT have no free API tier, so Gemini is the free choice.
"""
from __future__ import annotations

import functools
import json
import os

from src.common import resolve

MODEL = "gemini-2.5-flash"   # 2.0-flash quota exhausted; 2.5-flash has free quota


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


@functools.lru_cache(maxsize=1)
def _client():
    import google.generativeai as genai
    genai.configure(api_key=_key())
    return genai.GenerativeModel(MODEL)


def text(system: str, user: str, max_tokens: int = 500) -> str:
    m = _client()
    r = m.generate_content(
        f"{system}\n\n{user}",
        generation_config={"max_output_tokens": max_tokens, "temperature": 0.3})
    return (r.text or "").strip()


def vision(pil_image, prompt: str, max_tokens: int = 500) -> str:
    m = _client()
    r = m.generate_content(
        [prompt, pil_image.convert("RGB")],
        generation_config={"max_output_tokens": max_tokens, "temperature": 0.3})
    return (r.text or "").strip()


def vision_multi(images, prompt: str, max_tokens: int = 700) -> str:
    """Analyze MULTIPLE images together in one call (Gemini supports this)."""
    m = _client()
    parts = [prompt] + [im.convert("RGB") for im in images]
    r = m.generate_content(
        parts, generation_config={"max_output_tokens": max_tokens, "temperature": 0.3})
    return (r.text or "").strip()


def check_key(key: str):
    """Validate a key with a tiny live call. Returns (ok: bool, message: str)."""
    try:
        import google.generativeai as genai
        genai.configure(api_key=key.strip())
        m = genai.GenerativeModel(MODEL)
        r = m.generate_content("Reply with the single word: OK",
                               generation_config={"max_output_tokens": 5})
        return True, (r.text or "").strip()
    except Exception as e:  # noqa: BLE001
        return False, str(e)
