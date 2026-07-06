"""Fetch REAL labelled anatomy diagrams from Wikimedia Commons (free, no key).

Diffusion can't produce a usable labelled anatomy diagram (gibberish text). For
"labelled diagram of X" requests we instead retrieve genuine, properly-labelled
educational diagrams from Wikimedia Commons — accurate and openly licensed.
"""
from __future__ import annotations

import json
import urllib.parse
import urllib.request

_API = "https://commons.wikimedia.org/w/api.php"
_TIMEOUT = 12


def _get(params):
    url = _API + "?" + urllib.parse.urlencode(params)
    req = urllib.request.Request(url, headers={"User-Agent": "MultiMedAI/1.0 (education)"})
    with urllib.request.urlopen(req, timeout=_TIMEOUT) as r:
        return json.loads(r.read().decode("utf-8"))


def diagrams(subject: str, n: int = 4):
    """Return up to n real labelled diagrams: [{title, url, page}], ranked so
    that titles actually about the subject come first."""
    subj_l = subject.lower()
    # bias the SEARCH toward human, labelled diagrams (unless an animal is named)
    animalish = any(a in subj_l for a in ("dog", "cat", "insect", "animal", "fish",
                                          "bird", "horse", "veterinary"))
    q = (subject if animalish else f"human {subject}") + " labelled diagram anatomy"
    try:
        data = _get({
            "action": "query", "format": "json", "generator": "search",
            "gsrnamespace": "6",                       # File: namespace
            "gsrsearch": q, "gsrlimit": "40",
            "prop": "imageinfo", "iiprop": "url|mime",
            "iiurlwidth": "900",
        })
    except Exception:
        return []
    pages = (data.get("query", {}) or {}).get("pages", {})
    tokens = [t for t in subject.lower().split() if len(t) > 3]
    cand = []
    for p in pages.values():
        ii = (p.get("imageinfo") or [{}])[0]
        mime = ii.get("mime", "")
        url = ii.get("thumburl") or ii.get("url", "")
        if not url or not any(t in mime for t in ("png", "jpeg", "svg", "gif")):
            continue
        title = p.get("title", "").replace("File:", "").rsplit(".", 1)[0]
        tl = title.lower()
        rel = sum(2 for t in tokens if t in tl)             # subject match
        # off-topic: title mentions NONE of the subject words -> push down (keep as
        # last resort only). Stops a generic "human physiology" plate winning "brain".
        if tokens and not any(t in tl for t in tokens):
            rel -= 4
        # STRONG boost: title says it's actually labelled
        rel += 4 if any(w in tl for w in ("labeled", "labelled", "numbered",
                                          "annotated", "labels")) else 0
        # illustrative diagram (not a photo)
        rel += 2 if any(w in tl for w in ("diagram", "illustration", "scheme",
                                          "schematic", "blausen", "gray")) else 0
        rel += 1 if "anatomy" in tl else 0
        rel += 3 if "human" in tl else 0        # prefer human diagrams
        rel += 1 if "svg" in mime else 0
        # penalise gross-specimen photos and NON-HUMAN subjects
        rel -= 3 if any(w in tl for w in ("photograph", "photo", "autopsy",
                                          "specimen", "gross", "cadaver", "histolog")) else 0
        rel -= 6 if any(w in tl for w in ("insect", "dog", "cat", "fish", "bird",
                                          "animal", "plant", "veterinar", "frog",
                                          "horse", "cow", "sheep", "reptile",
                                          "amphibian", "bee", "worm")) else 0
        # UNLABELLED / non-English-label diagrams are useless for a labelled request
        rel -= 6 if any(w in tl for w in ("without labels", "without label",
                                          "unlabeled", "unlabelled", "no labels",
                                          "no label", "blank")) else 0
        rel -= 5 if "multilingual" in tl else 0
        # prefer ENGLISH: down-rank non-ASCII titles + ANY trailing 2-letter language
        # code (e.g. Blausen "... ku" / "... az" / "... de" variants)
        if any(ord(c) > 127 for c in title):
            rel -= 4
        import re as _re
        last = _re.sub(r"[()]", "", title.rsplit(" ", 1)[-1]).lower()
        if _re.fullmatch(r"[a-z]{2}", last):
            rel -= 4
        cand.append((rel, {"title": title, "url": url,
                           "page": "https://commons.wikimedia.org/wiki/"
                                   + urllib.parse.quote(p.get("title", ""))}))
    cand.sort(key=lambda x: -x[0])
    return [c for _, c in cand[:n]]
