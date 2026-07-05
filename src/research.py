"""Research-link finder — real open-access literature for documented evidence.

Sources (all free):
  - Google Scholar via SerpApi (user's free key; 250 searches/month -> CACHED hard)
  - PubMed E-utilities (free, official, no key, no quota)

Every query is CACHED in SQLite so repeats NEVER spend SerpApi quota. Returns
citable links the user can open — for evidence behind an answer or a topic.
"""
from __future__ import annotations

import json
import sqlite3
import time
import urllib.parse
import urllib.request

from src.common import resolve, ensure_dir

_TIMEOUT = 12

# Only search literature for the medical fields this app actually covers. This
# gate protects the limited SerpApi quota and keeps results on-topic.
MEDICAL_TERMS = (
    # imaging / pathology
    "pathology", "histopath", "radiology", "imaging", "mri", "ct scan", "x-ray",
    "xray", "radiograph", "ultrasound", "scan", "biopsy", "cytology", "lesion",
    # oncology
    "tumor", "tumour", "cancer", "carcinoma", "carcinogen", "malignan", "metasta",
    "glioma", "glioblastoma", "sarcoma", "lymphoma", "neoplasm", "adenocarcinoma",
    "mass", "nodule", "oncolog",
    # brain / neuro
    "brain", "neuro", "hemorrhage", "haemorrhage", "stroke", "infarct", "aneurysm",
    "hydrocephalus", "meningioma",
    # chest / pulmonary / infectious
    "chest", "lung", "pulmonary", "tuberculosis", "tb", "covid", "pneumonia",
    "pleural", "effusion", "cardiomegaly", "respiratory", "infection", "sepsis",
    # bone / skeletal / musculoskeletal
    "bone", "fracture", "osteo", "arthritis", "joint", "skeletal", "orthop",
    "spine", "spinal", "vertebra", "disc", "ligament", "tendon", "dislocation",
    "osteoporos", "scoliosis", "cartilage",
    # muscular
    "muscle", "muscular", "myopathy", "myositis", "sarcopenia", "atrophy",
    "rhabdo", "dystrophy", "myalgia",
    # dermatology / skin
    "skin", "dermatolog", "dermat", "melanoma", "nevus", "naevus", "keratosis",
    "eczema", "psoriasis", "rash", "mole", "squamous", "basal cell", "acne",
    "ulcer", "wound", "burn",
    # hair / scalp
    "hair", "scalp", "alopecia", "tricholog", "baldness", "dandruff",
    "folliculitis", "seborrhe",
    # neuro / dementia
    "alzheimer", "dementia", "cognitive", "parkinson", "epilep", "seizure",
    "multiple sclerosis", "migraine",
    # abdomen / other organs
    "abdomen", "liver", "kidney", "renal", "hepatic", "pancrea", "spleen",
    "retina", "retinopathy", "glaucoma", "diabetic", "cardiac", "heart",
    # endocrine / repro / general
    "diabetes", "hormone", "gynaec", "gynec", "obstetric", "pregnan", "fetal",
    "prostate", "breast", "mammogra",
    "diagnos", "disease", "ailment", "clinical", "medical", "patient", "treatment",
    "dosage", "therapy", "prognosis", "inflammation", "necrosis", "cell",
    "syndrome", "screening", "biopsy", "surgery", "surgical",
)


def is_medical(query: str) -> bool:
    """True only if the query is about a medical field this app addresses."""
    q = (query or "").lower()
    return any(t in q for t in MEDICAL_TERMS)


def _serpapi_key():
    kf = resolve(".keys.json")
    if kf.is_file():
        try:
            return json.loads(kf.read_text(encoding="utf-8")).get("serpapi")
        except Exception:
            return None
    return None


def _cache():
    ensure_dir(resolve("weights"))
    con = sqlite3.connect(str(resolve("weights", "research_cache.sqlite")))
    con.execute("""CREATE TABLE IF NOT EXISTS cache(
        key TEXT PRIMARY KEY, results TEXT, created REAL)""")
    return con


def _get_cache(key):
    con = _cache()
    row = con.execute("SELECT results FROM cache WHERE key=?", (key,)).fetchone()
    con.close()
    return json.loads(row[0]) if row else None


def _put_cache(key, results):
    con = _cache()
    con.execute("INSERT OR REPLACE INTO cache(key, results, created) VALUES(?,?,?)",
                (key, json.dumps(results), time.time()))
    con.commit(); con.close()


def _http_json(url):
    req = urllib.request.Request(url, headers={"User-Agent": "MultiMedAI/1.0"})
    with urllib.request.urlopen(req, timeout=_TIMEOUT) as r:
        return json.loads(r.read().decode("utf-8"))


def pubmed(query, n=5, year=None):
    """PubMed E-utilities — free, no key, no quota. Optional publication year filter."""
    base = "https://eutils.ncbi.nlm.nih.gov/entrez/eutils"
    term = query + (f" AND {year}[pdat]" if year else "")
    q = urllib.parse.quote(term)
    try:
        ids = _http_json(f"{base}/esearch.fcgi?db=pubmed&retmode=json&retmax={n}&term={q}")
        idlist = ids.get("esearchresult", {}).get("idlist", [])
        if not idlist:
            return []
        summ = _http_json(f"{base}/esummary.fcgi?db=pubmed&retmode=json&id={','.join(idlist)}")
        res = summ.get("result", {})
        out = []
        for pid in idlist:
            it = res.get(pid, {})
            if not it:
                continue
            out.append({
                "title": it.get("title", "").strip("."),
                "year": (it.get("pubdate", "") or "")[:4],
                "venue": it.get("fulljournalname", it.get("source", "")),
                "url": f"https://pubmed.ncbi.nlm.nih.gov/{pid}/",
                "source": "PubMed"})
        return out
    except Exception:
        return []


def scholar(query, n=5, year=None):
    """Google Scholar via SerpApi — CACHED (free plan = 250/month)."""
    key = _serpapi_key()
    if not key:
        return []
    ckey = f"scholar::{n}::{year}::" + query.lower().strip()
    cached = _get_cache(ckey)
    if cached is not None:
        return cached
    q = urllib.parse.quote(query)
    yr = f"&as_ylo={year}&as_yhi={year}" if year else ""
    url = (f"https://serpapi.com/search.json?engine=google_scholar&q={q}"
           f"&num={min(n, 20)}{yr}&api_key={key}")
    try:
        data = _http_json(url)
        out = []
        for it in data.get("organic_results", [])[:n]:
            pub = it.get("publication_info", {}).get("summary", "")
            out.append({"title": it.get("title", ""), "url": it.get("link", ""),
                        "venue": pub, "year": "", "source": "Google Scholar"})
        _put_cache(ckey, out)   # spend quota only once per unique query
        return out
    except Exception:
        return []


def find(query, n=5, use_scholar=True, year=None):
    """Combined evidence links, MEDICAL topics only. Mixes ~half Google Scholar
    (cached) + half PubMed by default; if Scholar is unavailable, PubMed fills in.
    Returns exactly up to `n` results. Status: 'ok' | 'not_medical'.
    """
    if not is_medical(query):
        return [], "not_medical"
    n = max(1, min(int(n), 50))
    sch = scholar(query, (n + 1) // 2, year) if use_scholar else []
    pub = pubmed(query, n, year)          # get up to n so PubMed can fill gaps

    seen, out, si, pi = set(), [], 0, 0
    want_sch = min(len(sch), (n + 1) // 2)
    # interleave: scholar, pubmed, scholar, pubmed … then fill remainder
    while len(out) < n and (si < len(sch) or pi < len(pub)):
        if si < want_sch:
            cand = sch[si]; si += 1
        elif pi < len(pub):
            cand = pub[pi]; pi += 1
        elif si < len(sch):
            cand = sch[si]; si += 1
        else:
            break
        k = (cand.get("title") or "").lower()[:80]
        if k and k not in seen:
            seen.add(k); out.append(cand)
    return out[:n], "ok"
