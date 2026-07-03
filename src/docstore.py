"""Document store for expert document analysis (RAG).

Lets a doctor/researcher upload a thesis/PDF and ask questions; answers are
GROUNDED in retrieved passages with page citations (no free-floating LLM
invention). Demonstrates the recommended stack:

  - SQLite (RDB)      : document registry, chunk text, and a QUERY AUDIT LOG
                        (provenance — important for medical/research use)
  - MiniLM embeddings : all-MiniLM-L6-v2 (free, CPU-fast) for chunk/question vectors
  - vector search     : numpy cosine over chunk embeddings stored in SQLite as
                        blobs (a sqlite-vec-style store). Chroma/LanceDB are
                        drop-in swaps at larger scale.

All local, all free, CPU-only.
"""
from __future__ import annotations

import functools
import sqlite3
import time

import numpy as np

from src.common import resolve, ensure_dir

EMB_MODEL = "sentence-transformers/all-MiniLM-L6-v2"
_CHUNK_CHARS = 900
_CHUNK_OVERLAP = 150


def _db_path():
    ensure_dir(resolve("weights"))
    return str(resolve("weights", "documents.sqlite"))


def _conn():
    con = sqlite3.connect(_db_path())
    con.execute("""CREATE TABLE IF NOT EXISTS documents(
        id INTEGER PRIMARY KEY, name TEXT, created REAL, n_chunks INTEGER)""")
    con.execute("""CREATE TABLE IF NOT EXISTS chunks(
        id INTEGER PRIMARY KEY, doc_id INTEGER, idx INTEGER, page INTEGER,
        text TEXT, emb BLOB)""")
    con.execute("""CREATE TABLE IF NOT EXISTS queries(
        id INTEGER PRIMARY KEY, doc_id INTEGER, question TEXT, created REAL,
        chunk_ids TEXT)""")   # AUDIT LOG: what was asked + which chunks answered
    # standing knowledge base: open medical literature + doctor-curated passages
    con.execute("""CREATE TABLE IF NOT EXISTS kb(
        id INTEGER PRIMARY KEY, source TEXT, title TEXT, text TEXT, emb BLOB)""")
    return con


@functools.lru_cache(maxsize=1)
def _embedder():
    from sentence_transformers import SentenceTransformer
    return SentenceTransformer(EMB_MODEL, device="cpu")


def embed(texts):
    m = _embedder()
    return m.encode(list(texts), normalize_embeddings=True,
                    convert_to_numpy=True).astype("float32")


# ---------------------------------------------------------------------------
def _extract_pages(file_bytes: bytes, name: str):
    """Return list of (page_number, text). Supports PDF and plain text."""
    if name.lower().endswith(".pdf"):
        import io
        from pypdf import PdfReader
        reader = PdfReader(io.BytesIO(file_bytes))
        return [(i + 1, (pg.extract_text() or "")) for i, pg in enumerate(reader.pages)]
    text = file_bytes.decode("utf-8", errors="ignore")
    return [(1, text)]


def _chunk(pages):
    """Split pages into overlapping chunks, keeping the page number."""
    out = []
    for page, text in pages:
        text = " ".join(text.split())
        if not text:
            continue
        i = 0
        while i < len(text):
            out.append((page, text[i:i + _CHUNK_CHARS]))
            i += _CHUNK_CHARS - _CHUNK_OVERLAP
    return out


def ingest(file_bytes: bytes, name: str) -> tuple[int, int]:
    """Ingest a document: extract -> chunk -> embed -> store. Returns (doc_id, n_chunks)."""
    pages = _extract_pages(file_bytes, name)
    chunks = _chunk(pages)
    if not chunks:
        raise ValueError("No extractable text (scanned PDF? needs OCR).")
    embs = embed([c for _, c in chunks])

    con = _conn()
    cur = con.execute("INSERT INTO documents(name, created, n_chunks) VALUES(?,?,?)",
                      (name, time.time(), len(chunks)))
    doc_id = cur.lastrowid
    con.executemany(
        "INSERT INTO chunks(doc_id, idx, page, text, emb) VALUES(?,?,?,?,?)",
        [(doc_id, i, pg, txt, embs[i].tobytes())
         for i, (pg, txt) in enumerate(chunks)])
    con.commit(); con.close()
    return doc_id, len(chunks)


def list_documents():
    con = _conn()
    rows = con.execute("SELECT id, name, n_chunks FROM documents ORDER BY id DESC").fetchall()
    con.close()
    return rows


def search(doc_id: int, question: str, k: int = 5):
    """Return top-k chunks for a question, and LOG the query (audit)."""
    con = _conn()
    rows = con.execute("SELECT id, page, text, emb FROM chunks WHERE doc_id=?",
                       (doc_id,)).fetchall()
    if not rows:
        con.close(); return []
    embs = np.stack([np.frombuffer(r[3], dtype="float32") for r in rows])
    q = embed([question])[0]
    sims = embs @ q
    top = np.argsort(-sims)[:k]
    hits = [{"chunk_id": rows[i][0], "page": rows[i][1], "text": rows[i][2],
             "score": float(sims[i])} for i in top]
    con.execute("INSERT INTO queries(doc_id, question, created, chunk_ids) VALUES(?,?,?,?)",
                (doc_id, question, time.time(), ",".join(str(h["chunk_id"]) for h in hits)))
    con.commit(); con.close()
    return hits


# ===========================================================================
# STANDING KNOWLEDGE BASE — open medical literature (grounds general Q&A) +
# doctor-curated additions. Persists across sessions in the same SQLite file.
# ===========================================================================
def kb_count() -> int:
    con = _conn()
    n = con.execute("SELECT COUNT(*) FROM kb").fetchone()[0]
    con.close()
    return n


def kb_sources():
    con = _conn()
    rows = con.execute("SELECT source, COUNT(*) FROM kb GROUP BY source").fetchall()
    con.close()
    return rows


def _kb_insert(con, texts, titles, source):
    embs = embed(texts)
    con.executemany("INSERT INTO kb(source, title, text, emb) VALUES(?,?,?,?)",
                    [(source, titles[i], texts[i], embs[i].tobytes())
                     for i in range(len(texts))])
    con.commit()


def build_kb(n=8000, batch=256):
    """Ingest an open medical corpus (PubMedQA abstracts) into the KB. Streamed
    so no giant download. Idempotent-ish: skips if already populated."""
    if kb_count() >= n * 0.8:
        print(f"[kb] already populated ({kb_count()} passages) — skipping.")
        return kb_count()
    from datasets import load_dataset

    con = _conn()
    ds = load_dataset("pubmed_qa", "pqa_artificial", split="train", streaming=True)
    texts, titles, total = [], [], 0
    for ex in ds:
        ctx = ex.get("context")
        if isinstance(ctx, dict):
            text = " ".join(ctx.get("contexts", []))
        elif isinstance(ctx, list):
            text = " ".join(ctx)
        else:
            text = str(ctx or "")
        text = " ".join(text.split())
        if len(text) < 120:
            continue
        texts.append(text[:1400])
        titles.append((ex.get("question") or "")[:140])
        if len(texts) >= batch:
            _kb_insert(con, texts, titles, "PubMed abstract")
            total += len(texts); texts, titles = [], []
            print(f"[kb] ingested {total}", end="\r")
        if total >= n:
            break
    if texts:
        _kb_insert(con, texts, titles, "PubMed abstract")
        total += len(texts)
    con.close()
    print(f"\n[kb] done — {kb_count()} passages in the knowledge base.")
    return kb_count()


def add_document_to_kb(file_bytes: bytes, name: str) -> int:
    """Doctor-curation: permanently add an uploaded document's passages to the KB."""
    pages = _extract_pages(file_bytes, name)
    chunks = _chunk(pages)
    if not chunks:
        raise ValueError("No extractable text.")
    con = _conn()
    _kb_insert(con, [c for _, c in chunks],
               [f"{name} (p{p})" for p, _ in chunks], f"curated:{name}")
    con.close()
    _kb_cache.clear()
    return len(chunks)


_kb_cache = {}


def _kb_matrix():
    """Load (and cache) all KB embeddings + rows for fast search."""
    if "E" in _kb_cache:
        return _kb_cache["E"], _kb_cache["rows"]
    con = _conn()
    rows = con.execute("SELECT id, source, title, text, emb FROM kb").fetchall()
    con.close()
    if not rows:
        return None, []
    E = np.stack([np.frombuffer(r[4], dtype="float32") for r in rows])
    _kb_cache["E"] = E; _kb_cache["rows"] = rows
    return E, rows


def search_kb(question: str, k: int = 5, min_score: float = 0.25):
    """Search the standing KB. Returns passages shaped for llm.answer_document
    (page = passage index; text prefixed with its source/title for citation)."""
    E, rows = _kb_matrix()
    if E is None:
        return []
    q = embed([question])[0]
    sims = E @ q
    order = np.argsort(-sims)[:k]
    hits = []
    for rank, i in enumerate(order, 1):
        if float(sims[i]) < min_score:
            continue
        _id, source, title, text, _ = rows[i]
        hits.append({"page": rank, "score": float(sims[i]), "source": source,
                     "title": title, "text": f"({source}: {title}) {text}"})
    return hits
