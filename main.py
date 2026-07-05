"""MultiMedAI — FastAPI backend + custom web UI (replaces Streamlit).

Serves a hand-built Claude-like chat frontend (static/) and a JSON API that
reuses all the existing model logic (src/*). Run:

    venv\\Scripts\\python -m uvicorn main:app --port 8000

Then open http://localhost:8000
"""
from __future__ import annotations

import io
import uuid

from fastapi import FastAPI, File, Form, UploadFile
from fastapi.responses import FileResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from PIL import Image

from src.common import load_config, resolve, ensure_dir, get_device, ROOT
from src import engine, llm, analysis

cfg = load_config()
app = FastAPI(title="MultiMedAI")


@app.middleware("http")
async def _no_cache(request, call_next):
    """Never cache the frontend, so UI/JS changes always show on refresh."""
    resp = await call_next(request)
    p = request.url.path
    if p == "/" or p.endswith((".js", ".css", ".html")):
        resp.headers["Cache-Control"] = "no-store, must-revalidate"
    return resp
OUT = ensure_dir(resolve(cfg["paths"]["outputs"], "web"))
MIN = cfg["retrieval"].get("min_score", 0.30)


# --------------------------------------------------------------------------- media
def _media_url(path):
    p = resolve(path) if not str(path).startswith(str(ROOT)) else path
    rel = str(p).replace(str(ROOT), "").lstrip("\\/").replace("\\", "/")
    return f"/media/{rel}"


@app.get("/media/{path:path}")
def media(path: str):
    # only serve from data/ or outputs/ (no traversal)
    safe = (path or "").replace("\\", "/")
    if ".." in safe or not (safe.startswith("data/") or safe.startswith("outputs/")):
        return JSONResponse({"error": "forbidden"}, status_code=403)
    fp = resolve(*safe.split("/"))
    if not fp.is_file():
        return JSONResponse({"error": "not found"}, status_code=404)
    return FileResponse(str(fp))


def _save_img(img, tag):
    name = f"{tag}_{uuid.uuid4().hex[:8]}.png"
    fp = OUT / name
    img.save(fp)
    return _media_url(fp)


# --------------------------------------------------------------------------- helpers
def _active_image(image, image_ref):
    if image is not None:
        return Image.open(io.BytesIO(image)).convert("RGB")
    if image_ref:
        safe = image_ref.replace("media/", "", 1).replace("\\", "/")
        if ".." in safe:
            return None
        fp = resolve(*safe.split("/"))
        if fp.is_file():
            return Image.open(fp).convert("RGB")
    return None


def _img_payload(hits):
    out = []
    for path, meta, score in hits:
        name, url = analysis.provenance(meta)
        out.append({"url": _media_url(path), "score": round(score, 2),
                    "caption": meta.get("question", ""), "source": name,
                    "source_url": url, "ref": _media_url(path).replace("/media/", "")})
    return out


# --------------------------------------------------------------------------- API
@app.post("/api/upload_doc")
async def upload_doc(file: UploadFile = File(...)):
    from src import docstore
    data = await file.read()
    try:
        doc_id, n = docstore.ingest(data, file.filename)
        return {"doc_id": doc_id, "name": file.filename, "chunks": n}
    except Exception as e:  # noqa: BLE001
        return JSONResponse({"error": str(e)}, status_code=400)


@app.post("/api/add_kb")
async def add_kb(file: UploadFile = File(...)):
    from src import docstore
    data = await file.read()
    try:
        n = docstore.add_document_to_kb(data, file.filename)
        return {"added": n, "kb_total": docstore.kb_count()}
    except Exception as e:  # noqa: BLE001
        return JSONResponse({"error": str(e)}, status_code=400)


@app.get("/api/status")
def status():
    from src import docstore, cloudllm
    return {"device": get_device().upper(),
            "kb": docstore.kb_count(),
            "bank": len(analysis._bank()[1]),
            "gemini": cloudllm.available(),
            "llm": (f"{cloudllm.MODEL} (fast)" if cloudllm.available()
                    else "local Qwen (CPU)")}


@app.post("/api/set_gemini_key")
def set_gemini_key(key: str = Form(...)):
    from src import cloudllm
    ok, msg = cloudllm.check_key(key)
    if ok:
        cloudllm.set_key(key)
    return {"ok": ok, "message": msg}


@app.post("/api/research")
def api_research(topic: str = Form(...), scholar: bool = Form(False)):
    from src import research
    results, stat = research.find(topic, n=5, use_scholar=scholar)
    return {"status": stat, "results": results}


@app.post("/api/feedback")
def api_feedback(text: str = Form(...)):
    fp = ensure_dir(resolve(cfg["paths"]["outputs"], "web")) / "feedback.log"
    with open(fp, "a", encoding="utf-8") as f:
        f.write(text.replace("\n", " ") + "\n")
    return {"ok": True}


@app.post("/api/chat")
async def chat(message: str = Form(""), allow_gen: bool = Form(False),
               doc_id: int = Form(0), doc_ids: str = Form(""), image_ref: str = Form(""),
               history: str = Form(""), images: list[UploadFile] = File(None)):
    # collect MULTIPLE active images (uploaded files + one grabbed ref)
    actives = []
    for im in (images or []):
        b = await im.read()
        if b:
            actives.append(Image.open(io.BytesIO(b)).convert("RGB"))
    if not actives and image_ref:
        a = _active_image(None, image_ref)
        if a:
            actives.append(a)
    active = actives[0] if actives else None
    has_img = bool(actives)
    corrected = engine.autocorrect(message)          # light typo fix
    message = corrected
    intent = engine.detect_intent(message, has_img)
    resp = {"kind": intent, "text": "", "images": [], "note": "", "corrected": corrected}

    # ---- conversation context + follow-up modifiers ----
    import json as _json
    try:
        hist = _json.loads(history) if history else []
    except Exception:
        hist = []
    prev_user = next((h["text"] for h in reversed(hist)
                      if h.get("role") == "user" and h.get("text", "").strip()), "")
    ctx = " | ".join(h.get("text", "")[:120] for h in hist[-4:])
    low = message.lower()
    want_detail = any(k in low for k in ("more detail", "detailed", "in depth", "in-depth",
                      "elaborate", "expand", "comprehensive", "longer", "thorough",
                      "full report", "detailed summary", "detailed report", "go deeper",
                      "explain more", "not a short"))
    want_more_img = any(k in low for k in ("more image", "better image", "more scan",
                        "additional image", "more picture", "other image", "more example",
                        "different image", "more of these"))
    want_images = any(k in low for k in ("with image", "with picture", "with scan",
                      "add image", "include image", "and images", "images too"))
    # a bare follow-up ("more detail", "more images") inherits the previous topic
    resolved = message
    if (want_detail or want_more_img) and len(message.split()) <= 8 and prev_user and not has_img:
        resolved = prev_user

    # follow-up: "more / better images" -> retrieve more on the resolved topic
    if want_more_img and not has_img and not did_list:
        hits = [h for h in analysis.retrieve(resolved, 12) if h[2] >= MIN]
        if hits:
            resp["kind"] = "retrieve"
            resp["text"] = f"Here are more images for *“{resolved}”*:"
            resp["images"] = _img_payload(hits)
            resp["note"] = "Real images from open datasets. Click one to analyze it."
        else:
            resp["text"] = f"No more confident matches for *“{resolved}”*."
        return resp

    # resolve document ids (multiple supported)
    did_list = [int(x) for x in doc_ids.split(",") if x.strip().isdigit()]
    if not did_list and doc_id:
        did_list = [doc_id]

    # ---- document Q&A / report (one or more docs) ----
    if did_list and intent in ("ask", "report"):
        from src import docstore, pdfgen
        is_report = any(w in message.lower() for w in ("report", "summary", "summarize"))
        query = "objective methods results conclusions findings" if is_report else message
        hits = []
        for did in did_list:
            hits += docstore.search(did, query, k=4)
        if not hits:
            resp["text"] = "No relevant text found in the document(s)."
            return resp
        if is_report:
            report = (llm.answer_detailed(cfg, message, hits, ctx) if want_detail
                      else llm.generate_document_report(cfg, hits))
            resp["text"] = report
            pdf = pdfgen.report_to_pdf(report, "MultiMedAI — Document Report")
            resp["download"] = {"url": _media_url(pdf), "name": "MultiMedAI_report.pdf"}
            resp["preview"] = report          # right-side preview
            resp["note"] = f"Report grounded in {len(did_list)} document(s), page-cited."
        else:
            resp["text"] = (llm.answer_detailed(cfg, message, hits, ctx) if want_detail
                            else llm.answer_document(cfg, message, hits))
            resp["note"] = f"Grounded in {len(did_list)} document(s). Educational, not advice."
        return resp

    # ---- research: real literature links (PubMed free; Scholar only if asked) ----
    if intent == "research":
        import re
        from src import research
        topic = engine.extract_research_topic(message)
        low = message.lower()
        # requested count, e.g. "1 / 20 / 50 papers" (cap 50; default 6)
        cm = re.search(r"(\d{1,3})\s*(sources|papers|links|studies|results|articles|refs|references)", low)
        n = min(int(cm.group(1)), 50) if cm else 6
        # publication year, e.g. "from 2024" / "in 2023"
        ym = re.search(r"\b(19|20)\d{2}\b", message)
        year = ym.group(0) if ym else None
        # clean the topic: drop a trailing year and "from/in/of <year>"
        topic = re.sub(r"\b(from|form|in|of|during|published|dated)?\s*(19|20)\d{2}\b", "", topic).strip(" ,.")
        results, stat = research.find(topic, n=n, use_scholar=True, year=year)
        if stat == "not_medical":
            resp["text"] = ("I search literature only for the **medical fields** I cover "
                            "(pathology, brain, chest, tumours, TB, COVID, fractures, "
                            "etc.). Try *“research papers on glioma grading”*.")
        elif not results:
            resp["text"] = f"No literature found for *“{topic}”*" + (f" in {year}." if year else ".")
        else:
            hdr = f"**Evidence on _{topic}_**" + (f" ({year})" if year else "") + \
                  f" — {len(results)} sources:"
            lines = [hdr, ""]
            for r in results:
                meta = " · ".join(x for x in [r.get("venue", ""), str(r.get("year", ""))] if x)
                lines.append(f"- [{r['title']}]({r['url']}) — _{r['source']}_"
                             + (f" · {meta}" if meta else ""))
            resp["text"] = "\n".join(lines)
            n_sch = sum(1 for r in results if r.get("source") == "Google Scholar")
            resp["note"] = (f"Live from Google Scholar ({n_sch}) + PubMed "
                            f"({len(results) - n_sch}). Open-access evidence links.")
        return resp

    # ---- retrieval (explicit image request) ----
    if intent == "retrieve":
        import re
        # strip appearance/filler so the query matches well
        q = re.sub(r"\b(how|what)\s+(does|do|is|are|would)\b|\blooks?\s+(like|alike)\b|"
                   r"\bappearance of\b|\bshow me\b|\bshow\b|\bimages?\s+of\b|"
                   r"\bpictures?\s+of\b|\bplease\b|\bcan you\b|\bgive me\b|"
                   r"\bwhat it looks\b|\blook like\b", " ", message.lower())
        q = " ".join(q.split()).strip(" ?.") or message
        hits = [h for h in analysis.retrieve(q, 9) if h[2] >= MIN]
        if not hits:
            resp["text"] = (f"**No confident match** for *“{message}”* in the image "
                            "bank — I won't show misleading results or a fake image.")
        else:
            resp["text"] = f"Found **{len(hits)}** real matching images:"
            resp["images"] = _img_payload(hits)
            resp["note"] = ("Real images from open datasets — not generated. Click one "
                            "to analyze it. Score = similarity.")
        return resp

    # ---- locate / circle a region on the active image ----
    if intent == "locate":
        if not has_img:
            resp["text"] = "Attach or pick an image first, then ask me to locate a region."
            return resp
        from src import annotate
        target = engine.extract_target(message)
        annotated, found, method = annotate.annotate(active, target)
        if found and method == "gemini":
            resp["images"] = [{"url": _save_img(annotated, "annot"), "caption": target}]
            resp["text"] = f"Marked the region Gemini identifies as **“{target}”** (box + circle)."
            resp["note"] = ("Localized by Gemini vision — indicative, NOT a validated "
                            "medical detector or diagnosis.")
        elif method == "gemini_none":
            resp["text"] = (f"Gemini did **not** find “{target}” in this image. It may not "
                            "be present, or the image may be a different modality.")
            resp["note"] = "No confident region — nothing marked (avoiding a false circle)."
        elif found:
            resp["images"] = [{"url": _save_img(annotated, "annot"),
                               "caption": f"attention: {target}"}]
            resp["text"] = (f"Highlighted where the local model's *attention* falls for "
                            f"“{target}” (no Gemini key set for precise detection).")
            resp["note"] = ("⚠️ Grad-CAM attention heatmap — often imprecise, indicative "
                            "only. Add a Gemini key in Settings for accurate localization.")
        else:
            resp["text"] = f"Couldn't localize “{target}” on this image."
        return resp

    # ---- analyze the active image(s) (VLM narration + explainable read) ----
    if intent in ("vqa", "report"):
        if not has_img:
            resp["text"] = "Attach or pick an image, then ask me to analyze it."
            return resp
        from src import vlm, cloudllm
        q = message if intent == "vqa" else "Describe this medical image technically."

        # MULTIPLE images -> analyze together, per-image modality/finding summary
        if len(actives) > 1:
            qm = (message or "Compare and analyze these medical images.") + \
                 " Be technical; address each image."
            if cloudllm.available():
                try:
                    body = cloudllm.vision_multi(actives, qm)
                except Exception:
                    body = "\n\n".join(f"**Image {i+1}:** {vlm.describe(im, message)}"
                                       for i, im in enumerate(actives))
            else:
                body = "\n\n".join(f"**Image {i+1}:** {vlm.describe(im, message)}"
                                   for i, im in enumerate(actives))
            lines = []
            for i, im in enumerate(actives):
                mod, mc = analysis.modality_check(im)
                pr, rel = analysis.finding_check(im, mod)
                lines.append(f"- **Image {i+1}:** {mod} ({mc:.0%})"
                             + (f" · finding: {pr[0][0]} ({pr[0][1]:.0%})" if rel else ""))
            resp["text"] = f"{body}\n\n---\n" + "\n".join(lines)
            resp["note"] = (f"Analyzed {len(actives)} images together. Educational, "
                            "not a diagnosis.")
            return resp

        narration = vlm.describe(active, q)
        modality, mconf = analysis.modality_check(active)
        preds, reliable = analysis.finding_check(active, modality)
        finding, fconf = (preds[0][0], preds[0][1]) if reliable else (None, None)
        evid = [c for c, s in analysis.neighbors(active, 4) if s >= MIN][:2]
        explain = llm.explain_image(cfg, modality, narration, finding, fconf, evid)
        resp["text"] = (f"{explain}\n\n---\n_Image type:_ **{modality}** ({mconf:.0%})"
                        + (f" · _finding (trained ~94%):_ **{finding}** ({fconf:.0%})"
                           if reliable else ""))
        resp["note"] = ("Vision narration by moondream (general-domain) + explainable "
                        "reasoning. Educational, not a diagnosis.")
        return resp

    # ---- image generation ----
    if intent == "generate":
        if not allow_gen:
            resp["text"] = "Image generation is off — enable it to generate (slow on CPU)."
            return resp
        img = analysis.generate(message)
        resp["images"] = [{"url": _save_img(img, "gen"), "caption": message}]
        wants_label = any(w in message.lower() for w in ("label", "labelled", "labeled",
                          "annotat", "structural", "diagram", "parts"))
        resp["text"] = f"Generated a **synthetic** image for *“{message}”*."
        if wants_label:
            resp["text"] += ("\n\n⚠️ **On labels:** diffusion models **cannot render "
                             "real text** — any labels appear as gibberish glyphs. For a "
                             "properly *labelled* diagram, ask me to **“show a labelled "
                             "diagram of the brain”** and I'll retrieve a real atlas image "
                             "instead.")
        resp["note"] = "⚠️ Synthetic (SD-Turbo) — NOT a real medical image."
        return resp

    # ---- ask: KB-grounded, else clarify/general ----
    pl = message.lower()
    if any(w in pl for w in ("recent", "latest", "current", "today", "2024", "2025",
                             "news")):
        resp["text"] = ("⚠️ I'm **offline** on local models, so I can't fetch recent/live "
                        "reports. I can: **show images** (say *“show …”*), **analyze a "
                        "document** you upload, or give **general background**.")
        return resp
    from src import docstore
    kb = docstore.search_kb(resolved, k=5, min_score=0.30)
    if kb:
        ans = (llm.answer_detailed(cfg, resolved, kb, ctx) if want_detail
               else llm.answer_document(cfg, message, kb))
        # KB passages may not directly answer -> give a general answer instead of
        # the confusing "Not found in the document."
        if "not found in the document" in ans.lower() or len(ans.strip()) < 15:
            ans = llm.answer_question(cfg, resolved)
            resp["note"] = "General knowledge (no direct match in the KB). Educational."
        else:
            resp["note"] = f"Grounded in the knowledge base ({len(kb)} PubMed passages)."
        resp["text"] = ans
    else:
        resp["text"] = (llm.answer_detailed(cfg, resolved, None, ctx) if want_detail
                        else llm.answer_question(cfg, message))
        resp["note"] = ("⚠️ No close KB match — general (offline) knowledge, not grounded "
                        "in a source.")
    # "with images" -> attach a few real illustrative images to the answer
    if want_images:
        ims = [h for h in analysis.retrieve(resolved, 3) if h[2] >= MIN]
        if ims:
            resp["images"] = _img_payload(ims)
    return resp


app.mount("/", StaticFiles(directory=str(resolve("static")), html=True), name="static")
