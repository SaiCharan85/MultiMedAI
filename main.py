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


def _access_methods():
    """Two ways to request access to restricted images (needs proof of being a
    medical professional/student + a clinical reason)."""
    acc = cfg.get("access", {})
    return {
        "email": acc.get("email", "access@multimedai.example"),
        "form_url": acc.get("form_url", ""),
        "required": "reason for access + role (medical student / doctor) + proof "
                    "(institution + registration/enrolment ID, or a credential file)",
    }


def _safe_filter(hits, granted=False):
    """Drop restricted images (genital/pubic/breast/buttock + minors) via caption
    keyword + NSFW image classifier. Returns (kept, n_hidden). Skipped if access
    has been granted (request-based professional access)."""
    if granted:
        return list(hits), 0
    from src import safety
    kept, hidden = [], 0
    for path, meta, score in hits:
        if safety.caption_restricted(meta.get("question", "")):
            hidden += 1; continue
        try:
            if safety.nsfw_score(Image.open(path)) >= 0.6:
                hidden += 1; continue
        except Exception:
            pass
        kept.append((path, meta, score))
    return kept, hidden


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


@app.post("/api/request_access")
async def api_request_access(reason: str = Form(...), role: str = Form(""),
                             proof: str = Form(""), credential: UploadFile = File(None)):
    """Request-based access to restricted images. Requires a clinical reason, a
    medical role, and proof (institution + registration ID and/or a credential
    file). Every request is AUDIT-LOGGED. Grant here is PROVISIONAL/self-attested
    for the session; genuine verification is manual (see the email method)."""
    import time as _t
    from src import safety
    cred_note = ""
    if credential is not None:
        data = await credential.read()
        cdir = ensure_dir(resolve(cfg["paths"]["outputs"], "web", "credentials"))
        (cdir / f"{int(_t.time())}_{credential.filename}").write_bytes(data)
        cred_note = f" | file={credential.filename}"
    role_ok = any(k in (role + " " + proof).lower() for k in
                  ("student", "doctor", "physician", "clinician", "resident", "nurse",
                   "md", "mbbs", "professor", "researcher"))
    has_proof = len(proof.strip()) >= 6 or credential is not None
    granted = safety.justification_ok(reason) and role_ok and has_proof

    fp = ensure_dir(resolve(cfg["paths"]["outputs"], "web")) / "access_requests.log"
    with open(fp, "a", encoding="utf-8") as f:
        f.write(f"[{'PROVISIONAL-GRANT' if granted else 'PENDING/DENIED'}] role={role!r} "
                f"proof={proof.strip()[:120]!r}{cred_note} reason={reason.strip()[:200]!r}\n")
    if granted:
        msg = ("✅ Provisional access granted for this session (self-attested + "
               "audit-logged). For permanent verified access, also email us the proof.")
    else:
        msg = ("❌ Not granted. Provide: a clinical reason, your role (medical "
               "student / doctor), and proof (institution + registration/enrolment ID "
               "or a credential file). Or use the email method for manual verification.")
    return {"granted": granted, "message": msg}


@app.post("/api/feedback")
def api_feedback(text: str = Form(...)):
    fp = ensure_dir(resolve(cfg["paths"]["outputs"], "web")) / "feedback.log"
    with open(fp, "a", encoding="utf-8") as f:
        f.write(text.replace("\n", " ") + "\n")
    return {"ok": True}


@app.post("/api/chat")
async def chat(message: str = Form(""), allow_gen: bool = Form(False),
               doc_id: int = Form(0), doc_ids: str = Form(""), image_ref: str = Form(""),
               history: str = Form(""), access_granted: bool = Form(False),
               images: list[UploadFile] = File(None)):
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
    if want_more_img and not has_img and not doc_ids.strip() and not doc_id:
        hits = [h for h in analysis.retrieve(resolved, 18) if h[2] >= MIN]
        hits, _hidden = _safe_filter(hits, granted=access_granted)
        hits = hits[:12]
        if hits:
            resp["kind"] = "retrieve"
            resp["text"] = f"Here are more images for *“{resolved}”*:"
            resp["images"] = _img_payload(hits)
            resp["note"] = "Real images from open datasets. Click one to analyze it."
        else:
            resp["text"] = f"No more confident matches for *“{resolved}”*."
        return resp

    # ---- conversation-context signals shared by the follow-up branches ----
    # Does the message refer back to the previous answer? (kept specific so a fresh,
    # unrelated question doesn't get wrongly grounded in the last reply.)
    refers_prev = any(k in low for k in ("above", "previously", "you mentioned",
                      "you said", "that you mentioned", "the above", "that report",
                      "this report", "from the report", "from the above", "aforementioned",
                      "you analyzed", "you analysed", "the previous", "earlier answer",
                      "these images", "those images", "the analysis")) \
        or (" report" in low and any(w in low for w in ("above", "previous", "that", "this")))
    prev_bot = next((h.get("text", "") for h in reversed(hist)
                     if h.get("role") in ("bot", "assistant")
                     and len(h.get("text", "")) > 60), "")
    # REGENERATE the previous answer as a structured report/summary. This is an EXPLICIT
    # "summarize / make a report of the above" — NOT a plain question about the previous
    # answer (that is answered contextually in the ask branch). Must not fall through to
    # the image gallery just because the word "images"/"report" appears.
    is_regen = refers_prev and (engine.is_report_request(message)
               or any(k in low for k in ("summarize", "summarise", "summary of",
                                          "make a report", "generate a report",
                                          "write a report", "give me a report",
                                          "into a report", "as a report", "a summary")))
    if (not has_img and not doc_ids.strip() and not doc_id and is_regen and prev_bot):
        from src import cloudllm, pdfgen
        instr = ("You are writing a clear, well-structured medical/educational report. "
                 "Using ONLY the prior answer below, produce a DETAILED report in ENGLISH "
                 "with markdown headers and bullet points: a short **Overview**, the key "
                 "**Points / Findings**, and a brief **Summary**. Keep the SAME topic as "
                 "the prior answer. Do NOT mention images or scans unless the prior answer "
                 "does. Do not invent facts beyond the prior answer.")
        try:
            if cloudllm.available():
                report = cloudllm.text(instr, "PRIOR ANSWER:\n" + prev_bot, max_tokens=1200)
            else:
                report = llm.answer_question(cfg, instr, context=prev_bot)
        except Exception:
            report = llm.answer_question(cfg, instr, context=prev_bot)
        resp["kind"] = "report"
        resp["text"] = report
        pdf = pdfgen.report_to_pdf(report, "MultiMedAI - Report")
        resp["download"] = {"url": _media_url(pdf), "name": "MultiMedAI_report.pdf"}
        resp["preview"] = report
        resp["note"] = ("Report composed from the previous answer (same topic). "
                        "Educational, not a diagnosis.")
        return resp

    # ---- labelled ANATOMY DIAGRAM request (no active image) -> fetch REAL
    #      labelled diagrams from Wikimedia (diffusion can't render usable labels) ----
    dl = message.lower()
    # LABELLED / DIAGRAM requests -> real illustrative Wikimedia diagrams (even if
    # phrased as "generate ..."). Only NON-labelled generation uses SD-Turbo.
    if not has_img and any(k in dl for k in (
            "labelled diagram", "labeled diagram", "diagram of", "anatomy of",
            "atlas of", "labelled image", "labeled image", "anatomy diagram",
            "with labels", "labels of", "labelled anatomy", "labelled", "labeled",
            "diagram")):
        import re
        from src import atlas
        subj = re.sub(r"\b(can you|could you|please|generate|create|make|give me|show me|"
                      r"a|an|the|of|with all parts|with labels|labels|labelled|labeled|"
                      r"diagram|anatomy|atlas|image|picture|human)\b", " ", dl)
        subj = " ".join(subj.split()).strip(" ?.") or message
        digs = atlas.diagrams(subj, 6)
        if digs:
            best = digs[0]                       # single highest-match labelled diagram
            resp["kind"] = "diagram"
            resp["text"] = (f"**Labelled diagram of {subj}** — *{best['title']}* "
                            f"([source]({best['page']})):")
            # no 'source' key -> frontend renders it as ONE large image (not a card)
            resp["images"] = [{"url": best["url"], "caption": best["title"][:70]}]
            resp["note"] = ("Real, openly-licensed labelled diagram from Wikimedia "
                            "Commons (not generated).")
            return resp
        # if none found, fall through to normal handling

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
        from src import safety
        wants_restricted = safety.query_wants_restricted(message)
        cand = [h for h in analysis.retrieve(q, 18) if h[2] >= MIN]
        kept, hidden = _safe_filter(cand, granted=access_granted)
        kept = kept[:9]
        # request-based gate: restricted content needs verified access
        if not kept and (wants_restricted or hidden) and not access_granted:
            resp["restricted"] = True
            resp["text"] = (
                "🔒 **Access-restricted content.** The images matching this request are "
                "restricted (explicit/intimate content, or involving minors under 18). "
                "To view them you must **verify you are a medical professional or student "
                "and state a clinical reason**, via one of the two methods below.")
            resp["access"] = _access_methods()
            return resp
        if not kept:
            resp["text"] = (f"**No confident match** for *“{message}”* in the image "
                            "bank — I won't show misleading results or a fake image.")
        else:
            resp["text"] = f"Found **{len(kept)}** real matching images:"
            resp["images"] = _img_payload(kept)
            note = ("Real images from open datasets — not generated. Click one to "
                    "analyze it.")
            if hidden and not access_granted:
                note += (f" · **{hidden} restricted image(s) hidden** — request "
                         "verified access to view.")
                resp["restricted"] = True
                resp["access"] = _access_methods()
            resp["note"] = note
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
            qm = ((message or "Analyze these medical images.") +
                  " For EACH image: describe the ANATOMY and abnormal FINDINGS actually "
                  "visible (location, size, shape, density/signal, borders) and the most "
                  "likely disease/ailment as a consideration. Name the modality in only a "
                  "few words. Do NOT explain imaging physics (no radiation, detectors, "
                  "wavelengths, Tesla, pulse sequences). Markdown, per-image headers. "
                  "Respond in ENGLISH ONLY.")
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
            if intent == "report":
                from src import pdfgen
                pdf = pdfgen.report_to_pdf(resp["text"], "MultiMedAI — Image Analysis Report")
                resp["download"] = {"url": _media_url(pdf), "name": "MultiMedAI_report.pdf"}
                resp["preview"] = resp["text"]
                resp["note"] = f"Report on {len(actives)} images. Educational, not a diagnosis."
            return resp

        # local (free) grounding signals — no Gemini quota used here
        modality, mconf = analysis.modality_check(active)
        preds, reliable = analysis.finding_check(active, modality)
        finding, fconf = (preds[0][0], preds[0][1]) if reliable else (None, None)
        evid = [c for c, s in analysis.neighbors(active, 4) if s >= MIN][:2]
        from src import cloudllm
        if cloudllm.available():
            # ONE Gemini vision call (saves quota): image + grounding -> explainable read
            ground = ""
            if reliable:
                ground += f"A trained classifier (~94% acc) suggests: {finding}. "
            if evid:
                ground += "Similar library cases: " + "; ".join(evid) + ". "
            uq = f"The user asks: {message}\n" if intent == "vqa" and message else ""
            cx = f"Conversation so far (for follow-up context): {ctx}\n" if ctx else ""
            vp = (llm.EXPLAIN_INSTRUCTIONS + "\n\n" + cx + uq + ground +
                  "Analyze the attached medical image accordingly.")
            try:
                explain = cloudllm.vision(active, vp, max_tokens=700)
            except Exception:
                explain = llm.explain_image(cfg, modality, vlm.describe(active, q),
                                            finding, fconf, evid)
        else:
            explain = llm.explain_image(cfg, modality, vlm.describe(active, q),
                                        finding, fconf, evid)
        # only claim a modality when the detector is reasonably confident (the
        # BiomedCLIP zero-shot guess is unreliable at low confidence -> hide it,
        # since the vision narration already describes the image correctly).
        extra = ""
        if reliable:
            extra += f"\n\n---\n_Finding (trained classifier ~94%):_ **{finding}** ({fconf:.0%})"
        if mconf >= 0.35:
            extra += ("\n\n" if not extra else " · ") + f"_Likely image type:_ **{modality}** ({mconf:.0%})"
        resp["text"] = explain + extra
        resp["note"] = ("Vision analysis (Gemini/moondream) + explainable reasoning "
                        "grounded in similar cases. Educational, not a diagnosis.")
        if intent == "report":
            from src import pdfgen
            pdf = pdfgen.report_to_pdf(resp["text"], "MultiMedAI — Image Analysis Report")
            resp["download"] = {"url": _media_url(pdf), "name": "MultiMedAI_report.pdf"}
            resp["preview"] = resp["text"]
        return resp

    # ---- image generation ----
    if intent == "generate":
        if not allow_gen:
            resp["text"] = "Image generation is off — enable it to generate (slow on CPU)."
            return resp
        img = analysis.generate(message)
        wants_label = any(w in message.lower() for w in ("label", "labelled", "labeled",
                          "annotat", "structural", "diagram", "parts"))
        note = "⚠️ Synthetic (SD-Turbo) — NOT a real medical image."
        if wants_label:
            from src import annotate
            leg = annotate.legend(img, message)   # text legend beside the image
            if leg:
                resp["text"] = (f"Generated a **synthetic** image for *“{message}”*.\n\n"
                                "**Labelled structures (legend):**\n" + leg)
                note += (" On-image text can't be rendered by diffusion, so labels are "
                         "given as a legend (Gemini) — indicative, not authoritative.")
            else:
                resp["text"] = (f"Generated a **synthetic** image for *“{message}”*. "
                                "Labelling needs Gemini quota — or try *“labelled diagram "
                                "of the brain”* (no 'generate') to retrieve a real image.")
        else:
            resp["text"] = f"Generated a **synthetic** image for *“{message}”*."
        resp["images"] = [{"url": _save_img(img, "gen"), "caption": message}]
        resp["note"] = note
        return resp

    # ---- ask: KB-grounded, else clarify/general ----
    from src import cloudllm
    pl = message.lower()
    # Only claim "offline / can't fetch live data" when there is NO cloud model. With
    # Gemini connected we CAN answer general/recent questions (with a not-live caveat),
    # or point the user to the research feature for actual up-to-date literature.
    if not cloudllm.available() and any(w in pl for w in (
            "recent", "latest", "current", "today", "2024", "2025", "2026", "news")):
        resp["text"] = ("⚠️ I'm **offline** on local models, so I can't fetch recent/live "
                        "reports. I can: **show images** (say *“show …”*), **analyze a "
                        "document** you upload, or give **general background**.")
        return resp
    # an EXPLICIT "generate/make/write a report on X" request (no doc/image) -> give a
    # FULLER, report-style answer AND a downloadable PDF. NB: we require a real
    # create-a-report phrase (engine.is_report_request), so the mere WORD "report"
    # appearing in a question ("...from the above report") does NOT trigger a PDF.
    is_report_ask = engine.is_report_request(message)
    detailed = want_detail or is_report_ask
    # CONTEXTUAL FOLLOW-UP: the user refers to the previous answer ("elaborate on the
    # risk factors you mentioned above", "throw more light on X from the report"). Ground
    # the answer in the PREVIOUS answer + recent context so we stay on-topic, instead of
    # a fresh KB search that can drift to unrelated passages.
    if refers_prev and prev_bot:
        gctx = prev_bot[:3000] + (("\n\nRecent conversation: " + ctx) if ctx else "")
        resp["text"] = (llm.answer_detailed(cfg, message, None, gctx) if detailed
                        else llm.answer_question(cfg, message, context=gctx))
        resp["note"] = "Contextual follow-up — grounded in the previous answer (same topic)."
        return resp
    from src import docstore
    kb = docstore.search_kb(resolved, k=5, min_score=0.30)
    if kb:
        ans = (llm.answer_detailed(cfg, resolved, kb, ctx) if detailed
               else llm.answer_document(cfg, message, kb))
        # KB passages may not directly answer -> give a general answer instead of
        # the confusing "Not found in the document."
        if "not found in the document" in ans.lower() or len(ans.strip()) < 15:
            ans = llm.answer_question(cfg, resolved, context=ctx)
            resp["note"] = "General knowledge (no direct match in the KB). Educational."
        else:
            resp["note"] = f"Grounded in the knowledge base ({len(kb)} PubMed passages)."
        resp["text"] = ans
    else:
        resp["text"] = (llm.answer_detailed(cfg, resolved, None, ctx) if detailed
                        else llm.answer_question(cfg, message, context=ctx))
        resp["note"] = ("General knowledge (not grounded in a specific source). "
                        "Educational, not a diagnosis.")
    # "with images" -> attach a few real illustrative images to the answer
    if want_images:
        ims = [h for h in analysis.retrieve(resolved, 8) if h[2] >= MIN]
        ims, _h = _safe_filter(ims, granted=access_granted)
        ims = ims[:3]
        if ims:
            resp["images"] = _img_payload(ims)
    # explicit "generate a report" -> also produce a downloadable PDF + preview
    if is_report_ask and resp["text"] and len(resp["text"]) > 40:
        import re as _re
        from src import pdfgen
        topic = _re.sub(r"\b(can you|could you|please|generate|create|make|write|give me|"
                        r"a|an|the|report|detailed|on|about|explaining|explain|what|is|of)\b",
                        " ", pl)
        topic = " ".join(topic.split()).strip(" ?.:") or "Report"
        resp["kind"] = "report"
        pdf = pdfgen.report_to_pdf(resp["text"], "MultiMedAI - " + topic.title()[:60])
        resp["download"] = {"url": _media_url(pdf), "name": "MultiMedAI_report.pdf"}
        resp["preview"] = resp["text"]
    return resp


app.mount("/", StaticFiles(directory=str(resolve("static")), html=True), name="static")
