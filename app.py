"""MultiMedAI — themed Streamlit chat engine (local, CPU).

Run:  streamlit run app.py   ->  http://localhost:8501

A retrieval/routing medical assistant: ask in natural language and it answers
with REAL pathology images (retrieval), answers questions about an uploaded
image (VQA), writes a short report (BLIP caption), or generates an image (SD).

HONESTY: retrieval returns REAL dataset images; VQA uses the trained head if
present; metrics shown are read from files produced by real eval runs.
"""
from __future__ import annotations

from pathlib import Path

import streamlit as st
from PIL import Image

from src.common import load_config, resolve, get_device
from src import engine, llm

cfg = load_config()
DEVICE = get_device()

st.set_page_config(page_title="MultiMedAI", page_icon="🧬", layout="wide")

# ============================ THEME (unique: "Clinical Aurora") =============
st.markdown(
    """
    <style>
    @import url('https://fonts.googleapis.com/css2?family=Plus+Jakarta+Sans:wght@500;600;700;800&family=Inter:wght@400;500;600&display=swap');

    :root{
      --bg:#0b1020; --bg2:#0e1428; --panel:#141c33; --panel2:#182142;
      --brand:#14b8a6; --brand2:#6366f1; --ink:#eaf0fb; --dim:#93a1c0;
      --line:rgba(148,163,184,.16); --good:#34d399; --warn:#fbbf24;
    }
    .stApp{
      background:
        radial-gradient(1200px 600px at 8% -10%, rgba(20,184,166,.10), transparent 55%),
        radial-gradient(1100px 600px at 95% 0%, rgba(99,102,241,.12), transparent 55%),
        linear-gradient(180deg,var(--bg),var(--bg2));
      color:var(--ink);
    }
    #MainMenu, header, footer{visibility:hidden;}
    .block-container{padding-top:1.2rem; max-width:1080px; position:relative; z-index:1;}
    section[data-testid="stSidebar"]{position:relative; z-index:1;}
    body,p,div,span,label,input,textarea{font-family:'Inter',system-ui,sans-serif;}

    /* ---- animated DNA double-helix background (subtle, behind everything) ---- */
    .stApp::before{
      content:""; position:fixed; inset:-12% -12% -12% -12%; z-index:0; pointer-events:none;
      background-image:url("data:image/svg+xml;base64,PHN2ZyB4bWxucz0iaHR0cDovL3d3dy53My5vcmcvMjAwMC9zdmciIHdpZHRoPSIxNDAiIGhlaWdodD0iMjYwIiB2aWV3Qm94PSIwIDAgMTQwIDI2MCI+CjxnIGZpbGw9Im5vbmUiIHN0cm9rZS13aWR0aD0iMyIgc3Ryb2tlLWxpbmVjYXA9InJvdW5kIj4KPHBhdGggZD0iTTQwIDAgQzExMCA0NSAxMTAgODUgNDAgMTMwIEMtMzAgMTc1IC0zMCAyMTUgNDAgMjYwIiBzdHJva2U9IiMxNGI4YTYiLz4KPHBhdGggZD0iTTEwMCAwIEMzMCA0NSAzMCA4NSAxMDAgMTMwIEMxNzAgMTc1IDE3MCAyMTUgMTAwIDI2MCIgc3Ryb2tlPSIjNjM2NmYxIi8+CjwvZz4KPGcgc3Ryb2tlPSIjNWVlYWQ0IiBzdHJva2Utd2lkdGg9IjIiIG9wYWNpdHk9IjAuOSI+CjxsaW5lIHgxPSI1MiIgeTE9IjE0IiB4Mj0iODgiIHkyPSIxNCIvPgo8bGluZSB4MT0iNjYiIHkxPSIzMCIgeDI9IjkyIiB5Mj0iMzAiLz4KPGxpbmUgeDE9Ijc0IiB5MT0iNDgiIHgyPSI5MiIgeTI9IjQ4Ii8+CjxsaW5lIHgxPSI3MCIgeTE9Ijk2IiB4Mj0iODgiIHkyPSI5NiIvPgo8bGluZSB4MT0iNTIiIHkxPSIxMTYiIHgyPSI4OCIgeTI9IjExNiIvPgo8bGluZSB4MT0iNTIiIHkxPSIxNDQiIHgyPSI4OCIgeTI9IjE0NCIvPgo8bGluZSB4MT0iNDgiIHkxPSIxNjQiIHgyPSI2NiIgeTI9IjE2NCIvPgo8bGluZSB4MT0iNDgiIHkxPSIyMTIiIHgyPSI2NiIgeTI9IjIxMiIvPgo8bGluZSB4MT0iNTIiIHkxPSIyMzAiIHgyPSI4OCIgeTI9IjIzMCIvPgo8bGluZSB4MT0iNTIiIHkxPSIyNDYiIHgyPSI4OCIgeTI9IjI0NiIvPgo8L2c+PC9zdmc+");
      background-size:150px auto; opacity:.06;
      animation:dnaflow 24s linear infinite;
    }
    @keyframes dnaflow{ to{ background-position:0 -1040px; } }
    @media (prefers-reduced-motion: reduce){ .stApp::before{ animation:none; } }
    h1,h2,h3,.hdr-title{font-family:'Plus Jakarta Sans',sans-serif!important; letter-spacing:-.02em;}

    /* ---- top header bar ---- */
    .hdr{
      display:flex; align-items:center; gap:14px; padding:16px 20px; margin-bottom:14px;
      background:linear-gradient(120deg, rgba(20,184,166,.12), rgba(99,102,241,.12));
      border:1px solid var(--line); border-radius:18px;
      box-shadow:0 12px 40px -20px rgba(0,0,0,.7);
    }
    .hdr-logo{
      width:46px;height:46px;border-radius:13px;display:grid;place-items:center;
      background:linear-gradient(135deg,var(--brand),var(--brand2));
      font-size:24px; box-shadow:0 8px 20px -6px rgba(20,184,166,.55);
    }
    .hdr-title{font-size:1.5rem;font-weight:800;line-height:1.1;color:#fff;}
    .hdr-sub{color:var(--dim);font-size:.86rem;margin-top:2px;}
    .hdr-chip{
      margin-left:auto; padding:.35rem .7rem;border-radius:999px;font-size:.72rem;
      font-weight:700;color:#052e2b;background:var(--brand);white-space:nowrap;
    }
    .disclaimer{
      display:flex;gap:10px;align-items:center;padding:10px 14px;border-radius:12px;
      background:rgba(251,191,36,.08);border:1px solid rgba(251,191,36,.28);
      color:#fde68a;font-size:.82rem;margin-bottom:6px;
    }

    /* ---- chat bubbles ---- */
    [data-testid="stChatMessage"]{
      background:var(--panel); border:1px solid var(--line);
      border-radius:16px; padding:.5rem 1rem; margin-bottom:.5rem;
      box-shadow:0 10px 34px -22px rgba(0,0,0,.8);
    }
    [data-testid="stChatMessage"]:has([data-testid="stChatMessageAvatarUser"]){
      background:linear-gradient(180deg,var(--panel2),var(--panel));
      border-color:rgba(99,102,241,.28);
    }

    /* ---- sidebar ---- */
    section[data-testid="stSidebar"]{
      background:linear-gradient(180deg,#0d1428,#0b1020); border-right:1px solid var(--line);
    }
    section[data-testid="stSidebar"] h3{
      font-size:.78rem!important;text-transform:uppercase;letter-spacing:.08em;
      color:var(--dim)!important;margin:.4rem 0 .3rem;
    }

    /* ---- pills / badges ---- */
    .pill{
      display:inline-block; padding:.2rem .6rem; border-radius:999px; font-size:.7rem;
      border:1px solid var(--line); color:var(--dim); margin:.12rem .2rem 0 0; font-weight:600;
    }
    .pill.on{color:#052e2b; background:var(--brand); border-color:var(--brand);}
    .metric-card{
      background:var(--panel); border:1px solid var(--line); border-radius:12px;
      padding:.6rem .8rem; margin-bottom:.5rem;
    }
    .metric-card b{color:var(--brand);}

    /* ---- inputs / buttons ---- */
    .stChatInput textarea{background:var(--panel)!important;color:var(--ink)!important;
      border-radius:12px!important;}
    .stButton>button{
      border-radius:11px; border:1px solid var(--line); background:var(--panel);
      color:var(--ink); font-weight:600; transition:all .15s;
    }
    .stButton>button:hover{border-color:var(--brand); color:var(--brand);
      box-shadow:0 6px 18px -10px rgba(20,184,166,.7);}
    .stImage img{border-radius:10px;border:1px solid var(--line);}
    a{color:var(--brand)!important;}
    </style>
    """,
    unsafe_allow_html=True,
)

st.markdown(
    '<div class="hdr">'
    '<div class="hdr-logo">🩺</div>'
    '<div><div class="hdr-title">MultiMedAI</div>'
    '<div class="hdr-sub">Medical image search · vision analysis · document intelligence '
    '— offline, open-source, CPU</div></div>'
    '<div class="hdr-chip">EDUCATION / RESEARCH</div>'
    '</div>',
    unsafe_allow_html=True,
)
st.markdown(
    '<div class="disclaimer">⚕️ <span><b>Educational / research use only — not a medical '
    'device.</b> Outputs are not a diagnosis or treatment advice. Always consult a '
    'qualified clinician.</span></div>',
    unsafe_allow_html=True,
)


# ============================ cached model loaders =========================
@st.cache_resource(show_spinner=False)
def _clip():
    from src.biomedclip import load_biomedclip
    return load_biomedclip(cfg["retrieval"]["clip_model"])


@st.cache_resource(show_spinner=False)
def _bank():
    from src.retrieval import load_bank
    return load_bank(cfg)


@st.cache_resource(show_spinner=False)
def _vqa():
    import json, torch
    from src.vqa import _head, _paths
    head_file, vocab_file = _paths(cfg)
    vocab = json.loads(vocab_file.read_text(encoding="utf-8"))
    head = _head(1024, len(vocab)); head.load_state_dict(torch.load(head_file)); head.eval()
    return head, vocab


# ============================ capability handlers ==========================
# --- provenance: which open dataset each bank image came from (traceable) ---
SOURCE_INFO = {
    "radiology": ("ROCOv2 Radiology (open)",
                  "https://huggingface.co/datasets/eltorio/ROCOv2-radiology"),
    "pathology": ("PathVQA (open)",
                  "https://huggingface.co/datasets/flaviagiammarino/path-vqa"),
    "tb": ("TB Chest X-ray (open)",
           "https://huggingface.co/datasets/DevVoyageR007/classify_Pneumonia_Tuberculosis_and_Normal__Non_Xray_chest_Xray_images"),
}


def provenance(meta):
    """Return (source_name, source_url) for a bank image — real, citable origin."""
    src = meta.get("source", "pathology")
    return SOURCE_INFO.get(src, SOURCE_INFO["pathology"])


def do_retrieve(query, topk):
    import torch
    from src.biomedclip import encode_texts
    model, preprocess, tokenizer, device = _clip()
    bank, metas = _bank()
    # Use the RAW query — do NOT force "histopathology", which biased every
    # search toward pathology and broke radiology (chest/brain) queries.
    q = encode_texts(model, tokenizer, device, [query])
    sims = (q @ bank.T).squeeze(0)
    vals, idx = sims.topk(min(topk, len(metas)))
    bank_dir = resolve(cfg["paths"]["data"], "image_bank")
    # return the FULL meta so the UI can show caption + source provenance
    return [(bank_dir / metas[i]["file"], metas[i], float(vals[k]))
            for k, i in enumerate(idx.tolist())]


def do_vqa(pil_img, question):
    import torch
    from src.biomedclip import encode_images, encode_texts
    model, preprocess, tokenizer, device = _clip()
    head, vocab = _vqa()
    ie = encode_images(model, preprocess, device, [pil_img])
    qe = encode_texts(model, tokenizer, device, [question])
    probs = head(torch.cat([ie, qe], dim=1)).softmax(1)[0]
    top = probs.topk(3)
    return [(vocab[i], float(top.values[k])) for k, i in enumerate(top.indices.tolist())]


def do_report(pil_img):
    from src.report import caption_image
    return caption_image(cfg, pil_img)


# candidate modalities for the zero-shot "what is this / is it a brain scan?" check
MODALITIES = [
    "a brain MRI or CT scan",
    "a chest X-ray radiograph",
    "an abdominal or pelvic CT scan",
    "a histopathology microscope slide (H&E stained tissue)",
    "a retinal fundus photograph",
    "a bone or limb X-ray",
    "an ultrasound image",
]


def neighbors_for_image(pil_img, k=3):
    """RAG grounding: find the k most similar REAL bank images and return their
    real captions. Used to ground reports so the LLM summarizes REAL context
    instead of inventing findings."""
    import torch
    from src.biomedclip import encode_images
    model, preprocess, tokenizer, device = _clip()
    bank, metas = _bank()
    ie = encode_images(model, preprocess, device, [pil_img])
    sims = (ie @ bank.T).squeeze(0)
    vals, idx = sims.topk(min(k, len(metas)))
    return [(metas[i].get("question", ""), float(vals[j]))
            for j, i in enumerate(idx.tolist())]


# modality -> candidate FINDINGS (zero-shot labels). Keyed by what the modality
# check detects, so findings are relevant to the image type (accurate across
# brain / chest / pathology / abdomen / bone / retina).
FINDINGS = {
    "brain": ["a brain tumor or mass", "an intracranial hemorrhage",
              "an ischemic stroke / infarct", "white matter lesions",
              "hydrocephalus (enlarged ventricles)", "a normal brain"],
    "chest": ["pulmonary tuberculosis", "pneumonia", "COVID-19 pneumonia",
              "a lung mass or nodule", "pleural effusion", "cardiomegaly",
              "a normal chest"],
    "abdom": ["a mass or tumor", "bowel obstruction", "free fluid / ascites",
              "an abscess or collection", "normal abdomen"],
    "patho": ["carcinoma / malignant tumor tissue", "inflammation",
              "necrosis", "benign tissue", "normal tissue"],
    "bone": ["a fracture", "arthritis / joint degeneration", "a bone lesion",
             "normal bone"],
    "retina": ["diabetic retinopathy", "glaucoma", "macular degeneration",
               "a normal retina"],
    "ultra": ["a cyst", "a mass", "a stone / calculus", "normal ultrasound"],
}


def _modality_key(modality: str) -> str:
    m = modality.lower()
    if "brain" in m: return "brain"
    if "chest" in m: return "chest"
    if "abdom" in m or "pelvic" in m: return "abdom"
    if "histopath" in m or "slide" in m: return "patho"
    if "bone" in m or "limb" in m: return "bone"
    if "retina" in m or "fundus" in m: return "retina"
    if "ultrasound" in m: return "ultra"
    return "patho"


def do_modality_check(pil_img):
    """Zero-shot: what kind of medical image is this? (BiomedCLIP image-vs-text)."""
    import torch
    from src.biomedclip import encode_images, encode_texts
    model, preprocess, tokenizer, device = _clip()
    ie = encode_images(model, preprocess, device, [pil_img])
    te = encode_texts(model, tokenizer, device, MODALITIES)
    probs = (ie @ te.T)[0].softmax(0)
    k = int(probs.argmax())
    return MODALITIES[k], float(probs[k])


def do_finding_check(pil_img, modality, topn=3):
    """Finding suggestion. Uses a SUPERVISED head (accurate, e.g. chest ~94%) if
    trained for this modality; else honest zero-shot (~chance). Returns
    (list[(finding, prob)], reliable: bool)."""
    from src.biomedclip import encode_images, encode_texts
    from src import findings
    model, preprocess, tokenizer, device = _clip()
    ie = encode_images(model, preprocess, device, [pil_img])
    key = _modality_key(modality)
    if findings.available(key):
        preds = findings.predict(key, ie, topn=topn)   # trained → reliable
        return preds, True
    labels = FINDINGS[key]                              # zero-shot fallback
    te = encode_texts(model, tokenizer, device,
                      [f"this medical image shows {l}" for l in labels])
    probs = (ie @ te.T)[0].softmax(0)
    order = probs.argsort(descending=True)[:topn].tolist()
    return [(labels[i], float(probs[i])) for i in order], False


def do_generate(prompt):
    import torch
    from diffusers import AutoPipelineForText2Image
    scfg = cfg["synthesis"]
    weights = resolve(cfg["paths"]["weights"], scfg.get("weights_subdir", "sdturbo"))
    pipe = AutoPipelineForText2Image.from_pretrained(
        scfg["model_id"], torch_dtype=torch.float32, safety_checker=None,
        cache_dir=str(weights)).to(DEVICE)
    lp = engine.lora_path(cfg)
    if lp:
        pipe.load_lora_weights(str(lp.parent))
    # realism steering: append a clinical-photo suffix + use a negative prompt so
    # the image follows the request and isn't hyper-stylized
    full = f"{prompt}, {scfg.get('style_suffix', '')}".strip(", ")
    kw = dict(num_inference_steps=scfg["num_inference_steps"],
              guidance_scale=scfg["guidance_scale"],
              height=scfg["image_size"], width=scfg["image_size"])
    if scfg.get("guidance_scale", 0) > 1 and scfg.get("negative_prompt"):
        kw["negative_prompt"] = scfg["negative_prompt"]
    return pipe(full, **kw).images[0]


# ============================ sidebar ======================================
with st.sidebar:
    st.markdown("### ⚙️ Session")
    st.caption(f"Device: **{DEVICE.upper()}** · open models + PathVQA")

    up = st.file_uploader("Upload an image (for VQA / report)", type=["png", "jpg", "jpeg"])
    uploaded_img = Image.open(up).convert("RGB") if up else None

    # --- grab an image from the last retrieved results as the active image ---
    picked_img = None
    last = st.session_state.get("last_results", [])
    if last:
        st.markdown("**Or analyze a result image:**")
        labels = [f"#{i+1} · {m.get('question','')[:28]}" for i, (_, m, _) in enumerate(last)]
        choice = st.selectbox("Pick from last results", ["(none)"] + labels, index=0)
        if choice != "(none)":
            sel = last[labels.index(choice)]
            from PIL import Image as _I
            picked_img = _I.open(str(sel[0])).convert("RGB")

    # active image = uploaded (priority) else the picked result
    user_img = uploaded_img or picked_img
    if user_img:
        st.image(user_img, caption="Active image (ask about it)", use_column_width=True)

    topk = st.slider("Images to retrieve", 3, 15, 9)
    allow_gen = st.toggle("Allow image generation (slow, ~50s)", value=False)

    # --- document analysis (thesis / PDF) ---
    st.markdown("### 📄 Document analysis")
    docup = st.file_uploader("Upload a thesis / PDF / TXT", type=["pdf", "txt"], key="docup")
    if docup is not None:
        import hashlib
        sig = hashlib.md5(docup.getvalue()).hexdigest()
        if st.session_state.get("doc_sig") != sig:
            try:
                with st.spinner("Ingesting document (extract → chunk → embed)…"):
                    from src import docstore
                    doc_id, nchunks = docstore.ingest(docup.getvalue(), docup.name)
                st.session_state["active_doc_id"] = doc_id
                st.session_state["active_doc_name"] = docup.name
                st.session_state["doc_sig"] = sig
                st.session_state["_last_doc_bytes"] = (docup.getvalue(), docup.name)
                st.success(f"Ingested “{docup.name}” ({nchunks} chunks). "
                           "Ask about it in the chat.")
            except Exception as e:
                st.error(f"Could not ingest: {e}")
    if st.session_state.get("active_doc_id"):
        st.caption(f"📄 Active document: **{st.session_state.get('active_doc_name','')}** "
                   "— your questions are answered from it (with page citations).")
        c1, c2 = st.columns(2)
        with c1:
            if st.button("Clear document"):
                for k in ("active_doc_id", "active_doc_name", "doc_sig"):
                    st.session_state.pop(k, None)
                st.rerun()
        with c2:
            # doctor-curation: permanently add this doc to the knowledge base
            if st.button("➕ Add to KB", help="Add this document to the permanent "
                         "knowledge base so future questions can cite it"):
                try:
                    from src import docstore
                    docv = st.session_state.get("_last_doc_bytes")
                    if docv:
                        with st.spinner("Adding to knowledge base…"):
                            n = docstore.add_document_to_kb(docv[0], docv[1])
                        st.success(f"Added {n} passages to the KB.")
                except Exception as e:
                    st.error(f"Could not add: {e}")

    # knowledge-base status (standing corpus + curated docs)
    try:
        from src import docstore as _ds
        _kbn = _ds.kb_count()
        if _kbn:
            st.caption(f"📚 Knowledge base: **{_kbn:,}** passages "
                       "(open literature + curated).")
    except Exception:
        pass

    st.markdown("### Capabilities")
    on = lambda b: "on" if b else "off"
    st.markdown(
        f'<span class="pill {on(engine.bank_available(cfg))}">retrieval</span>'
        f'<span class="pill {on(engine.vqa_available(cfg))}">VQA head</span>'
        f'<span class="pill on">report</span>'
        f'<span class="pill on">synthesis</span>'
        f'<span class="pill {on(engine.lora_path(cfg) is not None)}">LoRA</span>',
        unsafe_allow_html=True,
    )

    st.markdown("### 📊 Real metrics")
    for name, path in [
        ("Retrieval", resolve(cfg["paths"]["outputs"], "retrieval", "recall.txt")),
        ("VQA", resolve(cfg["paths"]["outputs"], "vqa", "accuracy.txt")),
        ("Report", resolve(cfg["paths"]["outputs"], "report", "metrics.txt")),
        ("Synthesis", resolve(cfg["paths"]["outputs"], "synthesis", "fid_metric.txt")),
    ]:
        if Path(path).exists():
            txt = Path(path).read_text(encoding="utf-8").strip().replace("\n", " · ")
            st.markdown(f'<div class="metric-card"><b>{name}</b><br><small>{txt}</small></div>',
                        unsafe_allow_html=True)


# ============================ chat state ===================================
if "messages" not in st.session_state:
    st.session_state.messages = [{
        "role": "assistant",
        "content": "👋 **Welcome to MultiMedAI.** I can help you:\n\n"
                   "• 🖼️ **Find real images** — *“show chest x-rays of tuberculosis”*\n"
                   "• 🔬 **Analyze an image** — upload/grab one, then *“analyse this”* "
                   "(detailed vision-model narration + trained findings)\n"
                   "• 📄 **Study a document** — upload a thesis/PDF, then *“generate a report”* "
                   "or *“what dosage of X?”* (grounded, page-cited)\n"
                   "• 📚 **Explain concepts** — *“what is a glioma?”*\n\n"
                   "_Educational & research use — not a diagnosis._",
    }]

for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])
        for img, cap in msg.get("images", []):
            try:
                st.image(Image.open(img).convert("RGB"), caption=cap, width=200)
            except Exception:
                pass  # skip unreadable/placeholder entries

prompt = st.chat_input("Ask, search images, analyze an upload, or query a document…")
if prompt:
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    intent = engine.detect_intent(prompt, user_img is not None)
    min_score = cfg["retrieval"].get("min_score", 0.0)
    with st.chat_message("assistant"):
        imgs_payload = []
        if engine.is_medical_advice_request(prompt):
            text = ("🩺 I can't give a personal **diagnosis, treatment, or dosage "
                    "recommendation** — I'm an offline **education/research** tool, not "
                    "a clinician or an approved medical device. Please consult a "
                    "qualified healthcare professional.\n\nI *can* help with: "
                    "educational info, **real reference images** (*“show chest x-rays”*), "
                    "**describing an uploaded image** (not a diagnosis), or answering "
                    "from a **document** you upload.")
            st.markdown(text)
        elif intent == "retrieve":
            with st.spinner("Searching the image bank…"):
                hits = do_retrieve(prompt, topk)
            confident = [h for h in hits if h[2] >= min_score]
            if not confident:
                best = hits[0][2] if hits else 0.0
                text = (f"**No confident match** for *“{prompt}”* in the image bank "
                        f"(best similarity {best:.2f} < {min_score:.2f}). The bank may "
                        "not contain this diagnosis. I won't show misleading results "
                        "or generate a fake image.")
                st.markdown(text)
            else:
                text = (f"Here are the top **{len(confident)}** REAL matching images "
                        f"for *“{prompt}”* (similarity ≥ {min_score:.2f}):")
                st.markdown(text)
                cols = st.columns(3)
                for i, (path, meta, score) in enumerate(confident):
                    ctx = meta.get("question", "")
                    src_name, src_url = provenance(meta)
                    with cols[i % 3]:
                        st.image(Image.open(str(path)).convert("RGB"),
                                 use_column_width=True)
                        st.markdown(
                            f"**{score:.2f}** · {ctx[:70]}<br>"
                            f"<small>📖 Source: <a href='{src_url}'>{src_name}</a></small>",
                            unsafe_allow_html=True)
                    imgs_payload.append((str(path), f"{score:.2f} · {ctx[:40]}"))
                # remember results so the user can grab one to analyze (sidebar)
                st.session_state["last_results"] = confident
                st.caption("REAL images from open datasets — not generated. Score = "
                           "similarity. To analyze one, pick it in the sidebar "
                           "(“analyze a result image”) and ask, e.g. *“what is this?”*")

        elif intent == "ask" and st.session_state.get("active_doc_id"):
            # DOCUMENT RAG: grounded Q&A OR a structured report, both cited.
            from src import docstore
            did = st.session_state["active_doc_id"]
            if engine.is_report_request(prompt):
                with st.spinner("Generating a structured report from the document…"):
                    aspects = ["objective and aim of the study",
                               "methods and study design",
                               "key results and findings",
                               "metrics measurements dosages values",
                               "limitations and conclusions"]
                    seen, hits = set(), []
                    for a in aspects:
                        for h in docstore.search(did, a, k=2):
                            if h["chunk_id"] not in seen:
                                seen.add(h["chunk_id"]); hits.append(h)
                    try:
                        answer = llm.generate_document_report(cfg, hits)
                    except Exception:
                        answer = "(LLM unavailable — try again in a moment.)"
                st.markdown(answer)
            else:
                with st.spinner("Searching the document…"):
                    hits = docstore.search(did, prompt, k=5)
                    try:
                        answer = llm.answer_document(cfg, prompt, hits)
                    except Exception:
                        answer = "(LLM unavailable — try again in a moment.)"
                st.markdown(f"**{answer}**")
            if hits:
                with st.expander("📑 Source passages (grounding — verify here)"):
                    for h in hits:
                        st.markdown(f"**p.{h['page']}** · sim {h['score']:.2f}\n\n"
                                    f"{h['text'][:320]}…")
            st.caption(f"Grounded in **{st.session_state.get('active_doc_name','')}** — "
                       "cites pages, refuses if not in the document. Educational, not advice.")
            text = answer

        elif intent == "ask":
            pl = prompt.lower()
            recency = any(w in pl for w in ("recent", "latest", "current", "today",
                          "this year", "2024", "2025", "news", "up to date", "update",
                          "nowadays"))
            wants_report = any(w in pl for w in ("report", "overview", "review",
                               "state of", "summary of", "guidelines"))
            if recency or wants_report:
                # CROSS-QUESTION instead of guessing — and be honest about no live data
                topic = prompt.strip().rstrip("?.")
                text = (
                    "Quick check so I give you the *right* thing 👇\n\n"
                    "⚠️ I run **fully offline on free local models**, so I **can't pull "
                    "recent/live web reports** — my text knowledge is static (and may be "
                    f"dated). For *“{topic}”*, do you want:\n\n"
                    "1. 🖼️ **Real images** from the bank — e.g. *“show covid chest x-rays”*\n"
                    "2. 📄 **Analysis of a document** — upload a recent paper/thesis in the "
                    "sidebar, then ask (I'll answer with page citations)\n"
                    "3. 📚 **General background** — e.g. *“explain covid”* (may be dated)\n\n"
                    "Which one?")
                st.markdown(text)
            else:
                # 1) try the standing KNOWLEDGE BASE (open literature) for grounding
                from src import docstore
                with st.spinner("Searching the medical knowledge base…"):
                    kb_hits = docstore.search_kb(prompt, k=4, min_score=0.30)
                if kb_hits:
                    with st.spinner("Composing a grounded answer…"):
                        try:
                            answer = llm.answer_document(cfg, prompt, kb_hits)
                        except Exception:
                            answer = kb_hits[0]["text"][:400]
                    text = f"**Answer:** {answer}"
                    st.markdown(text)
                    with st.expander(f"📚 Sources ({len(kb_hits)} passages from the knowledge base)"):
                        for h in kb_hits:
                            st.markdown(f"**[{h['page']}] {h['source']}** · sim {h['score']:.2f}  \n"
                                        f"*{h['title']}*")
                    st.caption("Grounded in the medical knowledge base (open PubMed "
                               "literature + curated docs). Educational, not a diagnosis.")
                else:
                    # 2) fallback: ungrounded LLM background (clearly flagged)
                    with st.spinner("Answering…"):
                        try:
                            answer = llm.answer_question(cfg, prompt)
                        except Exception:
                            answer = "(LLM unavailable — try again in a moment.)"
                    text = f"**Answer:** {answer}"
                    st.markdown(text)
                    st.caption("⚠️ No close match in the knowledge base — this is the "
                               "LLM's general (offline, possibly dated) knowledge, not "
                               "grounded in a source. Not a diagnosis.")

        elif intent == "locate":
            # circle + label a region ("where is the tumor?") via Grad-CAM attention
            target = engine.extract_target(prompt)
            with st.spinner(f"Locating “{target}” (model-attention heatmap)…"):
                from src import annotate as _annot
                annotated, found = _annot.annotate(user_img, target)
                gen_path = resolve(cfg["paths"]["outputs"], "synthesis", "annotated.png")
                annotated.save(gen_path)
            if found:
                st.image(annotated, caption=f"Attention heatmap for “{target}”", width=340)
                text = (f"Highlighted where the model most associates **“{target}”** "
                        "(red heatmap + circle on the peak region).")
                imgs_payload.append((str(gen_path), f"attention: {target}"))
            else:
                text = (f"I couldn't compute a reliable attention map for “{target}” "
                        "on this image.")
            st.markdown(text)
            st.caption("⚠️ This is a **model-attention (Grad-CAM) heatmap** showing where "
                       "BiomedCLIP looks for that concept — it is **indicative only, NOT a "
                       "validated lesion detector, segmentation, or diagnosis.**")

        elif intent in ("vqa", "report"):
            # analyze the ACTIVE image (uploaded OR grabbed from results)
            with st.spinner("Analyzing the image with the vision model… (~1 min on CPU)"):
                from src import vlm
                # VLM SEES the image: narrate, or answer the user's specific question
                q = prompt if intent == "vqa" else None
                try:
                    body = vlm.describe(user_img, q)
                except Exception as e:
                    body = f"(Vision model unavailable: {e})"
                modality, mconf = do_modality_check(user_img)
                findings, supervised = do_finding_check(user_img, modality)
                neighbors = neighbors_for_image(user_img, k=4)     # RAG grounding
                neigh_caps = [c for c, s in neighbors if s >= min_score]
                uniform = 1.0 / len(FINDINGS[_modality_key(modality)])
                reliable = supervised or findings[0][1] >= uniform * 1.5
            text = f"**{body}**\n\n🔬 _Image type:_ **{modality}** ({mconf:.0%})\n\n"
            if supervised:
                fs = " · ".join(f"{f} ({p:.0%})" for f, p in findings)
                text += (f"🩻 _Finding (trained classifier, ~94% val acc):_ "
                         f"**{findings[0][0]}** ({findings[0][1]:.0%}) · others: {fs}\n\n")
            elif reliable:
                text += (f"🩻 _Possible category (uncertain, zero-shot):_ "
                         f"**{findings[0][0]}** ({findings[0][1]:.0%}) — not confirmed.\n\n")
            else:
                text += ("🩻 _Automated finding detection **inconclusive** (near-chance "
                         "for this modality — no trained classifier yet). See real cases.*\n\n")
            if neigh_caps:
                def _wt(s):  # trim at a word boundary so captions don't cut mid-word
                    s = s.strip()
                    return s if len(s) <= 96 else s[:96].rsplit(" ", 1)[0] + "…"
                text += "📚 _Most similar REAL cases in the library:_\n" + "\n".join(
                    f"- “{_wt(c)}”" for c in neigh_caps[:3])
            text += ("\n\n_Narration by a general-domain vision model (moondream) — detailed "
                     "but **not medically validated** and may err. The trained finding "
                     "(where shown) and cited real cases are the reliable signals. "
                     "**Educational, not a diagnosis.**_")
            # honest modality-mismatch flag (e.g. asked about brain, image is chest)
            pl = prompt.lower()
            if "brain" in pl and "brain" not in modality.lower():
                text += ("\n\n⚠️ **Note:** you mentioned *brain*, but this image looks "
                         f"like **{modality}** — it may **not** be a brain scan.")
            text += "\n\n_Educational analysis, not a diagnosis._"
            st.markdown(text)

        elif intent == "generate":
            if not allow_gen:
                text = "Image generation is **off**. Enable the toggle in the sidebar (it's slow on CPU, ~50s)."
                st.markdown(text)
            else:
                with st.spinner("Generating on CPU (~50s)…"):
                    img = do_generate(prompt)
                st.image(img, caption=prompt, width=320)
                gen_path = resolve(cfg["paths"]["outputs"], "synthesis", "chat_gen.png")
                img.save(gen_path)
                text = f"Generated an image for *“{prompt}”* (SD v1.5, CPU)."
                st.markdown(text)
                imgs_payload.append((str(gen_path), prompt))

        st.session_state.messages.append(
            {"role": "assistant", "content": text, "images": imgs_payload})
