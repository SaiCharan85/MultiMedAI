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
    @import url('https://fonts.googleapis.com/css2?family=Sora:wght@400;600;700;800&family=Inter:wght@400;500;600&display=swap');

    :root{
      --bg:#0a0e1a; --panel:#111726; --accent:#2dd4bf; --accent2:#a78bfa;
      --text:#e6edf6; --dim:#8492ad; --border:rgba(148,163,184,.14);
    }
    .stApp{
      background:
        radial-gradient(1100px 520px at 12% -8%, rgba(45,212,191,.16), transparent 60%),
        radial-gradient(1000px 520px at 92% 4%, rgba(167,139,250,.16), transparent 60%),
        var(--bg);
      color:var(--text);
    }
    /* hide default chrome for a cleaner app feel */
    #MainMenu, header, footer{visibility:hidden;}
    .block-container{padding-top:2.2rem; max-width:1050px;}

    h1,h2,h3,.brand{font-family:'Sora',sans-serif!important; letter-spacing:-.02em;}
    body, p, div, span, label{font-family:'Inter',sans-serif;}

    .brand{
      font-size:2.1rem; font-weight:800; margin-bottom:.1rem;
      background:linear-gradient(100deg,#fff,#2dd4bf 55%,#a78bfa);
      -webkit-background-clip:text; background-clip:text; -webkit-text-fill-color:transparent;
    }
    .tagline{color:var(--dim); margin-bottom:1.2rem; font-size:.95rem;}

    /* chat bubbles */
    [data-testid="stChatMessage"]{
      background:var(--panel); border:1px solid var(--border);
      border-radius:16px; padding:.4rem .9rem; box-shadow:0 8px 30px -16px rgba(0,0,0,.6);
    }
    /* sidebar */
    section[data-testid="stSidebar"]{
      background:linear-gradient(180deg,#0c1120,#0a0e1a); border-right:1px solid var(--border);
    }
    .pill{
      display:inline-block; padding:.18rem .6rem; border-radius:999px; font-size:.72rem;
      border:1px solid var(--border); color:var(--dim); margin:.1rem .2rem 0 0;
    }
    .pill.on{color:#0a0e1a; background:var(--accent); border-color:var(--accent); font-weight:700;}
    .pill.off{color:var(--dim);}
    .metric-card{
      background:var(--panel); border:1px solid var(--border); border-radius:14px;
      padding:.7rem .9rem; margin-bottom:.6rem;
    }
    .metric-card b{color:var(--accent);}
    .stChatInput textarea{background:var(--panel)!important; color:var(--text)!important;}
    .stButton>button{
      border-radius:12px; border:1px solid var(--border); background:var(--panel); color:var(--text);
    }
    .stButton>button:hover{border-color:var(--accent); color:var(--accent);}
    </style>
    """,
    unsafe_allow_html=True,
)

st.markdown('<div class="brand">🧬 MultiMedAI</div>', unsafe_allow_html=True)
st.markdown(
    '<div class="tagline">A local pathology assistant — ask for images, ask about an '
    "image, request a report, or generate one. CPU-only, open models &amp; data.</div>",
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


def do_modality_check(pil_img):
    """Zero-shot: what kind of medical image is this? (BiomedCLIP image-vs-text).
    Lets the app say e.g. 'this looks like a chest X-ray, NOT a brain scan.'"""
    import torch
    from src.biomedclip import encode_images, encode_texts
    model, preprocess, tokenizer, device = _clip()
    ie = encode_images(model, preprocess, device, [pil_img])
    te = encode_texts(model, tokenizer, device, MODALITIES)
    sims = (ie @ te.T)[0]
    probs = sims.softmax(0)
    k = int(probs.argmax())
    return MODALITIES[k], float(probs[k])


def do_generate(prompt):
    import torch
    from diffusers import StableDiffusionPipeline
    scfg = cfg["synthesis"]
    weights = resolve(cfg["paths"]["weights"], "sd15")
    pipe = StableDiffusionPipeline.from_pretrained(
        scfg["model_id"], torch_dtype=torch.float32, safety_checker=None, cache_dir=str(weights)
    ).to(DEVICE)
    lp = engine.lora_path(cfg)
    if lp:
        pipe.load_lora_weights(str(lp.parent))
    return pipe(prompt, num_inference_steps=scfg["num_inference_steps"],
                guidance_scale=scfg["guidance_scale"],
                height=scfg["image_size"], width=scfg["image_size"]).images[0]


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
        "content": "Hi! Ask me to **show pathology images** (e.g. *“show me adenocarcinoma”*), "
                   "**upload an image** and ask a question, or request a **report**.",
    }]

for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])
        for img, cap in msg.get("images", []):
            try:
                st.image(Image.open(img).convert("RGB"), caption=cap, width=200)
            except Exception:
                pass  # skip unreadable/placeholder entries

prompt = st.chat_input("Ask about pathology images…")
if prompt:
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    intent = engine.detect_intent(prompt, user_img is not None)
    min_score = cfg["retrieval"].get("min_score", 0.0)
    with st.chat_message("assistant"):
        imgs_payload = []
        if intent == "retrieve":
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

        elif intent == "ask":
            # TEXT-ONLY answer (no images unless the user used display keywords).
            with st.spinner("Answering…"):
                try:
                    answer = llm.answer_question(cfg, prompt)
                except Exception:
                    answer = "(LLM unavailable — try again in a moment.)"
            st.markdown(f"**Answer:** {answer}")
            st.caption("Educational explanation by the LLM (Qwen2.5) — not a "
                       "diagnosis. Add words like *show / images / scans* to see images.")
            text = answer

        elif intent in ("vqa", "report"):
            # analyze the ACTIVE image (uploaded OR grabbed from results)
            with st.spinner("Analyzing the image…"):
                modality, mconf = do_modality_check(user_img)
                cap = do_report(user_img)
                vqa_line = ""
                if intent == "vqa" and engine.vqa_available(cfg):
                    ans = do_vqa(user_img, prompt)
                    vqa_line = ("_VQA head:_ **" + ans[0][0] + f"** ({ans[0][1]:.0%}); "
                                + ", ".join(f"{a} ({p:.0%})" for a, p in ans[1:]))
                # LLM grounds its explanation on modality + caption (+ VQA if any)
                try:
                    obs = f"Detected image type: {modality} (confidence {mconf:.0%}). {cap}"
                    if vqa_line:
                        obs += f" VQA prediction: {ans[0][0]}."
                    if intent == "report":
                        body = llm.compose_report(cfg, obs)
                    else:
                        body = llm.compose_answer(cfg, prompt, obs, cap)
                except Exception:
                    body = cap
            text = (f"**{body}**\n\n"
                    f"🔬 _Detected image type:_ **{modality}** ({mconf:.0%} confidence)\n\n"
                    f"_Visual description:_ {cap}"
                    + (f"\n\n{vqa_line}" if vqa_line else ""))
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
