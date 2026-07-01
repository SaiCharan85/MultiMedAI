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
    return [(bank_dir / metas[i]["file"], metas[i]["question"], float(vals[k]))
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
    user_img = Image.open(up).convert("RGB") if up else None
    if user_img:
        st.image(user_img, caption="Active image", use_column_width=True)

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
                for i, (path, ctx, score) in enumerate(confident):
                    cap = f"{score:.2f} · {ctx[:40]}"
                    with cols[i % 3]:
                        st.image(Image.open(str(path)).convert("RGB"),
                                 caption=cap, use_column_width=True)
                    imgs_payload.append((str(path), cap))
                st.caption("REAL images retrieved from PathVQA — not generated, "
                           "no hallucination. Score = cosine similarity (confidence).")

        elif intent == "ask":
            # a general QUESTION (no uploaded image) -> concise TEXT answer,
            # plus a few illustrative REAL images if the bank has confident ones.
            with st.spinner("Answering…"):
                try:
                    answer = llm.answer_question(cfg, prompt)
                except Exception:
                    answer = "(LLM unavailable — try again in a moment.)"
                hits = do_retrieve(prompt, 3)
                illustrative = [h for h in hits if h[2] >= min_score]
            st.markdown(f"**Answer:** {answer}")
            if illustrative:
                st.caption("Related real images from the bank:")
                cols = st.columns(3)
                for i, (path, ctx, score) in enumerate(illustrative):
                    cap = f"{score:.2f} · {ctx[:32]}"
                    with cols[i % 3]:
                        st.image(Image.open(str(path)).convert("RGB"),
                                 caption=cap, use_column_width=True)
                    imgs_payload.append((str(path), cap))
            st.caption("General explanation by the LLM (Qwen2.5) — educational, "
                       "not a diagnosis. Images (if any) are real, retrieved.")
            text = answer

        elif intent == "vqa":
            if not engine.vqa_available(cfg):
                text = ("No trained VQA head found yet. Train it with "
                        "`python -m src.vqa all`, then re-ask.")
                st.markdown(text)
            else:
                with st.spinner("Analyzing image + question…"):
                    ans = do_vqa(user_img, prompt)
                    cap = do_report(user_img)
                    try:
                        natural = llm.compose_answer(cfg, prompt, ans[0][0], cap)
                    except Exception:
                        natural = ans[0][0]
                text = (f"**Answer:** {natural}\n\n"
                        f"_Model prediction:_ **{ans[0][0]}** ({ans[0][1]:.0%}); others: "
                        + ", ".join(f"{a} ({p:.0%})" for a, p in ans[1:])
                        + f"\n\n_Visual description:_ {cap}")
                st.markdown(text)

        elif intent == "report":
            with st.spinner("Writing a report…"):
                cap = do_report(user_img)
                try:
                    report = llm.compose_report(cfg, cap)
                except Exception:
                    report = cap
            text = (f"**Report:** {report}\n\n"
                    f"_Grounded on visual description:_ {cap}\n\n"
                    "_LLM (Qwen2.5) rephrases grounded observations only; not a diagnosis._")
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
