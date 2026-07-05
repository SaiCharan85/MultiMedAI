"""LLM brain — free, efficient, CPU-runnable instruction model (Qwen2.5-Instruct).

Role: the assistant's language layer. It does NOT diagnose or invent facts; it
only rephrases GROUNDED inputs (the VQA head's answer, BLIP caption, retrieved
concepts) into a readable report or conversational reply. RAG-style: retrieval/
VQA supply the facts, the LLM supplies the prose.

Qwen2.5-1.5B-Instruct is Apache-2.0 (fully free) and decoder-only; we use its
chat template. All ops on CPU.
"""
from __future__ import annotations

import functools

import torch

from src.common import load_config, get_device


@functools.lru_cache(maxsize=1)
def _load(model_id):
    from transformers import AutoTokenizer, AutoModelForCausalLM

    device = get_device()
    tok = AutoTokenizer.from_pretrained(model_id)
    model = AutoModelForCausalLM.from_pretrained(
        model_id, torch_dtype=torch.float32
    ).eval().to(device)
    return tok, model, device


@torch.no_grad()
def chat(cfg, system, user, max_new_tokens=None):
    # Prefer the fast cloud model (Gemini) if a key is set; fall back to local
    # Qwen on any error (rate limit, offline, etc.).
    from src import cloudllm
    if cloudllm.available():
        try:
            return cloudllm.text(system, user,
                                 max_new_tokens or cfg["llm"]["max_new_tokens"])
        except Exception:
            pass
    tok, model, device = _load(cfg["llm"]["model_id"])
    mnt = max_new_tokens or cfg["llm"]["max_new_tokens"]
    messages = [{"role": "system", "content": system},
                {"role": "user", "content": user}]
    text = tok.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    ids = tok(text, return_tensors="pt").to(device)
    out = model.generate(**ids, max_new_tokens=mnt, do_sample=False,
                         repetition_penalty=1.1, pad_token_id=tok.eos_token_id)
    gen = out[0][ids["input_ids"].shape[1]:]
    return tok.decode(gen, skip_special_tokens=True).strip()


# Shared persona: a clinical assistant that writes for medical professionals with
# precise terminology, clean markdown formatting, and strict grounding.
SYSTEM = (
    "You are MultiMedAI, a clinical imaging assistant writing for medical students "
    "and clinicians. Use precise radiological/pathological terminology (e.g. signal "
    "intensity, attenuation, echogenicity, margins, enhancement pattern, morphology, "
    "distribution). Ground every statement ONLY in the observations provided — never "
    "invent findings, measurements, or diagnoses. Format cleanly in Markdown (short "
    "bold labels, bullet points). Frame conclusions as considerations, not diagnoses."
)


def compose_report(cfg, caption, vqa_answer=None, concepts=None):
    """Structured, technical imaging report grounded in the observations."""
    facts = [f"Vision-model description: {caption}"]
    if vqa_answer:
        facts.append(f"Classifier / predicted category: {vqa_answer}")
    if concepts:
        facts.append("Captions of the most similar real cases: " + "; ".join(concepts))
    user = (
        "From ONLY the observations below, write a structured **radiology/pathology "
        "report** using this exact template (Markdown, terse and technical):\n"
        "**Modality & View:** …\n**Findings:** … (describe location, size, "
        "morphology, margins, signal/density using technical descriptors)\n"
        "**Impression:** … (differential *considerations*, not a diagnosis)\n\n"
        "Do not invent anything not supported by the observations.\n\n"
        "Observations:\n" + "\n".join(f"- {f}" for f in facts)
    )
    return chat(cfg, SYSTEM, user, max_new_tokens=300)


def answer_question(cfg, question):
    """Clinical educational explanation of a concept (e.g. 'what is a glioma?').
    One-shot prompted for a consistent, well-structured, non-truncated answer."""
    system = (
        "You are MultiMedAI, a medical education assistant for students and "
        "clinicians. INSTRUCTIONS: (1) Use correct medical terminology and stay "
        "factual. (2) Follow the EXACT Markdown format shown in the examples: a "
        "one-line **Definition**, 2–4 concise bullets, then a one-line **Clinical "
        "relevance**. (3) Always finish your sentences — do not stop mid-word. "
        "(4) If unsure, say so. Educational only, not medical advice. Study the "
        "examples, then answer the final question in the same style."
    )
    # FEW-SHOT exemplars (2) to lock format, depth, and terminology
    fewshot = (
        "Q: What is a glioma?\n"
        "**Definition:** A glioma is a primary CNS tumour arising from glial cells "
        "(astrocytes, oligodendrocytes, or ependymal cells).\n"
        "- **Grading:** WHO grade I–IV; glioblastoma (IV) is the most aggressive.\n"
        "- **Imaging:** Typically T2/FLAIR-hyperintense with variable enhancement and "
        "surrounding vasogenic oedema.\n"
        "- **Presentation:** Headache, seizures, or focal neurological deficits.\n"
        "**Clinical relevance:** Grade drives prognosis and combined surgery / "
        "radiotherapy / chemotherapy (e.g. temozolomide).\n\n"
        "Q: What is a pneumothorax?\n"
        "**Definition:** A pneumothorax is air in the pleural space causing partial "
        "or complete lung collapse.\n"
        "- **Types:** Spontaneous, traumatic, or tension (a medical emergency).\n"
        "- **Imaging:** CXR shows a visceral pleural line with absent lung markings "
        "peripherally; tension causes mediastinal shift.\n"
        "- **Presentation:** Sudden pleuritic chest pain and dyspnoea.\n"
        "**Clinical relevance:** Tension pneumothorax needs immediate needle "
        "decompression; small ones may be observed."
    )
    user = f"{fewshot}\n\nQ: {question}"
    return chat(cfg, system, user, max_new_tokens=340)


def explain_image(cfg, modality, narration, finding=None, finding_conf=None,
                  evidence=None):
    """EXPLAINABILITY: synthesize a grounded, reasoned read of an image from the
    structured signals (modality, trained finding, vision narration, similar
    real cases) — WITHOUT dumping raw captions. Explains *why*, as considerations."""
    obs = [f"Detected modality: {modality}.",
           f"Vision-model narration: {narration}"]
    if finding:
        obs.append(f"Trained classifier suggests: {finding}"
                   + (f" (confidence {finding_conf:.0%})." if finding_conf else "."))
    if evidence:
        obs.append("Similar confirmed cases in the library describe: "
                   + "; ".join(evidence) + ".")
    system = (
        "You are MultiMedAI, a clinical imaging assistant that gives EXPLAINABLE "
        "reads. INSTRUCTIONS: (1) Base everything ONLY on the given observations — "
        "never invent findings. (2) Output EXACTLY this Markdown structure, as in the "
        "examples:\n"
        "**Assessment:** one sentence naming the most likely category as a "
        "*consideration* (not a diagnosis).\n"
        "**Why (evidence):** 2–3 bullets, each tying a SPECIFIC visual feature or "
        "signal (modality, trained finding, similar cases) to the assessment.\n"
        "**Caveats:** one bullet on what is needed to confirm.\n"
        "(3) Use precise radiological terminology and finish every sentence."
    )
    # FEW-SHOT exemplars showing the reasoning style (not raw captions)
    fewshot = (
        "Observations:\n- Detected modality: chest X-ray.\n- Trained classifier "
        "suggests: tuberculosis (confidence 88%).\n- Vision narration: upper-zone "
        "opacity with possible cavitation.\n"
        "→\n**Assessment:** Findings are most consistent with pulmonary tuberculosis "
        "(consideration).\n**Why (evidence):**\n- Upper-lobe predilection and "
        "cavitation are classic for reactivation TB.\n- The trained chest classifier "
        "supports TB at high confidence (88%).\n**Caveats:**\n- Confirm with sputum "
        "AFB / culture or NAAT; imaging alone is not diagnostic.\n\n"
        "Observations:\n- Detected modality: brain MRI.\n- Vision narration: "
        "well-demarcated T2-hyperintense mass with peritumoral oedema.\n- Similar "
        "cases describe: enhancing intra-axial mass.\n"
        "→\n**Assessment:** A neoplastic intra-axial mass (e.g. glioma) is the "
        "leading consideration.\n**Why (evidence):**\n- A well-demarcated "
        "T2-hyperintense mass with surrounding vasogenic oedema fits a tumour.\n- "
        "Similar library cases show comparable enhancing intra-axial masses.\n"
        "**Caveats:**\n- Contrast sequences and histopathology are needed to "
        "characterise and grade it."
    )
    user = fewshot + "\n\nObservations:\n" + "\n".join(f"- {o}" for o in obs) + "\n→"
    return chat(cfg, system, user, max_new_tokens=340)


def answer_detailed(cfg, question, passages=None, context=""):
    """A COMPREHENSIVE, multi-section answer (for 'give a detailed report/summary').
    Grounded in passages when provided; otherwise educational. Longer output."""
    ctx = f"Conversation so far: {context}\n\n" if context else ""
    if passages:
        src = "\n\n".join(f"[p{p['page']}] {p['text']}" for p in passages)
        system = (
            "You are a medical research assistant. Write a DETAILED, well-structured "
            "report grounded ONLY in the passages, using Markdown section headers "
            "(**Overview**, **Key Findings**, **Mechanisms/Details**, **Metrics & "
            "Dosages**, **Clinical Implications**, **Limitations**). Cite pages like "
            "(p.3). Be thorough but never invent facts not in the passages."
        )
        user = f"{ctx}Sources:\n{src}\n\nWrite a detailed report on: {question}"
    else:
        system = (
            "You are a medical education assistant. Write a DETAILED, comprehensive, "
            "well-structured explanation in Markdown with clear section headers and "
            "bullet points (definition, pathophysiology, subtypes/grading, imaging "
            "features, clinical presentation, diagnosis, management, prognosis where "
            "relevant). Use correct terminology, finish every sentence. Educational only."
        )
        user = f"{ctx}Give a detailed, multi-section explanation of: {question}"
    return chat(cfg, system, user, max_new_tokens=900)


def answer_document(cfg, question, passages):
    """Answer GROUNDED in retrieved document passages (RAG), with page citations."""
    context = "\n\n".join(f"[p{p['page']}] {p['text']}" for p in passages)
    system = (
        "You are a research assistant for a medical expert. Answer ONLY from the "
        "provided passages, using precise technical/quantitative detail (exact "
        "values, dosages, endpoints). Cite page numbers inline like (p.3). If the "
        "answer is not in the passages, reply exactly 'Not found in the document.' "
        "Never invent numbers, dosages, or findings. Format in clean Markdown."
    )
    user = f"Passages:\n{context}\n\nQuestion: {question}\n\nGrounded answer:"
    return chat(cfg, system, user, max_new_tokens=260)


def generate_document_report(cfg, passages):
    """Structured research report grounded ONLY in the passages, page-cited."""
    context = "\n\n".join(f"[p{p['page']}] {p['text']}" for p in passages)
    system = (
        "You are a research assistant for a medical expert. Using ONLY the passages, "
        "write a structured report with these exact **bold** section headers, each "
        "concise and technical:\n"
        "**Objective**, **Methods**, **Key Findings**, **Metrics & Dosages** "
        "(quote exact numeric values/units), **Limitations**.\n"
        "Cite page numbers inline like (p.3). If a section is not covered, write "
        "'Not stated in document'. Never invent numbers, dosages, or findings."
    )
    user = f"Passages:\n{context}\n\nStructured report:"
    return chat(cfg, system, user, max_new_tokens=460)


def compose_answer(cfg, question, vqa_answer, caption):
    """Grounded, technical one-paragraph answer about an image."""
    user = (
        f"Question: {question}\n"
        f"Predicted category: {vqa_answer}\n"
        f"Vision-model description: {caption}\n\n"
        "Answer the question in 1–3 sentences using precise clinical terminology, "
        "grounded ONLY in the above. Frame as a consideration, not a diagnosis. "
        "Do not add unsupported facts."
    )
    return chat(cfg, SYSTEM, user, max_new_tokens=220)


if __name__ == "__main__":
    cfg = load_config()
    print(compose_report(cfg,
                         caption="a stained tissue section with dense cellular regions",
                         vqa_answer="adenocarcinoma",
                         concepts=["glandular structures", "atypia"]))
