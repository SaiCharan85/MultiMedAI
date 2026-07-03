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
    """Clinical educational explanation of a concept (e.g. 'what is a glioma?')."""
    system = (
        "You are MultiMedAI, a medical education assistant for students and "
        "clinicians. Explain the concept accurately using correct medical "
        "terminology, structured in Markdown: a one-line **definition**, then 2–4 "
        "concise bullets (e.g. pathophysiology, key features, clinical relevance). "
        "Be factual and neutral. This is educational information, not medical advice."
    )
    return chat(cfg, system, question, max_new_tokens=260)


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
