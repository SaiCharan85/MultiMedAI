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


SYSTEM = ("You are a careful medical imaging assistant. Use ONLY the observations "
          "you are given. Never invent findings. Be concise and clinical.")


def compose_report(cfg, caption, vqa_answer=None, concepts=None):
    facts = [f"Visual description: {caption}"]
    if vqa_answer:
        facts.append(f"Model's answer: {vqa_answer}")
    if concepts:
        facts.append(f"Related concepts in similar images: {', '.join(concepts)}")
    user = ("Write a concise 2-3 sentence descriptive pathology report from these "
            "observations. Do not invent findings.\n" + "\n".join(facts))
    return chat(cfg, SYSTEM, user)


def compose_answer(cfg, question, vqa_answer, caption):
    user = (f"Question: {question}\nModel's predicted answer: {vqa_answer}\n"
            f"Image description: {caption}\n"
            "Answer the question in one natural sentence grounded in the above. "
            "Do not add facts.")
    return chat(cfg, SYSTEM, user)


if __name__ == "__main__":
    cfg = load_config()
    print(compose_report(cfg,
                         caption="a stained tissue section with dense cellular regions",
                         vqa_answer="adenocarcinoma",
                         concepts=["glandular structures", "atypia"]))
