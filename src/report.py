"""Component: REPORT / CAPTION — image -> short text.

Inference: BLIP-base (Salesforce/blip-image-captioning-base) generates a short
description for an image. In the chat engine this powers "describe this image"
and "write a report" answers.

Embeddings note: BLIP has its own vision encoder + text decoder (its own token
embedding space) — this is a SEPARATE space from BiomedCLIP's shared space and
from SD's CLIP space. We do not merge them.

Eval: PathVQA has NO gold captions. So we compute REAL BLEU/ROUGE against a
PROXY reference built from descriptive ("what ...") Q/A pairs (reference =
answer text). BLIP is general-domain, so scores are LOW and only indicative —
reported honestly, not as a medical-captioning benchmark.

Run:  python -m src.report caption:PATH | eval | all
"""
from __future__ import annotations

import functools
import sys

import torch

from src.common import load_config, resolve, ensure_dir, get_device, set_seed


@functools.lru_cache(maxsize=1)
def _load_blip(model_id):
    from transformers import BlipProcessor, BlipForConditionalGeneration

    device = get_device()
    processor = BlipProcessor.from_pretrained(model_id)
    model = BlipForConditionalGeneration.from_pretrained(model_id).eval().to(device)
    return processor, model, device


@torch.no_grad()
def caption_image(cfg, pil_image, max_new_tokens=None) -> str:
    rcfg = cfg["report"]
    processor, model, device = _load_blip(rcfg["model_id"])
    mnt = max_new_tokens or rcfg["max_new_tokens"]
    inputs = processor(pil_image.convert("RGB"), return_tensors="pt").to(device)
    out = model.generate(**inputs, max_new_tokens=mnt)
    return processor.decode(out[0], skip_special_tokens=True).strip()


def evaluate(cfg):
    """REAL BLEU/ROUGE on a held-out PathVQA subset vs proxy references."""
    import evaluate as hf_evaluate
    from datasets import load_dataset

    rcfg = cfg["report"]
    n = rcfg["eval_subset"]
    print(f"[eval] Loading {n} descriptive PathVQA examples (proxy refs)...")
    ds = load_dataset(rcfg["dataset"], split="test")

    preds, refs = [], []
    for ex in ds:
        q, a = ex["question"].strip().lower(), ex["answer"].strip()
        # descriptive questions only; skip yes/no so the proxy reference is content
        if a.lower() in ("yes", "no") or not q.startswith(("what", "where", "how")):
            continue
        cap = caption_image(cfg, ex["image"])
        preds.append(cap)
        refs.append(a)
        if len(preds) % 10 == 0:
            print(f"  captioned {len(preds)}/{n}", end="\r")
        if len(preds) >= n:
            break
    print()

    bleu = hf_evaluate.load("bleu")
    rouge = hf_evaluate.load("rouge")
    bleu_res = bleu.compute(predictions=preds, references=[[r] for r in refs])
    rouge_res = rouge.compute(predictions=preds, references=refs)

    print("=" * 60)
    print(f"  REAL caption metrics on {len(preds)} held-out images (CPU)")
    print(f"  BLEU    = {bleu_res['bleu']*100:5.2f}")
    print(f"  ROUGE-1 = {rouge_res['rouge1']*100:5.2f}")
    print(f"  ROUGE-L = {rouge_res['rougeL']*100:5.2f}")
    print("  Proxy refs = PathVQA answers (no gold captions exist). BLIP is")
    print("  general-domain, so scores are LOW and indicative only.")
    print("=" * 60)

    mfile = resolve(cfg["paths"]["outputs"], "report", "metrics.txt")
    ensure_dir(mfile.parent)
    with open(mfile, "w", encoding="utf-8") as f:
        f.write(f"eval_images={len(preds)}\n")
        f.write(f"BLEU={bleu_res['bleu']*100:.2f}\n")
        f.write(f"ROUGE-1={rouge_res['rouge1']*100:.2f}\n")
        f.write(f"ROUGE-L={rouge_res['rougeL']*100:.2f}\n")
        f.write("refs=proxy(PathVQA answers); model=BLIP-base(general-domain); device=cpu\n")
    print(f"[eval] Wrote {mfile}")
    return bleu_res, rouge_res


def main():
    cfg = load_config()
    set_seed(cfg["seed"])
    arg = sys.argv[1] if len(sys.argv) > 1 else "all"
    if arg.startswith("caption:"):
        from PIL import Image

        print(caption_image(cfg, Image.open(arg[len("caption:"):])))
        return
    if arg in ("eval", "all"):
        evaluate(cfg)
    if arg not in ("eval", "all"):
        print('usage: python -m src.report [eval|all|"caption:PATH"]')


if __name__ == "__main__":
    main()
