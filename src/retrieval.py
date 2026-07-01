"""Component: RETRIEVAL backbone — "answer with images".

Builds a BiomedCLIP image bank from PathVQA, then answers a TEXT query by
returning the top-k most similar images (cosine similarity in the shared
BiomedCLIP space). This is the backbone of the chat engine's image answers.

Subcommands:
  prepare : build the image bank (dedup unique images, embed, save thumbnails
            + embeddings + metadata). CPU-sized via config bank_size.
  query   : ad-hoc text query -> top-k images (prints paths + scores).
  eval    : REAL Recall@1/5/10 for text->image retrieval on a held-out gallery.

Run:  python -m src.retrieval prepare | "query:your text here" | eval | all
"""
from __future__ import annotations

import hashlib
import json
import sys

import numpy as np
import torch

from src.common import load_config, resolve, ensure_dir, set_seed
from src.biomedclip import load_biomedclip, encode_images, encode_texts


def _img_hash(img, size=64) -> str:
    """Stable hash of an image (dedup PathVQA's repeated images)."""
    small = img.convert("L").resize((size, size))
    return hashlib.md5(small.tobytes()).hexdigest()


def _bank_paths(cfg):
    bank_dir = ensure_dir(resolve(cfg["paths"]["data"], "image_bank"))
    emb_file = resolve(cfg["paths"]["weights"], "retrieval_bank.npz")
    meta_file = resolve(cfg["paths"]["weights"], "retrieval_bank.json")
    ensure_dir(resolve(cfg["paths"]["weights"]))
    return bank_dir, emb_file, meta_file


# ---------------------------------------------------------------------------
def prepare(cfg):
    """Build the image bank: dedup unique images, embed with BiomedCLIP, save."""
    from datasets import load_dataset

    rcfg = cfg["retrieval"]
    bank_dir, emb_file, meta_file = _bank_paths(cfg)
    model, preprocess, tokenizer, device = load_biomedclip(rcfg["clip_model"])

    n_target = rcfg["bank_size"]
    thumb = rcfg["thumb_size"]
    print(f"[prepare] Building image bank of {n_target} unique PathVQA images...")

    ds = load_dataset(rcfg["dataset"], split="train")
    seen = set()
    images, metas = [], []
    for ex in ds:
        img = ex["image"]
        h = _img_hash(img)
        if h in seen:
            # accumulate extra question text onto the existing entry for context
            continue
        seen.add(h)
        idx = len(images)
        thumb_img = img.convert("RGB").resize((thumb, thumb))
        rel = f"img_{idx:05d}.jpg"
        thumb_img.save(bank_dir / rel, quality=85)
        images.append(thumb_img)
        metas.append({"file": rel, "question": ex["question"], "answer": ex["answer"]})
        if len(images) >= n_target:
            break

    print(f"[prepare] {len(images)} unique images saved to {bank_dir}")
    print("[prepare] Embedding bank with BiomedCLIP (CPU)...")
    embs = []
    B = 32
    for i in range(0, len(images), B):
        embs.append(encode_images(model, preprocess, device, images[i:i + B]))
        print(f"  embedded {min(i+B, len(images))}/{len(images)}", end="\r")
    embs = torch.cat(embs).numpy().astype("float32")
    print()

    np.savez_compressed(emb_file, embeddings=embs)
    with open(meta_file, "w", encoding="utf-8") as f:
        json.dump(metas, f)
    print(f"[prepare] Saved embeddings {embs.shape} -> {emb_file}")
    print(f"[prepare] Saved metadata ({len(metas)}) -> {meta_file}")


# ---------------------------------------------------------------------------
def prepare_radiology(cfg):
    """Append REAL radiology images (ROCOv2: chest/CT/MRI/brain) to the bank so
    radiology queries have hits. Embeds with BiomedCLIP and appends to the
    existing pathology bank (does NOT re-embed pathology)."""
    from datasets import load_dataset

    rcfg = cfg["retrieval"]
    bank_dir, emb_file, meta_file = _bank_paths(cfg)
    model, preprocess, tokenizer, device = load_biomedclip(rcfg["clip_model"])

    if not emb_file.exists():
        raise FileNotFoundError("Build the pathology bank first (prepare).")
    bank, metas = load_bank(cfg)
    # drop any previous radiology rows so this is idempotent
    keep = [i for i, m in enumerate(metas) if m.get("source") != "radiology"]
    bank = bank[keep]; metas = [metas[i] for i in keep]
    start = len([m for m in metas])

    n = rcfg["radiology_subset"]
    thumb = rcfg["thumb_size"]
    print(f"[radiology] Adding {n} ROCOv2 radiology images to the bank...")
    ds = load_dataset(rcfg["radiology_dataset"], split="train", streaming=True)
    imgs, new_metas, seen = [], [], set()
    for ex in ds:
        cap = (ex.get("caption") or "").strip()
        if not cap:
            continue
        h = _img_hash(ex["image"])
        if h in seen:
            continue
        seen.add(h)
        idx = start + len(imgs)
        rel = f"rad_{idx:05d}.jpg"
        thumb_img = ex["image"].convert("RGB").resize((thumb, thumb))
        thumb_img.save(bank_dir / rel, quality=85)
        imgs.append(thumb_img)
        new_metas.append({"file": rel, "question": cap[:80],
                          "answer": "radiology", "source": "radiology"})
        if len(imgs) >= n:
            break

    print(f"[radiology] Embedding {len(imgs)} radiology images (CPU)...")
    embs, B = [], 32
    for i in range(0, len(imgs), B):
        embs.append(encode_images(model, preprocess, device, imgs[i:i + B]))
        print(f"  {min(i+B, len(imgs))}/{len(imgs)}", end="\r")
    print()
    rad_embs = torch.cat(embs).numpy().astype("float32")

    combined = np.concatenate([bank.numpy().astype("float32"), rad_embs], axis=0)
    metas = metas + new_metas
    np.savez_compressed(emb_file, embeddings=combined)
    with open(meta_file, "w", encoding="utf-8") as f:
        json.dump(metas, f)
    print(f"[radiology] Bank now {combined.shape[0]} images "
          f"({start} pathology + {len(imgs)} radiology)")


def load_bank(cfg):
    """Load saved bank embeddings + metadata for querying."""
    _, emb_file, meta_file = _bank_paths(cfg)
    embs = np.load(emb_file)["embeddings"]
    with open(meta_file, "r", encoding="utf-8") as f:
        metas = json.load(f)
    return torch.from_numpy(embs), metas


def query(cfg, text, topk=5):
    """Return top-k (meta, score) for a text query."""
    rcfg = cfg["retrieval"]
    model, preprocess, tokenizer, device = load_biomedclip(rcfg["clip_model"])
    bank, metas = load_bank(cfg)
    q = encode_texts(model, tokenizer, device, [text])  # [1, 512], normalized
    sims = (q @ bank.T).squeeze(0)                       # cosine (both normalized)
    vals, idx = sims.topk(min(topk, len(metas)))
    results = [(metas[i], float(vals[k])) for k, i in enumerate(idx.tolist())]
    return results


# ---------------------------------------------------------------------------
def evaluate(cfg):
    """REAL Recall@k for text->image retrieval on a held-out gallery.

    Protocol: take `eval_gallery` UNIQUE validation images as the gallery.
    Each image's query text = "question answer" (PathVQA has no captions).
    For each query, rank all gallery images by cosine similarity; Recall@k =
    fraction of queries whose OWN image is in the top-k. This is a standard,
    honest text->image retrieval metric (chance = k / gallery_size).
    """
    from datasets import load_dataset

    rcfg = cfg["retrieval"]
    model, preprocess, tokenizer, device = load_biomedclip(rcfg["clip_model"])
    n = rcfg["eval_gallery"]
    topks = rcfg["topk"]

    print(f"[eval] Loading {n} unique validation images as gallery...")
    ds = load_dataset(rcfg["dataset"], split="validation")
    seen = set()
    imgs, texts = [], []
    for ex in ds:
        h = _img_hash(ex["image"])
        if h in seen:
            continue
        seen.add(h)
        imgs.append(ex["image"].convert("RGB"))
        texts.append(f"{ex['question']} {ex['answer']}".strip())
        if len(imgs) >= n:
            break

    print(f"[eval] Embedding {len(imgs)} images + {len(texts)} queries (CPU)...")
    img_embs, B = [], 32
    for i in range(0, len(imgs), B):
        img_embs.append(encode_images(model, preprocess, device, imgs[i:i + B]))
        print(f"  images {min(i+B, len(imgs))}/{len(imgs)}", end="\r")
    img_embs = torch.cat(img_embs)
    print()
    txt_embs = []
    for i in range(0, len(texts), B):
        txt_embs.append(encode_texts(model, tokenizer, device, texts[i:i + B]))
    txt_embs = torch.cat(txt_embs)

    sims = txt_embs @ img_embs.T              # [Q, G]
    ranks = sims.argsort(dim=1, descending=True)
    gt = torch.arange(len(texts)).unsqueeze(1)
    recalls = {}
    for k in topks:
        hit = (ranks[:, :k] == gt).any(dim=1).float().mean().item()
        recalls[k] = hit

    print("=" * 60)
    print(f"  REAL text->image retrieval on {len(imgs)} held-out images (CPU)")
    for k in topks:
        chance = k / len(imgs)
        print(f"  Recall@{k:<2d} = {recalls[k]*100:5.1f}%   (chance {chance*100:4.1f}%)")
    print("  Query text = PathVQA 'question answer' (no captions in dataset).")
    print("=" * 60)

    metrics_file = resolve(cfg["paths"]["outputs"], "retrieval", "recall.txt")
    ensure_dir(metrics_file.parent)
    with open(metrics_file, "w", encoding="utf-8") as f:
        f.write(f"gallery_images={len(imgs)}\n")
        for k in topks:
            f.write(f"Recall@{k}={recalls[k]*100:.2f}%\n")
        f.write("device=cpu\n")
    print(f"[eval] Wrote {metrics_file}")
    return recalls


# ---------------------------------------------------------------------------
def evaluate_concepts(cfg):
    """REAL concept-relevance retrieval — the metric that matches the product.

    "Show me examples of X": success = a RELEVANT image (sharing concept X) is in
    the top-k, NOT the one specific paired image. We aggregate each validation
    image's open-ended answers into a concept set, pick specific concepts (not
    too rare, not too generic), query "histopathology image, H&E stain, showing
    {concept}", and measure hit@k = top-k contains >=1 image with that concept.
    """
    from collections import defaultdict, Counter
    from datasets import load_dataset

    rcfg = cfg["retrieval"]
    model, preprocess, tokenizer, device = load_biomedclip(rcfg["clip_model"])
    n = rcfg["eval_gallery"]
    topks = rcfg["topk"]

    print(f"[concepts] Building gallery of {n} unique val images + concept sets...")
    ds = load_dataset(rcfg["dataset"], split="validation")
    order, imgs, concepts = [], {}, defaultdict(set)
    GENERIC = {"yes", "no", "tissue", "image", "nothing", "normal", "present"}
    for ex in ds:
        h = _img_hash(ex["image"])
        if h not in imgs:
            imgs[h] = ex["image"].convert("RGB"); order.append(h)
        a = ex["answer"].strip().lower()
        if a not in GENERIC and 3 <= len(a) <= 40:
            concepts[h].add(a)
        if len(order) >= n:
            break

    gallery = [imgs[h] for h in order]
    concept_sets = [concepts[h] for h in order]

    # pick query concepts: specific (in 1%-25% of images)
    freq = Counter(c for s in concept_sets for c in s)
    lo, hi = max(2, int(0.01 * len(order))), int(0.25 * len(order))
    query_concepts = [c for c, f in freq.most_common() if lo <= f <= hi][:40]
    print(f"[concepts] {len(query_concepts)} query concepts over {len(gallery)} images")

    print("[concepts] Embedding gallery (CPU)...")
    g_embs, B = [], 32
    for i in range(0, len(gallery), B):
        g_embs.append(encode_images(model, preprocess, device, gallery[i:i + B]))
        print(f"  {min(i+B, len(gallery))}/{len(gallery)}", end="\r")
    g_embs = torch.cat(g_embs)
    print()
    queries = [f"histopathology image, H&E stain, showing {c}" for c in query_concepts]
    q_embs = encode_texts(model, tokenizer, device, queries)

    sims = q_embs @ g_embs.T
    ranks = sims.argsort(1, descending=True)
    hits = {k: 0 for k in topks}
    for qi, c in enumerate(query_concepts):
        topk_idx = ranks[qi, :max(topks)].tolist()
        for k in topks:
            if any(c in concept_sets[j] for j in topk_idx[:k]):
                hits[k] += 1

    print("=" * 60)
    print(f"  REAL concept-relevance retrieval ({len(query_concepts)} concepts, "
          f"{len(gallery)} images, CPU)")
    for k in topks:
        print(f"  Hit@{k:<2d} = {hits[k]/len(query_concepts)*100:5.1f}%   "
              f"(found a relevant image in top-{k})")
    print("  Success = a RELEVANT image (shares the concept) appears in top-k —")
    print("  this is the metric that matches 'show me examples of X'.")
    print("=" * 60)

    mfile = resolve(cfg["paths"]["outputs"], "retrieval", "concept_recall.txt")
    ensure_dir(mfile.parent)
    with open(mfile, "w", encoding="utf-8") as f:
        f.write(f"query_concepts={len(query_concepts)}\ngallery_images={len(gallery)}\n")
        for k in topks:
            f.write(f"Hit@{k}={hits[k]/len(query_concepts)*100:.2f}%\n")
        f.write("device=cpu\n")
    print(f"[concepts] Wrote {mfile}")
    return hits


def main():
    cfg = load_config()
    set_seed(cfg["seed"])
    arg = sys.argv[1] if len(sys.argv) > 1 else "all"
    if arg == "concepts":
        evaluate_concepts(cfg)
        return
    if arg == "radiology":
        prepare_radiology(cfg)
        return
    if arg.startswith("query:"):
        for meta, score in query(cfg, arg[len("query:"):], topk=5):
            print(f"  {score:.3f}  {meta['file']}  | Q: {meta['question']}")
        return
    if arg in ("prepare", "all"):
        prepare(cfg)
    if arg in ("eval", "all"):
        evaluate(cfg)
    if arg not in ("prepare", "eval", "all"):
        print('usage: python -m src.retrieval [prepare|eval|all|"query:TEXT"]')


if __name__ == "__main__":
    main()
