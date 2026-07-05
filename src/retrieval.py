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
from PIL import Image

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
    base = bank.numpy().astype("float32")          # ADDITIVE: keep everything
    existing_rad = sum(1 for m in metas if m.get("source") == "radiology")
    target = rcfg["radiology_subset"]
    thumb = rcfg["thumb_size"]
    need = target - existing_rad
    if need <= 0:
        print(f"[radiology] already have {existing_rad} (target {target}) — nothing to do.")
        return
    print(f"[radiology] have {existing_rad}; adding up to {need} more (target {target}). "
          "Checkpointed + retry-safe.")

    new_embs, new_metas, buf, CKPT = [], [], [], 1500

    def flush():
        if buf:
            e = encode_images(model, preprocess, device, buf).numpy().astype("float32")
            new_embs.append(e); buf.clear()

    def save():
        flush()
        if not new_metas:
            return
        combined = np.concatenate([base] + new_embs, axis=0)
        np.savez_compressed(emb_file, embeddings=combined)
        with open(meta_file, "w", encoding="utf-8") as f:
            json.dump(metas + new_metas, f)
        print(f"  [checkpoint] radiology +{len(new_metas)} (bank now {combined.shape[0]})")

    for attempt in range(6):
        try:
            ds = load_dataset(rcfg["radiology_dataset"], split="train", streaming=True)
            got = 0                     # count of caption-bearing rows seen this stream
            for ex in ds:
                cap = (ex.get("caption") or "").strip()
                if not cap:
                    continue
                got += 1
                # resume: skip rows already ingested (prior runs + this run)
                if got <= existing_rad + len(new_metas):
                    continue
                idx = len(metas) + len(new_metas)
                rel = f"rad_{idx:05d}.jpg"
                t = ex["image"].convert("RGB").resize((thumb, thumb))
                t.save(bank_dir / rel, quality=85)
                buf.append(t)
                new_metas.append({"file": rel, "question": " ".join(cap.split())[:300],
                                  "answer": "radiology", "source": "radiology"})
                if len(buf) >= 32:
                    flush()
                if len(new_metas) % CKPT == 0:
                    save()
                if len(new_metas) >= need:
                    break
            break                        # stream finished cleanly
        except Exception as e:           # ROCO CDN can drop; retry, keep progress
            print(f"[radiology] stream error: {str(e)[:80]} — retry {attempt+1}/6, "
                  f"kept {len(new_metas)}")
            save()
    save()
    print(f"[radiology] done. radiology total now {existing_rad + len(new_metas)}.")


# generic non-radiology sources (dermatology, extra brain MRI, …). Each is
# additive + checkpointed + retry-safe, tagged with its own source for provenance.
_HAM = {"mel": "melanoma", "nv": "melanocytic nevus", "bcc": "basal cell carcinoma",
        "akiec": "actinic keratosis / Bowen's disease", "bkl": "benign keratosis",
        "df": "dermatofibroma", "vasc": "vascular lesion"}

EXTRA_SOURCES = [
    {"dataset": "marmal88/skin_cancer", "source": "derm", "add": 6000,
     "cap_field": "dx", "cap_map": _HAM, "template": "dermatology skin lesion, {}"},
    {"dataset": "Falah/Alzheimer_MRI", "source": "brainmri", "add": 4000,
     "cap_field": "label", "template": "brain MRI scan, {} (dementia screening)"},
    {"dataset": "AbishekFranklin/medai-vision-dataset-hair_scalp_conditions",
     "source": "hair", "add": 3000, "cap_field": "label",
     "template": "dermatology — hair and scalp condition: {}"},
    {"dataset": "Hemg/bone-fracture-detection", "source": "bone", "add": 4000,
     "cap_field": "label", "template": "skeletal X-ray, bone / fracture: {}"},
]


def add_source(cfg, spec):
    """Additive + checkpointed + retry ingest of one image dataset into the bank."""
    from datasets import load_dataset

    rcfg = cfg["retrieval"]
    bank_dir, emb_file, meta_file = _bank_paths(cfg)
    model, preprocess, tokenizer, device = load_biomedclip(rcfg["clip_model"])
    bank, metas = load_bank(cfg)
    base = bank.numpy().astype("float32")
    src = spec["source"]
    existing = sum(1 for m in metas if m.get("source") == src)
    need = spec["add"] - existing
    if need <= 0:
        print(f"[{src}] already have {existing}; skip."); return
    thumb = rcfg["thumb_size"]
    label_names = None
    print(f"[{src}] adding up to {need} from {spec['dataset']} (checkpointed)...")

    new_embs, new_metas, buf, CKPT = [], [], [], 1000

    def flush():
        if buf:
            new_embs.append(encode_images(model, preprocess, device, buf)
                            .numpy().astype("float32")); buf.clear()

    def save():
        flush()
        if not new_metas:
            return
        combined = np.concatenate([base] + new_embs, axis=0)
        np.savez_compressed(emb_file, embeddings=combined)
        with open(meta_file, "w", encoding="utf-8") as f:
            json.dump(metas + new_metas, f)
        print(f"  [checkpoint] {src} +{len(new_metas)} (bank {combined.shape[0]})")

    def caption(ex):
        v = ex.get(spec["cap_field"])
        if spec.get("cap_map"):
            v = spec["cap_map"].get(str(v).lower(), str(v))
        elif label_names is not None and isinstance(v, int):
            v = label_names[v]
        return spec.get("template", "{}").format(str(v).replace("_", " ").strip())

    for attempt in range(6):
        try:
            ds = load_dataset(spec["dataset"], split="train", streaming=True)
            try:
                label_names = ds.features[spec["cap_field"]].names
            except Exception:
                label_names = None
            got = 0
            for ex in ds:
                got += 1
                if got <= existing + len(new_metas):
                    continue
                idx = len(metas) + len(new_metas)
                rel = f"{src}_{idx:05d}.jpg"
                ex["image"].convert("RGB").resize((thumb, thumb)).save(bank_dir / rel, quality=85)
                buf.append(Image.open(bank_dir / rel).convert("RGB"))
                new_metas.append({"file": rel, "question": caption(ex),
                                  "answer": src, "source": src})
                if len(buf) >= 32:
                    flush()
                if len(new_metas) % CKPT == 0:
                    save()
                if len(new_metas) >= need:
                    break
            break
        except Exception as e:
            print(f"[{src}] stream error: {str(e)[:70]} — retry {attempt+1}/6")
            save()
    save()
    print(f"[{src}] done. total {src} now {existing + len(new_metas)}.")


def expand_all(cfg):
    """Grow the bank across ALL fields: top up radiology, then add each extra
    source (dermatology, brain MRI, …). Sequential (one bank writer at a time)."""
    from PIL import Image  # noqa: F401  (used inside add_source)
    prepare_radiology(cfg)
    for spec in EXTRA_SOURCES:
        add_source(cfg, spec)
    _bank_dir, _, meta_file = _bank_paths(cfg)
    import collections
    metas = json.loads(meta_file.read_text(encoding="utf-8"))
    print("[expand] FINAL bank:", len(metas),
          dict(collections.Counter(m.get("source", "pathology") for m in metas)))


def prepare_chestxray(cfg):
    """Append a SMALL streamed subset of chest-disease X-rays (TB, COVID,
    Pneumonia) with clear captions. Streaming avoids the 7.7GB full download."""
    from datasets import load_dataset

    rcfg = cfg["retrieval"]
    bank_dir, emb_file, meta_file = _bank_paths(cfg)
    model, preprocess, tokenizer, device = load_biomedclip(rcfg["clip_model"])
    if not emb_file.exists():
        raise FileNotFoundError("Build the pathology bank first (prepare).")
    bank, metas = load_bank(cfg)
    keep = [i for i, m in enumerate(metas) if m.get("source") != "tb"]
    bank = bank[keep]; metas = [metas[i] for i in keep]
    start = len(metas)

    # label index -> caption phrase (from dataset ClassLabel names)
    want = {3: "pulmonary tuberculosis", 0: "COVID-19 pneumonia", 2: "pneumonia"}
    per = rcfg["chest_per_class"]
    thumb = rcfg["thumb_size"]
    counts = {k: 0 for k in want}
    imgs, new_metas = [], []
    print(f"[chest] Streaming up to {per} each of TB/COVID/Pneumonia (resilient)...")
    # This dataset's CDN drops connections; retry a few times and KEEP partial
    # results instead of failing the whole append.
    for attempt in range(5):
        try:
            ds = load_dataset(rcfg["chest_dataset"], split="train", streaming=True)
            for ex in ds:
                lab = ex["label"]
                if lab not in want or counts[lab] >= per:
                    continue
                counts[lab] += 1
                idx = start + len(imgs)
                rel = f"tb_{idx:05d}.jpg"
                t = ex["image"].convert("RGB").resize((thumb, thumb))
                t.save(bank_dir / rel, quality=85)
                imgs.append(t)
                new_metas.append({"file": rel, "question": f"chest X-ray, {want[lab]}",
                                  "answer": want[lab], "source": "tb"})
                if all(counts[k] >= per for k in want):
                    break
            break  # finished cleanly
        except Exception as e:  # noqa: BLE001 - connection drops expected
            print(f"  [attempt {attempt+1}] connection dropped ({type(e).__name__}); "
                  f"kept {len(imgs)} so far, retrying...")
    print(f"[chest] Collected {counts} ({len(imgs)} images)")
    if not imgs:
        print("[chest] Got nothing (network). TB is still covered via ROCO captions.")
        return

    print(f"[chest] Embedding {len(imgs)} chest X-rays (CPU)...")
    embs, B = [], 32
    for i in range(0, len(imgs), B):
        embs.append(encode_images(model, preprocess, device, imgs[i:i + B]))
        print(f"  {min(i+B, len(imgs))}/{len(imgs)}", end="\r")
    print()
    new = torch.cat(embs).numpy().astype("float32")
    combined = np.concatenate([bank.numpy().astype("float32"), new], axis=0)
    metas = metas + new_metas
    np.savez_compressed(emb_file, embeddings=combined)
    with open(meta_file, "w", encoding="utf-8") as f:
        json.dump(metas, f)
    print(f"[chest] Bank now {combined.shape[0]} images (+{len(imgs)} chest X-rays)")


def repair_radiology_captions(cfg):
    """Restore FULL ROCO captions (they were stored at 80 chars) WITHOUT
    re-embedding — re-stream in the same order and overwrite meta text only."""
    from datasets import load_dataset

    _, emb_file, meta_file = _bank_paths(cfg)
    with open(meta_file, encoding="utf-8") as f:
        metas = json.load(f)
    rad_idx = [i for i, m in enumerate(metas) if m.get("source") == "radiology"]
    if not rad_idx:
        print("[repair] no radiology rows."); return
    print(f"[repair] restoring captions for {len(rad_idx)} radiology images...")

    rcfg = cfg["retrieval"]
    ds = load_dataset(rcfg["radiology_dataset"], split="train", streaming=True)
    seen, caps = set(), []
    for ex in ds:
        cap = (ex.get("caption") or "").strip()
        if not cap:
            continue
        h = _img_hash(ex["image"])
        if h in seen:
            continue
        seen.add(h)
        caps.append(" ".join(cap.split())[:300])
        if len(caps) >= len(rad_idx):
            break

    n = min(len(caps), len(rad_idx))
    for j in range(n):
        metas[rad_idx[j]]["question"] = caps[j]
    with open(meta_file, "w", encoding="utf-8") as f:
        json.dump(metas, f)
    print(f"[repair] updated {n} captions (no re-embedding). e.g. {caps[0][:90]}")


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
    if arg == "chest":
        prepare_chestxray(cfg)
        return
    if arg == "expand":
        expand_all(cfg)
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
