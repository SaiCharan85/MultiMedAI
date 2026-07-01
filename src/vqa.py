"""Component: VQA — BiomedCLIP (frozen) + small trainable head.

Embeddings: BiomedCLIP gives ALIGNED image and text embeddings in one shared
512-d space. The head input is concat(image_emb, question_emb) = 1024-d, so the
answer depends on BOTH the image and the question. The encoder is FROZEN; only
the small MLP head trains (feasible on CPU because embeddings are precomputed
once and images are dedup-cached).

Answer space: closed vocabulary of the top-`max_answers` most frequent training
answers (classification head). Eval reports REAL accuracy, split into yes/no vs
open-ended (the honest breakdown — yes/no has a ~50% baseline).

Run:  python -m src.vqa train | eval | all
      python -m src.vqa "ask:PATH_TO_IMAGE::your question"
"""
from __future__ import annotations

import hashlib
import json
import sys

import torch
import torch.nn as nn

from src.common import load_config, resolve, ensure_dir, get_device, set_seed
from src.biomedclip import load_biomedclip, encode_images, encode_texts


def _img_hash(img):
    return hashlib.md5(img.convert("L").resize((64, 64)).tobytes()).hexdigest()


def _head(in_dim, n_classes):
    return nn.Sequential(
        nn.Linear(in_dim, 512), nn.ReLU(), nn.Dropout(0.5), nn.Linear(512, n_classes)
    )


def _paths(cfg):
    d = ensure_dir(resolve(cfg["paths"]["weights"], "vqa"))
    return d / "vqa_head.pt", d / "vqa_vocab.json"


def _build_vocab(cfg):
    from collections import Counter
    from datasets import load_dataset

    ds = load_dataset(cfg["vqa"]["dataset"], split="train")
    freq = Counter(ex["answer"].strip().lower() for ex in ds)
    return [a for a, _ in freq.most_common(cfg["vqa"]["max_answers"])]


def _featurize(cfg, split, limit, a2i, model, preprocess, tokenizer, device):
    """Return X=[N,1024] (image+question embeds), y=[N], and the yes/no mask."""
    from datasets import load_dataset

    ds = load_dataset(cfg["vqa"]["dataset"], split=split)
    uniq_imgs, img_idx_of, questions, labels = [], [], [], []
    hash_to_idx = {}
    for ex in ds:
        a = ex["answer"].strip().lower()
        if a not in a2i:
            continue
        h = _img_hash(ex["image"])
        if h not in hash_to_idx:
            hash_to_idx[h] = len(uniq_imgs)
            uniq_imgs.append(ex["image"])
        img_idx_of.append(hash_to_idx[h])
        questions.append(ex["question"])
        labels.append(a2i[a])
        if len(labels) >= limit:
            break

    print(f"  [{split}] {len(labels)} examples over {len(uniq_imgs)} unique images; embedding...")
    # embed unique images once (the expensive part on CPU), then index
    img_embs, B = [], 32
    for i in range(0, len(uniq_imgs), B):
        img_embs.append(encode_images(model, preprocess, device, uniq_imgs[i:i + B]))
        print(f"    images {min(i+B,len(uniq_imgs))}/{len(uniq_imgs)}", end="\r")
    img_embs = torch.cat(img_embs)
    print()
    q_embs = []
    for i in range(0, len(questions), B):
        q_embs.append(encode_texts(model, tokenizer, device, questions[i:i + B]))
    q_embs = torch.cat(q_embs)

    img_per_ex = img_embs[torch.tensor(img_idx_of)]
    X = torch.cat([img_per_ex, q_embs], dim=1)
    y = torch.tensor(labels)
    return X, y


def _featurize_cached(cfg, a2i, model, preprocess, tokenizer, device):
    """Featurize train/val once and cache to disk so re-training is instant."""
    vcfg = cfg["vqa"]
    tag = vcfg["clip_model"].split("/")[-1].replace(":", "_")
    cache = resolve(cfg["paths"]["weights"], "vqa",
                    f"_feat_{tag}_{vcfg['train_subset']}_{vcfg['val_subset']}_{len(a2i)}.pt")
    if cache.exists():
        print(f"[train] Loading cached features {cache.name}")
        d = torch.load(cache)
        return d["Xtr"], d["ytr"], d["Xva"], d["yva"]
    print("[train] Featurizing train/val (BiomedCLIP embeddings, CPU)...")
    Xtr, ytr = _featurize(cfg, "train", vcfg["train_subset"], a2i,
                          model, preprocess, tokenizer, device)
    Xva, yva = _featurize(cfg, "test", vcfg["val_subset"], a2i,
                          model, preprocess, tokenizer, device)
    ensure_dir(cache.parent)
    torch.save({"Xtr": Xtr, "ytr": ytr, "Xva": Xva, "yva": yva}, cache)
    return Xtr, ytr, Xva, yva


def train(cfg):
    set_seed(cfg["seed"])
    vcfg = cfg["vqa"]
    model, preprocess, tokenizer, device = load_biomedclip(vcfg["clip_model"])

    print("[train] Building answer vocabulary...")
    vocab = _build_vocab(cfg)
    a2i = {a: i for i, a in enumerate(vocab)}
    print(f"[train] vocab={len(vocab)}; top: {vocab[:8]}")

    Xtr, ytr, Xva, yva = _featurize_cached(cfg, a2i, model, preprocess, tokenizer, device)
    print(f"[train] train={tuple(Xtr.shape)} val={tuple(Xva.shape)}")

    # MILD class weighting (sqrt inverse-freq): nudges the head toward rare
    # open-ended answers without the instability that full inverse-freq caused.
    counts = torch.bincount(ytr, minlength=len(vocab)).float()
    weights = (1.0 / (counts + 1.0)).sqrt()
    weights = weights / weights.mean()
    weights = weights.clamp(max=5.0)          # cap so rare classes can't explode

    head = _head(Xtr.shape[1], len(vocab))
    opt = torch.optim.AdamW(head.parameters(), lr=vcfg["lr"], weight_decay=1e-3)
    lossf = nn.CrossEntropyLoss(weight=weights, label_smoothing=0.05)

    bs = 64
    n = Xtr.shape[0]
    print(f"[train] Mini-batch training: {vcfg['epochs']} epochs x ~{n//bs} batches...")
    g = torch.Generator().manual_seed(cfg["seed"])
    for epoch in range(vcfg["epochs"]):
        head.train()
        perm = torch.randperm(n, generator=g)
        for i in range(0, n, bs):
            idx = perm[i:i + bs]
            opt.zero_grad()
            loss = lossf(head(Xtr[idx]), ytr[idx]); loss.backward(); opt.step()
        if (epoch + 1) % 10 == 0:
            head.eval()
            with torch.no_grad():
                pred = head(Xva).argmax(1)
                acc = (pred == yva).float().mean().item() * 100
                yn = torch.tensor([vocab[y] in ("yes", "no") for y in yva.tolist()])
                oe = (pred[~yn] == yva[~yn]).float().mean().item() * 100
            print(f"  epoch {epoch+1:3d}  loss {loss.item():.3f}  "
                  f"val_acc {acc:.2f}%  open-ended {oe:.2f}%")

    head_file, vocab_file = _paths(cfg)
    torch.save(head.state_dict(), head_file)
    with open(vocab_file, "w", encoding="utf-8") as f:
        json.dump(vocab, f)
    print(f"[train] Saved head -> {head_file}")
    # stash val features for eval without recompute
    torch.save({"Xva": Xva, "yva": yva}, resolve(cfg["paths"]["weights"], "vqa", "_val_cache.pt"))
    return head, vocab, Xva, yva


def evaluate(cfg):
    vcfg = cfg["vqa"]
    head_file, vocab_file = _paths(cfg)
    cache = resolve(cfg["paths"]["weights"], "vqa", "_val_cache.pt")
    if not (head_file.exists() and cache.exists()):
        print("[eval] No trained head/cache. Run: python -m src.vqa train")
        return None
    vocab = json.loads(vocab_file.read_text(encoding="utf-8"))
    head = _head(1024, len(vocab)); head.load_state_dict(torch.load(head_file)); head.eval()
    d = torch.load(cache); Xva, yva = d["Xva"], d["yva"]

    with torch.no_grad():
        pred = head(Xva).argmax(1)
    correct = (pred == yva)
    yn_mask = torch.tensor([vocab[y] in ("yes", "no") for y in yva.tolist()])
    overall = correct.float().mean().item() * 100
    yn = correct[yn_mask].float().mean().item() * 100 if yn_mask.any() else float("nan")
    oe = correct[~yn_mask].float().mean().item() * 100 if (~yn_mask).any() else float("nan")

    print("=" * 56)
    print(f"  REAL VQA accuracy (val n={len(yva)}, CPU, frozen BiomedCLIP + head)")
    print(f"  overall    = {overall:5.2f}%")
    print(f"  yes/no     = {yn:5.2f}%   (n={int(yn_mask.sum())}, ~50% chance)")
    print(f"  open-ended = {oe:5.2f}%   (n={int((~yn_mask).sum())}, {len(vocab)}-way)")
    print("=" * 56)

    mfile = resolve(cfg["paths"]["outputs"], "vqa", "accuracy.txt")
    ensure_dir(mfile.parent)
    with open(mfile, "w", encoding="utf-8") as f:
        f.write(f"val_n={len(yva)}\noverall={overall:.2f}%\n")
        f.write(f"yes_no={yn:.2f}%\nopen_ended={oe:.2f}%\ndevice=cpu\n")
    print(f"[eval] Wrote {mfile}")
    return overall


@torch.no_grad()
def ask(cfg, image_path, question):
    from PIL import Image

    vcfg = cfg["vqa"]
    head_file, vocab_file = _paths(cfg)
    vocab = json.loads(vocab_file.read_text(encoding="utf-8"))
    model, preprocess, tokenizer, device = load_biomedclip(vcfg["clip_model"])
    head = _head(1024, len(vocab)); head.load_state_dict(torch.load(head_file)); head.eval()
    img = Image.open(image_path)
    ie = encode_images(model, preprocess, device, [img])
    qe = encode_texts(model, tokenizer, device, [question])
    logits = head(torch.cat([ie, qe], dim=1))
    probs = logits.softmax(1)[0]
    top = probs.topk(3)
    return [(vocab[i], float(top.values[k])) for k, i in enumerate(top.indices.tolist())]


def main():
    cfg = load_config()
    arg = sys.argv[1] if len(sys.argv) > 1 else "all"
    if arg.startswith("ask:"):
        path, q = arg[len("ask:"):].split("::", 1)
        for ans, p in ask(cfg, path, q):
            print(f"  {p:.3f}  {ans}")
        return
    if arg in ("train", "all"):
        train(cfg)
    if arg in ("eval", "all"):
        evaluate(cfg)
    if arg not in ("train", "eval", "all"):
        print('usage: python -m src.vqa [train|eval|all|"ask:PATH::question"]')


if __name__ == "__main__":
    main()
