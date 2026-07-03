"""Supervised FINDING classifiers on frozen BiomedCLIP image embeddings.

Zero-shot finding detection is ~chance (unreliable). A small SUPERVISED head on
labeled data is accurate: e.g. 3-class chest (TB/COVID/Pneumonia) ~94% held-out.
This module trains + serves such heads per modality. Findings for modalities
WITHOUT a trained head fall back to an honest "inconclusive" in the app.

All CPU, embeddings precomputed (reuses the retrieval bank).
"""
from __future__ import annotations

import functools
import json

import numpy as np
import torch
import torch.nn as nn

from src.common import resolve, ensure_dir, set_seed


def _head(n_classes):
    return nn.Sequential(nn.Linear(512, 128), nn.ReLU(), nn.Dropout(0.3),
                         nn.Linear(128, n_classes))


def _paths(modality_key):
    d = ensure_dir(resolve("weights", "findings"))
    return d / f"{modality_key}_head.pt", d / f"{modality_key}_classes.json", \
        d / f"{modality_key}_acc.txt"


def train_chest():
    """Train the chest finding head from the chest rows already in the bank."""
    set_seed(42)
    emb = np.load(resolve("weights", "retrieval_bank.npz"))["embeddings"]
    meta = json.loads(resolve("weights", "retrieval_bank.json").read_text(encoding="utf-8"))
    idx = [i for i, m in enumerate(meta) if m.get("source") == "tb"]
    if not idx:
        print("[findings] no chest rows in bank; run: python -m src.retrieval chest")
        return
    X = torch.tensor(emb[idx])
    labs = [meta[i]["answer"] for i in idx]
    classes = sorted(set(labs))
    c2i = {c: k for k, c in enumerate(classes)}
    y = torch.tensor([c2i[l] for l in labs])

    g = torch.Generator().manual_seed(0)
    perm = torch.randperm(len(y), generator=g)
    ntr = int(0.8 * len(y)); tr, te = perm[:ntr], perm[ntr:]
    head = _head(len(classes))
    opt = torch.optim.AdamW(head.parameters(), lr=1e-3, weight_decay=1e-3)
    lf = nn.CrossEntropyLoss()
    for ep in range(120):
        head.train(); p = tr[torch.randperm(len(tr))]
        for i in range(0, len(tr), 32):
            b = p[i:i + 32]; opt.zero_grad(); lf(head(X[b]), y[b]).backward(); opt.step()
    head.eval()
    with torch.no_grad():
        acc = (head(X[te]).argmax(1) == y[te]).float().mean().item() * 100

    hf, cf, af = _paths("chest")
    torch.save(head.state_dict(), hf)
    cf.write_text(json.dumps(classes), encoding="utf-8")
    af.write_text(f"chest_finding_accuracy={acc:.2f}%\nclasses={classes}\n"
                  f"n={len(y)}\ndevice=cpu\n", encoding="utf-8")
    print(f"[findings] chest head saved. Held-out accuracy: {acc:.2f}% "
          f"over {classes}")
    return acc


@functools.lru_cache(maxsize=4)
def _load(modality_key):
    hf, cf, _ = _paths(modality_key)
    if not (hf.exists() and cf.exists()):
        return None
    classes = json.loads(cf.read_text(encoding="utf-8"))
    head = _head(len(classes)); head.load_state_dict(torch.load(hf)); head.eval()
    return head, classes


def available(modality_key) -> bool:
    hf, cf, _ = _paths(modality_key)
    return hf.exists() and cf.exists()


@torch.no_grad()
def predict(modality_key, image_embedding, topn=3):
    """Return top-N (finding, prob) from the trained head, or None if untrained."""
    loaded = _load(modality_key)
    if loaded is None:
        return None
    head, classes = loaded
    x = image_embedding if image_embedding.ndim == 2 else image_embedding.unsqueeze(0)
    probs = head(x).softmax(1)[0]
    order = probs.argsort(descending=True)[:topn].tolist()
    return [(classes[i], float(probs[i])) for i in order]


if __name__ == "__main__":
    train_chest()
