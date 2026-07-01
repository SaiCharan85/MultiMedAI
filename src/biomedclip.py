"""Shared BiomedCLIP loader — used by retrieval, VQA, and report components.

BiomedCLIP (microsoft/BiomedCLIP-PubMedBERT_256-vit_base_patch16_224) produces
ALIGNED image and text embeddings in ONE shared 512-d space:
  - image encoder: ViT-B/16
  - text encoder : PubMedBERT
Trained on biomedical literature (PMC), so it separates clinical features that
web-trained vanilla CLIP misreads. We use it FROZEN (no training).

All ops are CPU (see src/common.py for why).
"""
from __future__ import annotations

import functools

import torch

from src.common import get_device


@functools.lru_cache(maxsize=1)
def load_biomedclip(model_name: str):
    """Load BiomedCLIP once (cached). Returns (model, preprocess, tokenizer, device)."""
    import open_clip

    device = get_device()
    model, preprocess = open_clip.create_model_from_pretrained(model_name)
    tokenizer = open_clip.get_tokenizer(model_name)
    model.eval().to(device)
    return model, preprocess, tokenizer, device


@torch.no_grad()
def encode_images(model, preprocess, device, pil_images) -> torch.Tensor:
    """Encode a list of PIL images -> L2-normalized embeddings [N, 512]."""
    batch = torch.stack([preprocess(im.convert("RGB")) for im in pil_images]).to(device)
    feats = model.encode_image(batch)
    feats = feats / feats.norm(dim=-1, keepdim=True)
    return feats.cpu()


@torch.no_grad()
def encode_texts(model, tokenizer, device, texts) -> torch.Tensor:
    """Encode a list of strings -> L2-normalized embeddings [N, 512]."""
    toks = tokenizer(list(texts)).to(device)
    feats = model.encode_text(toks)
    feats = feats / feats.norm(dim=-1, keepdim=True)
    return feats.cpu()
